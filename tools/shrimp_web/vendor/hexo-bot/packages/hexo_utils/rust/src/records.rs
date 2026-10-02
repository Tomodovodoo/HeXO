//! Compact binary Hexo runner records.
//!
//! The `.hxr` format stores only durable replay-core data. In particular,
//! scenarios are intentionally not encoded; scenario persistence needs a
//! separate versioned contract before it becomes durable.
//!
//! Wire format (schema v1): magic `HEXOREC1`, then a varint-encoded header
//! (schema version, engine rules_version, backend string, player list), then
//! zero or more `G`-marked game payloads. Each payload is length-prefixed and
//! self-contained: game_id, optional zigzag-varint seed, status byte, action
//! ids as u32 LE (the engine's packed (q,r) coordinate ids from
//! hexo_engine legal.rs), optional winner/placements, optional abort record.
//! Appends are flush-per-game, so a crashed run leaves a readable prefix.
//!
//! Consumers: exposed to Python via `pybridge.rs` (`hexo_utils._rust`), then
//! `python/hexo_utils/records.py`, then
//! `packages/hexo_runner/python/hexo_runner/records/record.py` -- the path all
//! model selfplay/evaluation writers and the hexo_frontend dashboard reader
//! use. Bump `HEXO_RECORD_SCHEMA_VERSION` on any wire change: readers reject
//! unknown versions (`RecordError::UnsupportedVersion`).
//!
//! File map: error enum -> header/record structs -> `HexoRecordFile`
//! reader/writer -> `HexoRecordGameWriter` -> free encode/decode helpers ->
//! `PayloadCursor` -> round-trip/corruption tests.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic bytes at the start of every `.hxr` file.
pub const HEXO_RECORD_MAGIC: &[u8; 8] = b"HEXOREC1";
/// Current `.hxr` schema version.
pub const HEXO_RECORD_SCHEMA_VERSION: u64 = 1;

const GAME_MARKER: u8 = b'G';
const STATUS_COMPLETED: u8 = 1;
const STATUS_ABORTED: u8 = 2;

/// Errors produced while reading or writing `.hxr` records.
#[derive(Debug)]
pub enum RecordError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion { found: u64, expected: u64 },
    UnexpectedEof(&'static str),
    VarintTooLarge,
    LengthTooLarge(u64),
    InvalidUtf8(std::string::FromUtf8Error),
    InvalidBool(u8),
    InvalidMarker(u8),
    InvalidStatus(u8),
    TrailingPayloadBytes { trailing: usize },
    ReadOnlyFile,
    // UNUSED(2026-06-12): never constructed anywhere in the crate --
    // iter_records() on a Write-mode file reopens the path for reading
    // instead of returning this error. Kept only for the Display arm and the
    // pybridge error-kind mapping.
    WriteOnlyFile,
    ClosedFile,
    FinishedWriter { game_id: String },
}

impl fmt::Display for RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::InvalidMagic => write!(f, "not a HexoRecordFile"),
            Self::UnsupportedVersion { found, expected } => write!(
                f,
                "unsupported HexoRecordFile schema version: {found} (expected {expected})"
            ),
            Self::UnexpectedEof(context) => {
                write!(f, "unexpected EOF while reading {context}")
            }
            Self::VarintTooLarge => write!(f, "varint is too large"),
            Self::LengthTooLarge(length) => write!(f, "record length does not fit usize: {length}"),
            Self::InvalidUtf8(error) => write!(f, "{error}"),
            Self::InvalidBool(value) => write!(f, "invalid boolean flag: {value}"),
            Self::InvalidMarker(value) => {
                write!(f, "invalid HexoRecordFile game marker: {value:#04x}")
            }
            Self::InvalidStatus(value) => write!(f, "invalid HexoRecord status byte: {value}"),
            Self::TrailingPayloadBytes { trailing } => {
                write!(f, "trailing bytes in HexoRecord payload: {trailing}")
            }
            Self::ReadOnlyFile => write!(f, "cannot write games to a read-only HexoRecordFile"),
            Self::WriteOnlyFile => write!(f, "cannot stream read from a write-only HexoRecordFile"),
            Self::ClosedFile => write!(f, "HexoRecordFile is closed"),
            Self::FinishedWriter { game_id } => {
                write!(f, "record writer for {game_id:?} is already finished")
            }
        }
    }
}

impl std::error::Error for RecordError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidUtf8(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RecordError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<std::string::FromUtf8Error> for RecordError {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Self::InvalidUtf8(error)
    }
}

/// Engine metadata stored once in the file header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HexoRecordEngineMetadata {
    pub rules_version: u64,
    pub backend: String,
}

impl HexoRecordEngineMetadata {
    pub fn new(rules_version: u64, backend: impl Into<String>) -> Self {
        Self {
            rules_version,
            backend: backend.into(),
        }
    }
}

/// Player identity stored once in the file header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HexoRecordPlayer {
    pub player_id: String,
    pub role: String,
    pub label: Option<String>,
}

impl HexoRecordPlayer {
    pub fn new(
        player_id: impl Into<String>,
        role: impl Into<String>,
        label: Option<String>,
    ) -> Self {
        Self {
            player_id: player_id.into(),
            role: role.into(),
            label,
        }
    }
}

/// Abort information for fail-loud runner outcomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbortRecord {
    pub stage: String,
    pub exception_type: String,
    pub message: String,
}

impl AbortRecord {
    pub fn new(
        stage: impl Into<String>,
        exception_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            stage: stage.into(),
            exception_type: exception_type.into(),
            message: message.into(),
        }
    }
}

/// Stored game completion status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HexoRecordStatus {
    Completed,
    Aborted,
}

impl HexoRecordStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Aborted => "aborted",
        }
    }

    fn from_byte(value: u8) -> Result<Self, RecordError> {
        match value {
            STATUS_COMPLETED => Ok(Self::Completed),
            STATUS_ABORTED => Ok(Self::Aborted),
            _ => Err(RecordError::InvalidStatus(value)),
        }
    }

    fn to_byte(self) -> u8 {
        match self {
            Self::Completed => STATUS_COMPLETED,
            Self::Aborted => STATUS_ABORTED,
        }
    }
}

/// Replay-core data for one Hexo game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HexoRecord {
    pub game_id: String,
    pub seed: Option<i64>,
    pub status: HexoRecordStatus,
    pub action_ids: Vec<u32>,
    pub abort: Option<AbortRecord>,
    pub winner: Option<String>,
    pub placements: Option<i64>,
}

impl HexoRecord {
    pub fn completed(
        game_id: impl Into<String>,
        seed: Option<i64>,
        action_ids: Vec<u32>,
        winner: Option<String>,
        placements: i64,
    ) -> Self {
        Self {
            game_id: game_id.into(),
            seed,
            status: HexoRecordStatus::Completed,
            action_ids,
            abort: None,
            winner,
            placements: Some(placements),
        }
    }

    pub fn aborted(
        game_id: impl Into<String>,
        seed: Option<i64>,
        action_ids: Vec<u32>,
        abort: AbortRecord,
    ) -> Self {
        Self {
            game_id: game_id.into(),
            seed,
            status: HexoRecordStatus::Aborted,
            action_ids,
            abort: Some(abort),
            winner: None,
            placements: None,
        }
    }
}

/// Small return value mirroring the Python runner's record reference payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HexoRecordRef {
    pub path: PathBuf,
    pub game_id: String,
    pub status: HexoRecordStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HexoRecordFileMode {
    Read,
    Write,
}

impl HexoRecordFileMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "r",
            Self::Write => "w",
        }
    }
}

/// Reader/writer for the binary Hexo runner record file format.
#[derive(Debug)]
pub struct HexoRecordFile {
    path: PathBuf,
    mode: HexoRecordFileMode,
    file: Option<File>,
    engine_metadata: HexoRecordEngineMetadata,
    players: Vec<HexoRecordPlayer>,
    data_offset: u64,
}

impl HexoRecordFile {
    /// Create a new `.hxr` file and write its header.
    pub fn create(
        path: impl AsRef<Path>,
        engine_metadata: HexoRecordEngineMetadata,
        players: Vec<HexoRecordPlayer>,
    ) -> Result<Self, RecordError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)?;
        write_header(&mut file, &engine_metadata, &players)?;
        let data_offset = file.stream_position()?;

        Ok(Self {
            path,
            mode: HexoRecordFileMode::Write,
            file: Some(file),
            engine_metadata,
            players,
            data_offset,
        })
    }

    /// Open an existing `.hxr` file for reading.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RecordError> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new().read(true).open(&path)?;
        let (engine_metadata, players, data_offset) = read_header(&mut file)?;

        Ok(Self {
            path,
            mode: HexoRecordFileMode::Read,
            file: Some(file),
            engine_metadata,
            players,
            data_offset,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn mode(&self) -> HexoRecordFileMode {
        self.mode
    }

    pub fn engine_metadata(&self) -> &HexoRecordEngineMetadata {
        &self.engine_metadata
    }

    pub fn players(&self) -> &[HexoRecordPlayer] {
        &self.players
    }

    /// Start an append-only writer for one game.
    ///
    /// The writer holds a cloned file handle, so it stays valid independently
    /// of this `HexoRecordFile` and multiple games may be begun sequentially.
    /// Nothing hits disk until the writer's `finish_*` call.
    pub fn begin_game(
        &mut self,
        game_id: impl Into<String>,
        seed: Option<i64>,
    ) -> Result<HexoRecordGameWriter, RecordError> {
        if self.mode != HexoRecordFileMode::Write {
            return Err(RecordError::ReadOnlyFile);
        }

        let path = self.path.clone();
        let file = self.require_file_mut()?.try_clone()?;
        self.require_file_mut()?.flush()?;

        Ok(HexoRecordGameWriter {
            path,
            file,
            game_id: game_id.into(),
            seed,
            action_ids: Vec::new(),
            finished: false,
        })
    }

    /// Append one already-finalized record and flush it to disk.
    pub fn append_record(&mut self, record: &HexoRecord) -> Result<(), RecordError> {
        if self.mode != HexoRecordFileMode::Write {
            return Err(RecordError::ReadOnlyFile);
        }

        let file = self.require_file_mut()?;
        file.seek(SeekFrom::End(0))?;
        write_record(file, record)?;
        file.flush()?;
        Ok(())
    }

    /// Read every game record in the file (eager, not streaming).
    ///
    /// Works in both modes: a Write-mode file is flushed and the path is
    /// reopened for reading, which is why `RecordError::WriteOnlyFile` is
    /// never produced in practice.
    pub fn iter_records(&mut self) -> Result<Vec<HexoRecord>, RecordError> {
        match self.mode {
            HexoRecordFileMode::Read => {
                let data_offset = self.data_offset;
                let file = self.require_file_mut()?;
                file.seek(SeekFrom::Start(data_offset))?;
                read_records(file)
            }
            HexoRecordFileMode::Write => {
                if let Some(file) = self.file.as_mut() {
                    file.flush()?;
                }
                let mut reader = Self::open(&self.path)?;
                reader.iter_records()
            }
        }
    }

    pub fn close(&mut self) {
        self.file = None;
    }

    fn require_file_mut(&mut self) -> Result<&mut File, RecordError> {
        self.file.as_mut().ok_or(RecordError::ClosedFile)
    }
}

/// Append-only writer for one game inside a `HexoRecordFile`.
///
/// Actions are buffered in memory; the whole game payload is written and
/// flushed in one `finish_completed`/`finish_aborted` call, after which the
/// writer is dead (further calls return `RecordError::FinishedWriter`). A
/// game that never reaches `finish_*` leaves no bytes in the file.
#[derive(Debug)]
pub struct HexoRecordGameWriter {
    path: PathBuf,
    file: File,
    game_id: String,
    seed: Option<i64>,
    action_ids: Vec<u32>,
    finished: bool,
}

impl HexoRecordGameWriter {
    pub fn game_id(&self) -> &str {
        &self.game_id
    }

    pub fn seed(&self) -> Option<i64> {
        self.seed
    }

    pub fn action_count(&self) -> usize {
        self.action_ids.len()
    }

    pub fn record_action(&mut self, action_id: u32) -> Result<(), RecordError> {
        self.ensure_open()?;
        self.action_ids.push(action_id);
        Ok(())
    }

    pub fn finish_completed(
        &mut self,
        winner: Option<String>,
        placements: i64,
    ) -> Result<HexoRecordRef, RecordError> {
        self.ensure_open()?;
        let record = HexoRecord::completed(
            self.game_id.clone(),
            self.seed,
            self.action_ids.clone(),
            winner,
            placements,
        );
        self.finish(record)
    }

    pub fn finish_aborted(&mut self, abort: AbortRecord) -> Result<HexoRecordRef, RecordError> {
        self.ensure_open()?;
        let record = HexoRecord::aborted(
            self.game_id.clone(),
            self.seed,
            self.action_ids.clone(),
            abort,
        );
        self.finish(record)
    }

    fn ensure_open(&self) -> Result<(), RecordError> {
        if self.finished {
            return Err(RecordError::FinishedWriter {
                game_id: self.game_id.clone(),
            });
        }
        Ok(())
    }

    fn finish(&mut self, record: HexoRecord) -> Result<HexoRecordRef, RecordError> {
        self.file.seek(SeekFrom::End(0))?;
        write_record(&mut self.file, &record)?;
        self.file.flush()?;
        self.finished = true;
        Ok(HexoRecordRef {
            path: self.path.clone(),
            game_id: self.game_id.clone(),
            status: record.status,
        })
    }
}

// --- Wire helpers: header/record encode-decode, varint + zigzag primitives --

fn write_header(
    writer: &mut impl Write,
    engine_metadata: &HexoRecordEngineMetadata,
    players: &[HexoRecordPlayer],
) -> Result<(), RecordError> {
    writer.write_all(HEXO_RECORD_MAGIC)?;
    write_varint(writer, HEXO_RECORD_SCHEMA_VERSION)?;
    write_varint(writer, engine_metadata.rules_version)?;
    write_string(writer, &engine_metadata.backend)?;
    write_varint(writer, players.len() as u64)?;
    for player in players {
        write_string(writer, &player.player_id)?;
        write_string(writer, &player.role)?;
        write_optional_string(writer, player.label.as_deref())?;
    }
    Ok(())
}

fn read_header(
    reader: &mut (impl Read + Seek),
) -> Result<(HexoRecordEngineMetadata, Vec<HexoRecordPlayer>, u64), RecordError> {
    let mut magic = [0_u8; 8];
    read_exact(reader, &mut magic, "magic")?;
    if &magic != HEXO_RECORD_MAGIC {
        return Err(RecordError::InvalidMagic);
    }

    let version = read_varint(reader, "schema version")?;
    if version != HEXO_RECORD_SCHEMA_VERSION {
        return Err(RecordError::UnsupportedVersion {
            found: version,
            expected: HEXO_RECORD_SCHEMA_VERSION,
        });
    }

    let rules_version = read_varint(reader, "rules version")?;
    let backend = read_string(reader, "backend")?;
    let player_count = usize_len(read_varint(reader, "player count")?)?;
    let mut players = Vec::with_capacity(player_count);
    for _ in 0..player_count {
        players.push(HexoRecordPlayer {
            player_id: read_string(reader, "player id")?,
            role: read_string(reader, "player role")?,
            label: read_optional_string(reader, "player label")?,
        });
    }

    let data_offset = reader.stream_position()?;
    Ok((
        HexoRecordEngineMetadata {
            rules_version,
            backend,
        },
        players,
        data_offset,
    ))
}

fn write_record(writer: &mut impl Write, record: &HexoRecord) -> Result<(), RecordError> {
    let payload = encode_game_payload(record)?;
    writer.write_all(&[GAME_MARKER])?;
    write_varint(writer, payload.len() as u64)?;
    writer.write_all(&payload)?;
    Ok(())
}

fn read_records(reader: &mut impl Read) -> Result<Vec<HexoRecord>, RecordError> {
    let mut records = Vec::new();
    loop {
        let mut marker = [0_u8; 1];
        match reader.read(&mut marker) {
            Ok(0) => return Ok(records),
            Ok(1) if marker[0] == GAME_MARKER => {}
            Ok(1) => return Err(RecordError::InvalidMarker(marker[0])),
            Ok(_) => unreachable!("one-byte buffer cannot read more than one byte"),
            Err(error) => return Err(error.into()),
        }

        let length = usize_len(read_varint(reader, "game payload length")?)?;
        let mut payload = vec![0_u8; length];
        read_exact(reader, &mut payload, "game payload")?;
        records.push(decode_game_payload(&payload)?);
    }
}

fn encode_game_payload(record: &HexoRecord) -> Result<Vec<u8>, RecordError> {
    let mut buffer = Vec::new();
    append_string(&mut buffer, &record.game_id);
    append_optional_i64(&mut buffer, record.seed);
    buffer.push(record.status.to_byte());
    append_varint(&mut buffer, record.action_ids.len() as u64);
    for &action_id in &record.action_ids {
        buffer.extend(action_id.to_le_bytes());
    }
    append_optional_string(&mut buffer, record.winner.as_deref());
    append_optional_i64(&mut buffer, record.placements);
    match &record.abort {
        None => buffer.push(0),
        Some(abort) => {
            buffer.push(1);
            append_string(&mut buffer, &abort.stage);
            append_string(&mut buffer, &abort.exception_type);
            append_string(&mut buffer, &abort.message);
        }
    }
    Ok(buffer)
}

fn decode_game_payload(payload: &[u8]) -> Result<HexoRecord, RecordError> {
    let mut cursor = PayloadCursor::new(payload);
    let game_id = cursor.read_string("game id")?;
    let seed = cursor.read_optional_i64("seed")?;
    let status = HexoRecordStatus::from_byte(cursor.read_byte("status")?)?;
    let action_count = usize_len(cursor.read_varint("action count")?)?;
    let mut action_ids = Vec::with_capacity(action_count);
    for _ in 0..action_count {
        action_ids.push(cursor.read_u32("action id")?);
    }
    let winner = cursor.read_optional_string("winner")?;
    let placements = cursor.read_optional_i64("placements")?;
    let abort = if cursor.read_bool("abort flag")? {
        Some(AbortRecord {
            stage: cursor.read_string("abort stage")?,
            exception_type: cursor.read_string("abort exception type")?,
            message: cursor.read_string("abort message")?,
        })
    } else {
        None
    };
    cursor.finish()?;

    Ok(HexoRecord {
        game_id,
        seed,
        status,
        action_ids,
        abort,
        winner,
        placements,
    })
}

fn write_varint(writer: &mut impl Write, value: u64) -> Result<(), RecordError> {
    let mut buffer = Vec::new();
    append_varint(&mut buffer, value);
    writer.write_all(&buffer)?;
    Ok(())
}

fn read_varint(reader: &mut impl Read, context: &'static str) -> Result<u64, RecordError> {
    let mut shift = 0;
    let mut value = 0_u64;
    loop {
        let mut raw = [0_u8; 1];
        read_exact(reader, &mut raw, context)?;
        let byte = raw[0];
        if shift == 63 && byte > 1 {
            return Err(RecordError::VarintTooLarge);
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err(RecordError::VarintTooLarge);
        }
    }
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<(), RecordError> {
    write_varint(writer, value.len() as u64)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_string(reader: &mut impl Read, context: &'static str) -> Result<String, RecordError> {
    let length = usize_len(read_varint(reader, context)?)?;
    let mut encoded = vec![0_u8; length];
    read_exact(reader, &mut encoded, context)?;
    Ok(String::from_utf8(encoded)?)
}

fn write_optional_string(writer: &mut impl Write, value: Option<&str>) -> Result<(), RecordError> {
    match value {
        None => writer.write_all(&[0])?,
        Some(value) => {
            writer.write_all(&[1])?;
            write_string(writer, value)?;
        }
    }
    Ok(())
}

fn read_optional_string(
    reader: &mut impl Read,
    context: &'static str,
) -> Result<Option<String>, RecordError> {
    if read_bool(reader, context)? {
        Ok(Some(read_string(reader, context)?))
    } else {
        Ok(None)
    }
}

fn read_bool(reader: &mut impl Read, context: &'static str) -> Result<bool, RecordError> {
    let mut raw = [0_u8; 1];
    read_exact(reader, &mut raw, context)?;
    match raw[0] {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(RecordError::InvalidBool(value)),
    }
}

fn read_exact(
    reader: &mut impl Read,
    buffer: &mut [u8],
    context: &'static str,
) -> Result<(), RecordError> {
    reader.read_exact(buffer).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            RecordError::UnexpectedEof(context)
        } else {
            RecordError::Io(error)
        }
    })
}

fn append_string(buffer: &mut Vec<u8>, value: &str) {
    append_varint(buffer, value.len() as u64);
    buffer.extend(value.as_bytes());
}

fn append_optional_string(buffer: &mut Vec<u8>, value: Option<&str>) {
    match value {
        None => buffer.push(0),
        Some(value) => {
            buffer.push(1);
            append_string(buffer, value);
        }
    }
}

fn append_optional_i64(buffer: &mut Vec<u8>, value: Option<i64>) {
    match value {
        None => buffer.push(0),
        Some(value) => {
            buffer.push(1);
            append_varint(buffer, zigzag_encode(value));
        }
    }
}

fn append_varint(buffer: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buffer.push(((value & 0x7f) as u8) | 0x80);
        value >>= 7;
    }
    buffer.push(value as u8);
}

fn usize_len(value: u64) -> Result<usize, RecordError> {
    usize::try_from(value).map_err(|_| RecordError::LengthTooLarge(value))
}

fn zigzag_encode(value: i64) -> u64 {
    ((value as u64) << 1) ^ ((value >> 63) as u64)
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

/// Bounds-checked reader over one length-prefixed game payload.
///
/// `finish()` must be called after decoding; it rejects trailing bytes so a
/// schema drift between writer and reader fails loudly instead of silently
/// ignoring data.
struct PayloadCursor<'a> {
    payload: &'a [u8],
    offset: usize,
}

impl<'a> PayloadCursor<'a> {
    fn new(payload: &'a [u8]) -> Self {
        Self { payload, offset: 0 }
    }

    fn read_byte(&mut self, context: &'static str) -> Result<u8, RecordError> {
        if self.offset >= self.payload.len() {
            return Err(RecordError::UnexpectedEof(context));
        }
        let value = self.payload[self.offset];
        self.offset += 1;
        Ok(value)
    }

    fn read_bool(&mut self, context: &'static str) -> Result<bool, RecordError> {
        match self.read_byte(context)? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(RecordError::InvalidBool(value)),
        }
    }

    fn read_varint(&mut self, context: &'static str) -> Result<u64, RecordError> {
        let mut shift = 0;
        let mut value = 0_u64;
        loop {
            let byte = self.read_byte(context)?;
            if shift == 63 && byte > 1 {
                return Err(RecordError::VarintTooLarge);
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte < 0x80 {
                return Ok(value);
            }
            shift += 7;
            if shift > 63 {
                return Err(RecordError::VarintTooLarge);
            }
        }
    }

    fn read_u32(&mut self, context: &'static str) -> Result<u32, RecordError> {
        let bytes = self.read_bytes_exact(4, context)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_bytes(&mut self, context: &'static str) -> Result<&'a [u8], RecordError> {
        let length = usize_len(self.read_varint(context)?)?;
        self.read_bytes_exact(length, context)
    }

    fn read_bytes_exact(
        &mut self,
        length: usize,
        context: &'static str,
    ) -> Result<&'a [u8], RecordError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(RecordError::LengthTooLarge(u64::MAX))?;
        if end > self.payload.len() {
            return Err(RecordError::UnexpectedEof(context));
        }
        let bytes = &self.payload[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }

    fn read_string(&mut self, context: &'static str) -> Result<String, RecordError> {
        Ok(String::from_utf8(self.read_bytes(context)?.to_vec())?)
    }

    fn read_optional_string(
        &mut self,
        context: &'static str,
    ) -> Result<Option<String>, RecordError> {
        if self.read_bool(context)? {
            Ok(Some(self.read_string(context)?))
        } else {
            Ok(None)
        }
    }

    fn read_optional_i64(&mut self, context: &'static str) -> Result<Option<i64>, RecordError> {
        if self.read_bool(context)? {
            Ok(Some(zigzag_decode(self.read_varint(context)?)))
        } else {
            Ok(None)
        }
    }

    fn finish(&self) -> Result<(), RecordError> {
        if self.offset == self.payload.len() {
            Ok(())
        } else {
            Err(RecordError::TrailingPayloadBytes {
                trailing: self.payload.len() - self.offset,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{
        apply_placement, pack_coord, unpack_coord, HexCoord, HexoState, Placement, Player,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hexo_utils_records_{name}_{}_{}.hxr",
            std::process::id(),
            nonce
        ))
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
    }

    fn header_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(HEXO_RECORD_MAGIC);
        append_varint(&mut bytes, HEXO_RECORD_SCHEMA_VERSION);
        append_varint(&mut bytes, 77);
        append_string(&mut bytes, "rust-test");
        append_varint(&mut bytes, 0);
        bytes
    }

    fn winning_action_ids() -> Vec<u32> {
        [
            HexCoord::new(0, 0),
            HexCoord::new(0, 1),
            HexCoord::new(0, 2),
            HexCoord::new(1, 0),
            HexCoord::new(2, 0),
            HexCoord::new(1, 1),
            HexCoord::new(1, 2),
            HexCoord::new(3, 0),
            HexCoord::new(4, 0),
            HexCoord::new(2, 1),
            HexCoord::new(2, 2),
            HexCoord::new(5, 0),
        ]
        .into_iter()
        .map(pack_coord)
        .collect()
    }

    #[test]
    fn completed_record_round_trips_and_replays() {
        let path = temp_path("completed");
        let result = (|| {
            let players = vec![
                HexoRecordPlayer::new("p0", "player0", Some("Alpha".to_owned())),
                HexoRecordPlayer::new("p1", "player1", None),
            ];
            let mut file = HexoRecordFile::create(
                &path,
                HexoRecordEngineMetadata::new(77, "rust-test"),
                players,
            )?;
            let mut writer = file.begin_game("scripted", Some(7))?;
            for action_id in winning_action_ids() {
                writer.record_action(action_id)?;
            }
            writer.finish_completed(Some("player0".to_owned()), 12)?;

            let records = file.iter_records()?;
            assert_eq!(records.len(), 1);
            let record = &records[0];
            assert_eq!(record.game_id, "scripted");
            assert_eq!(record.seed, Some(7));
            assert_eq!(record.status, HexoRecordStatus::Completed);
            assert_eq!(record.winner.as_deref(), Some("player0"));
            assert_eq!(record.placements, Some(12));
            assert_eq!(record.abort, None);

            let mut state = HexoState::new();
            for action_id in &record.action_ids {
                apply_placement(
                    &mut state,
                    Placement {
                        coord: unpack_coord(*action_id),
                    },
                )
                .unwrap();
            }
            assert_eq!(state.terminal().unwrap().winner, Player::Player0);
            Ok::<(), RecordError>(())
        })();
        let _ = fs::remove_file(&path);
        result.unwrap();
    }

    #[test]
    fn aborted_record_round_trips_abort_data() {
        let path = temp_path("aborted");
        let result = (|| {
            let mut file = HexoRecordFile::create(
                &path,
                HexoRecordEngineMetadata::new(77, "rust-test"),
                Vec::new(),
            )?;
            let mut writer = file.begin_game("bad-game", None)?;
            writer.record_action(pack_coord(HexCoord::ZERO))?;
            writer.finish_aborted(AbortRecord::new(
                "engine.apply_action",
                "IllegalActionError",
                "opening placement must be at the origin",
            ))?;

            let records = file.iter_records()?;
            assert_eq!(records.len(), 1);
            let record = &records[0];
            assert_eq!(record.status, HexoRecordStatus::Aborted);
            assert_eq!(record.action_ids, vec![pack_coord(HexCoord::ZERO)]);
            assert_eq!(record.winner, None);
            assert_eq!(record.placements, None);
            let abort = record.abort.as_ref().unwrap();
            assert_eq!(abort.stage, "engine.apply_action");
            assert_eq!(abort.exception_type, "IllegalActionError");
            assert_eq!(abort.message, "opening placement must be at the origin");
            Ok::<(), RecordError>(())
        })();
        let _ = fs::remove_file(&path);
        result.unwrap();
    }

    #[test]
    fn open_rejects_bad_magic() {
        let path = temp_path("bad_magic");
        write_bytes(&path, b"NOTHXREC");
        let error = HexoRecordFile::open(&path).unwrap_err();
        let _ = fs::remove_file(&path);
        assert!(matches!(error, RecordError::InvalidMagic));
    }

    #[test]
    fn open_rejects_unsupported_version() {
        let path = temp_path("bad_version");
        let mut bytes = Vec::new();
        bytes.extend(HEXO_RECORD_MAGIC);
        append_varint(&mut bytes, HEXO_RECORD_SCHEMA_VERSION + 1);
        write_bytes(&path, &bytes);

        let error = HexoRecordFile::open(&path).unwrap_err();
        let _ = fs::remove_file(&path);
        assert!(matches!(
            error,
            RecordError::UnsupportedVersion {
                found,
                expected: HEXO_RECORD_SCHEMA_VERSION
            } if found == HEXO_RECORD_SCHEMA_VERSION + 1
        ));
    }

    #[test]
    fn iter_records_rejects_truncated_payload() {
        let path = temp_path("truncated");
        let mut bytes = header_bytes();
        bytes.push(GAME_MARKER);
        append_varint(&mut bytes, 5);
        bytes.extend([1, 2]);
        write_bytes(&path, &bytes);

        let mut file = HexoRecordFile::open(&path).unwrap();
        let error = file.iter_records().unwrap_err();
        let _ = fs::remove_file(&path);
        assert!(matches!(error, RecordError::UnexpectedEof("game payload")));
    }
}
