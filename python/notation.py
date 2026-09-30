"""Official Hexagonal Tic Tac Toe notation v1, with native legality validation.

Spec revision: hex-tic-tac-toe/hexagonal-tic-tac-toe-notation@15bb7877ae020d661497e332adf0810d00d24e3e.
The implicit origin belongs to cross (native player 0). A one-placement winning
final turn is supported for compatibility with Tyto, without padding or truncation.
Metadata values and optional threat annotations are preserved, not inferred.
"""
from dataclasses import dataclass
import re

from hexo import Game

MAX_TEXT = 1_048_576
MAX_STONES = 4097


class NotationConflict(ValueError):
    """A valid Hexo state that complete-turn notation cannot represent."""


@dataclass
class Record:
    history: list
    metadata: dict
    threats: list


def _validate(history):
    if not history:
        raise NotationConflict("V1 always implies cross at [0,0]; an empty board is not representable")
    if len(history) > MAX_STONES:
        raise ValueError(f"History exceeds local limit of {MAX_STONES} placements")
    game = Game()
    try:
        for move in history:
            if not isinstance(move, (list, tuple)) or len(move) != 2:
                raise ValueError("Each placement must have two coordinates")
            game.play(*move)
    finally:
        game.close()


def loads(text):
    """Parse a complete v1 record. Returned history includes the implicit origin."""
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise ValueError("Notation must be text within the local size limit")
    metadata, pos = {}, 0
    # Whitespace outside metadata values has no semantic meaning.
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos].isalpha():
        while True:
            match = re.match(r"\s*((?:[a-z]\s*)+)\[([^\]]*)\]", text[pos:])
            if not match:
                raise ValueError("Malformed metadata")
            key = ''.join(match[1].split())
            if key in metadata:
                raise ValueError(f"Duplicate metadata key: {key}")
            metadata[key] = match[2]
            pos += match.end()
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos < len(text) and text[pos] == ';':
                pos += 1
                break
    if 'version' in metadata and (not re.fullmatch(r'[0-9]+', metadata['version'])
                                   or int(metadata['version']) != 1):
        raise ValueError("Only notation version 1 is supported")
    rest = ''.join(text[pos:].split())
    pattern = re.compile(r'([0-9]+)\.\[(-?[0-9]+),(-?[0-9]+)\](?:\[(-?[0-9]+),(-?[0-9]+)\])?(!*);')
    history, threats, pos = [(0, 0)], [], 0
    while pos < len(rest):
        match = pattern.match(rest, pos)
        if not match or int(match[1]) != len(threats)+1:
            raise ValueError("Expected consecutive numbered turns with two coordinates, or one stone in the final turn")
        history.append((int(match[2]), int(match[3])))
        if match[4] is not None:
            history.append((int(match[4]), int(match[5])))
        elif match.end() != len(rest):
            raise ValueError("A single-stone turn must be the final turn")
        threats.append(len(match[6]))
        pos = match.end()
    _validate(history)
    return Record(history, metadata, threats)


def dumps(record):
    """Export Record or a placement history; never add or discard placements."""
    if isinstance(record, Record):
        history, metadata, threats = record.history, record.metadata, record.threats
    else:
        history, metadata, threats = list(record), {'version': '1'}, None
    _validate(history)
    turns = len(history)//2
    if threats is None:
        threats = [0]*turns
    if len(threats) != turns or any(type(t) is not int or not 0 <= t <= MAX_TEXT for t in threats):
        raise ValueError("Threat annotations must match the number of turns")
    if sum(threats) > MAX_TEXT:
        raise ValueError("Threat annotations exceed local size limit")
    entries = []
    for key, value in metadata.items():
        if not isinstance(key, str) or not re.fullmatch('[a-z]+', key) or not isinstance(value, str) or ']' in value:
            raise ValueError("Metadata keys must be lowercase letters; values cannot contain ]")
        if len(key)+len(value) > MAX_TEXT:
            raise ValueError("Metadata exceeds local size limit")
        entries.append(f'{key}[{value}]')
    if sum(map(len, entries)) > MAX_TEXT:
        raise ValueError("Metadata exceeds local size limit")
    if 'version' in metadata and (not re.fullmatch(r'[0-9]+', metadata['version'])
                                   or int(metadata['version']) != 1):
        raise ValueError("Only notation version 1 is supported")
    rows = [''.join(entries)+';'] if entries else []
    for turn, count in enumerate(threats, 1):
        moves = history[turn*2-1:turn*2+1]
        coordinates = ''.join(f'[{q},{r}]' for q, r in moves)
        rows.append(f'{turn}. {coordinates}{"!"*count};')
    text = '\n'.join(rows)
    if len(text) > MAX_TEXT:
        raise ValueError("Notation exceeds local size limit")
    return text


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('import', 'export'))
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    source = args.input.read_text(encoding='utf-8')
    if args.action == 'import':
        record = loads(source)
        print(json.dumps(record.__dict__, ensure_ascii=False))
    else:
        data = json.loads(source)
        print(dumps(Record(**data) if isinstance(data, dict) else data))
