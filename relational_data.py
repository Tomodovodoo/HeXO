"""Full-history human examples, retaining the verified corpus family split."""
import json
from pathlib import Path

import numpy as np

from corpus_warmstart import verified_shards
from human_corpus import digest, owner, validate
from klent import digest as file_digest


def human_examples(directory, positions=0, seed=1729):
    """Load geometry from histories; old replay is used only to verify split identity.

    Outcome labels describe the recorded human continuation at the chosen action.
    Unchosen action values are unknown, not zero. Test/excluded histories never
    become examples. `positions=0` retains every allowed pre-move position.
    """
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    files = verified_shards(directory)
    metadata = {str(Path(item['path'])): item for item in manifest['shards']}
    allowed = {}
    for split, paths in files.items():
        for path in paths:
            item = metadata[str(path.relative_to(directory.resolve()))]
            with np.load(path, allow_pickle=False) as data:
                families = data['family']
                offset = 0
                for game, count in zip(item['games'], item['game_row_counts'], strict=True):
                    values = np.unique(families[offset:offset+count])
                    if count < 0 or (count and len(values) != 1) or game in allowed:
                        raise ValueError('Malformed or duplicate corpus game membership')
                    allowed[game] = (split, int(values[0]) if count else None)
                    offset += count
                if offset != len(families):
                    raise ValueError('Corpus game row counts disagree with split shard')
    result = {'train': [], 'validation': []}
    histories = {}
    converted_games = set(allowed)
    limited = manifest.get('conversion', {}).get('limit_games_per_split', 0) > 0
    minimum = manifest['minimum_ply']
    if minimum < 3:
        raise ValueError('Unsafe corpus family prefix')
    for text in (directory/'games.jsonl').read_text().splitlines():
        record = json.loads(text)
        key = record['content_sha256']
        if key not in allowed:
            if key in converted_games:
                raise ValueError('Duplicate converted human history')
            if record['split'] in result and not limited:
                raise ValueError('Training history is absent from verified split membership')
            continue
        split, family = allowed.pop(key)
        if record['split'] != split or (family is not None and record['family'] != family) or digest(record['moves']) != key:
            raise ValueError('History changed or disagrees with verified split membership')
        validate(record)
        if family is None:
            if len(record['moves'])>minimum:
                raise ValueError('Zero-row corpus game has eligible positions')
            continue
        histories[key] = record['moves']
        for ply in range(minimum, len(record['moves'])):
            result[split].append(dict(game=key, ply=ply, action=record['moves'][ply],
                                      player=owner(ply), family=family,
                                      target=record['winner']*(1 if owner(ply)==0 else -1)))
    if allowed or not all(result.values()):
        raise ValueError('Missing allowed human histories')
    available = {split: len(rows) for split, rows in result.items()}
    if positions:
        if positions < 2:
            raise ValueError('At least two human positions required')
        rng = np.random.default_rng(seed)
        for split, fraction in (('train', .8), ('validation', .2)):
            rows = result[split]
            count = min(len(rows), max(1, int(positions*fraction)))
            result[split] = [rows[int(i)] for i in rng.permutation(len(rows))[:count]]
    provenance = dict(manifest_sha256=file_digest(directory/'manifest.json'),
                      histories_sha256=file_digest(directory/'games.jsonl'),
                      shards={split: {str(p): file_digest(p) for p in paths} for split, paths in files.items()},
                      available=available, selected={split: len(rows) for split, rows in result.items()},
                      minimum_ply=minimum, target='Human continuation terminal STM outcome at recorded action',
                      test_examples_loaded=False, excluded_examples_loaded=False)
    return histories, result, provenance
