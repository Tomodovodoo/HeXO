"""Convert gumbel-policy-value-v1 search corpora into dense shards `<run>/shards/NNNNNN/`, one per old corpus.

The old corpora record no root values, so capped games keep their policy rows with value weight 0.
Opening plies and the rows dropped from early capped games are marked full_search=False.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from hexo import Game
from klent import digest
import dense_data

OLD_SCHEMA = 'hexo-search-selfplay-v1'


def read_corpus(path):
    """Read an old search corpus after verifying its manifest hashes and policy offsets."""
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest['schema'] != OLD_SCHEMA:
        raise ValueError(f'Expected a search self-play corpus: {path}')
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(path/name) != sha:
            raise ValueError(f'Corpus changed: {path/name}')
    episodes = json.loads((path/'episodes.json').read_text()); rows = json.loads((path/'rows.json').read_text())
    with np.load(path/'targets.npz', allow_pickle=False) as data:
        offsets = data['offsets']; probabilities = data['probabilities']
    if len(offsets) != len(rows)+1 or offsets[0] != 0 or offsets[-1] != len(probabilities) or np.any(np.diff(offsets) <= 0):
        raise ValueError(f'Malformed search-policy offsets: {path}')
    for i, row in enumerate(rows):
        row['policy'] = probabilities[offsets[i]:offsets[i+1]]
    return manifest, episodes, rows


def convert(manifest, manifest_sha256, episodes, rows):
    actor = manifest['identity'].get('actor_sha256')
    index = {e['id']: i for i, e in enumerate(episodes)}
    searched = {(r['game'], r['ply']) for r in rows}
    new_episodes = [dict(moves=e['moves'], winner=e['winner'], reason=e['reason'], opening_plies=len(e['opening']),
                         actor=actor, root_values=None, full_search=[(e['id'], p) in searched for p in range(len(e['moves']))])
                    for e in episodes]
    new_rows = []
    for r in rows:
        e = episodes[index[r['game']]]
        if list(e['moves'][r['ply']]) != list(r['action']):
            raise ValueError(f'Row action disagrees with episode {e["id"]} ply {r["ply"]}')
        if r['target'] is not None and r['target'] != (1. if r['player'] == e['winner'] else -1.):
            raise ValueError('Row target disagrees with episode winner')
        new_rows.append(dict(game=index[r['game']], ply=r['ply'], player=r['player'], remaining=r['remaining'],
                             target=None if r['target'] is None else (r['target']+1)/2,
                             weight=0. if r['target'] is None else 1., legal_sha256=r['legal_sha256'], policy=r['policy']))
    identity = dict(source='gumbel-policy-value-v1', source_manifest_sha256=manifest_sha256, actor_sha256=actor,
                    policy_target=manifest['identity'].get('policy_target'), value_target='terminal outcome; capped games masked')
    return identity, new_episodes, new_rows


def check(path):
    """Replay every row of a shard: side to move, legal-list hash and (for policy rows) policy length must match.
    Returns the row count."""
    episodes, rows = dense_data.read_shard(path)
    by_game = {}
    for row in rows:
        by_game.setdefault(row['game'], []).append(row)
    for g, items in by_game.items():
        moves = episodes[g]['moves']; game = Game(); ply = 0
        for row in sorted(items, key=lambda r: r['ply']):
            while ply < row['ply']:
                game.play(*moves[ply]); ply += 1
            actions = np.array(game.legal_moves(), np.int64).reshape(-1, 2)
            if (game.player, game.remaining) != (row['player'], row['remaining']) \
                    or len(row['policy']) not in (0, len(actions)) or dense_data.legal_digest(actions) != row['legal_sha256']:
                raise ValueError(f'Row game {g} ply {row["ply"]} of {path} does not replay')
        game.close()
    return len(rows)


def parse_range(text):
    a, _, b = text.partition('-')
    return range(int(a), int(b or a)+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--corpora', type=parse_range, default=None, help='e.g. 1-24 (default: all)')
    args = parser.parse_args()
    numbers = args.corpora or [int(p.name) for p in sorted((args.source/'corpus').iterdir()) if p.name.isdigit()]
    print(f'{"corpus":>6} {"games":>6} {"rows":>7} {"policy":>7} {"terminal":>8} {"capped":>6} {"plies":>6}')
    totals = np.zeros(5, np.int64); plies = 0
    for number in numbers:
        source = args.source/'corpus'/f'{number:04d}'
        manifest, episodes, rows = read_corpus(source)
        identity, new_episodes, new_rows = convert(manifest, digest(source/'manifest.json'), episodes, rows)
        target = args.run/'shards'/f'{number:06d}'
        counts = dense_data.write_shard(target, identity, new_episodes, new_rows)['counts']
        check(target)
        mean = np.mean([len(e['moves']) for e in new_episodes])
        row = [counts[k] for k in ('games', 'rows', 'policy_rows', 'terminal_games', 'capped_games')]
        totals += row; plies += sum(len(e['moves']) for e in new_episodes)
        print(f'{number:>6} {row[0]:>6} {row[1]:>7} {row[2]:>7} {row[3]:>8} {row[4]:>6} {mean:>6.1f}')
    print(f'{"total":>6} {totals[0]:>6} {totals[1]:>7} {totals[2]:>7} {totals[3]:>8} {totals[4]:>6} {plies/totals[0]:>6.1f}')


if __name__ == '__main__':
    main()
