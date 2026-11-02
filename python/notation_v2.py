"""HTTTX v2 syntax tree. Coordinates are axial; annotations belong to positions after a stone.

Specification: https://github.com/hex-tic-tac-toe/htttx-notation
Custom setups are preserved as positions, never invented placement histories.
"""
import re
import copy
from dataclasses import dataclass

INTEGER = r'(?:0|-?[1-9][0-9]*)'
POINT = rf'{INTEGER},{INTEGER}'


@dataclass
class StudyRecord:
    metadata: dict
    setup: dict | None
    turns: list

    @property
    def history(self):
        if self.setup and self.setup['first'] == 'x' and not self.setup['x'] and not self.setup['o'] and not self.turns:
            return []
        if self.setup and (self.setup['first'] != 'o' or self.setup['o'] or self.setup['x'] != [[0, 0]]):
            raise ValueError('Custom setups have no placement history; use setup and turns')
        return [(0, 0)] + [tuple(m['stone']) for t in self.turns for m in t['moves'] if m['stone'] is not None]


def loads(text):
    if not isinstance(text, str) or len(text) > 1_048_576:
        raise ValueError('Notation exceeds local text limit')
    text = text.strip()
    metadata = {}
    while not text.startswith(';'):
        match = re.match(r'((?:[a-z]\s*)+)\[([^\]]*)\]\s*', text)
        if not match:
            raise ValueError('Malformed v2 metadata')
        key = ''.join(match[1].split())
        if key in metadata:
            raise ValueError('Duplicate metadata tag')
        metadata[key] = match[2]
        text = text[match.end():]
    if metadata.get('version') != '2' or next(iter(metadata)) != 'version':
        raise ValueError('Expected version[2] first')
    text = ''.join(text[1:].split())
    at, count = 0, 0

    def take(pattern):
        nonlocal at
        match = re.match(pattern, text[at:])
        if not match:
            raise ValueError(f'Malformed v2 notation near {text[at:at+32]}')
        at += match.end()
        return match

    def visuals():
        out = []
        while text[at:at+1] == '<':
            out.append(take(r'<[^<>]*>')[0])
        return out

    def move(terminal=False, setup=False):
        nonlocal count
        token = take(rf'\[({POINT}' + (r'|/' if terminal else '') + r')\]')[1]
        stone = None if token == '/' else list(map(int, token.split(',')))
        if stone is not None:
            count += 1
            if count > 4096 or any(abs(v) > 2**53-1 for v in stone):
                raise ValueError('Stone count or coordinate exceeds local limit')
        info = take(rf'\{{(?:@[0-9]+(?::[%#]{INTEGER})?|[%#]{INTEGER})\}}')[0] if not setup and text[at:at+1] == '{' else None
        return dict(stone=stone, info=info, visuals=[] if setup else visuals())

    setup = None
    if text[:1] in ('x', 'o'):
        first = text[0]
        at += 1
        second = 'o' if first == 'x' else 'x'
        setup = dict(first=first, x=[], o=[], visuals=[])
        while text[at:at+1] == '[':
            setup[first].append(move(setup=True)['stone'])
        take(':' + second)
        while text[at:at+1] == '[':
            setup[second].append(move(setup=True)['stone'])
        setup['visuals'] = visuals()
        take(';')
        cells = [tuple(p) for p in setup['x'] + setup['o']]
        if len(set(cells)) != len(cells):
            raise ValueError('Duplicate setup cell')

    def line(start, depth=0):
        nonlocal at
        if depth > 64:
            raise ValueError('Variations exceed 64 levels')
        out, ended = [], False
        while at < len(text) and text[at] != ')':
            if ended:
                raise ValueError('A terminator must end its line')
            number = int(take(r'(0|[1-9][0-9]*)\.')[1])
            if number != start + len(out):
                raise ValueError('Expected consecutive turn numbers')
            moves, variations = [move(), move(True)], []
            while text[at:at+1] == '(':
                at += 1
                variations.append(line(number, depth+1))
                take(r'\)')
            take(';')
            out.append(dict(number=number, moves=moves, variations=variations))
            ended = moves[1]['stone'] is None
        if not out and (not setup or depth):
            raise ValueError('Expected a line')
        return out

    turns = line(1)
    if at != len(text):
        raise ValueError('Unexpected trailing notation')
    return StudyRecord(metadata, setup, turns)


def dumps(record):
    if isinstance(record, dict):
        record = StudyRecord(**record)
    tags = {'version': '2', **record.metadata}
    tags['version'] = '2'
    for key, value in tags.items():
        if not re.fullmatch('[a-z]+', key) or not isinstance(value, str) or ']' in value:
            raise ValueError('Invalid metadata')
    text = ''.join(f'{key}[{value}]' for key, value in tags.items()) + ';\n'

    def stone(m):
        p = ','.join(map(str, m['stone'])) if m['stone'] is not None else '/'
        return f'[{p}]' + (m.get('info') or '') + ''.join(m.get('visuals', []))

    def line(turns):
        return '\n'.join(f"{t['number']}. " + ' '.join(map(stone, t['moves'])) +
                         ''.join(' (' + line(v) + ')' for v in t['variations']) + ';' for t in turns)

    if record.setup:
        setup = record.setup
        first = setup['first']
        second = 'o' if first == 'x' else 'x'
        text += first + ''.join(stone({'stone': p}) for p in setup[first]) + ':' + second
        text += ''.join(stone({'stone': p}) for p in setup[second]) + ''.join(setup['visuals']) + ';\n'
    text += line(record.turns)
    loads(text)
    return text


def position_key(history):
    return ';'.join(','.join(map(str, p)) for p in history)


class Study:
    def __init__(self, data=None):
        self.data = copy.deepcopy(data or dict(metadata={'version': '2'}, setup=None, lines=[], annotations={}))

    def capture(self, history):
        if not history:
            return
        setup = self.data['setup']
        if setup and setup['first'] == 'x' and not setup['x'] and not setup['o']:
            self.data['setup'] = dict(first='o', x=[[0, 0]], o=[], visuals=setup['visuals'])
        history = [list(p) for p in history]
        lines = self.data['lines']
        if any(h[:len(history)] == history for h in lines):
            return
        for i, h in enumerate(lines):
            if history[:len(h)] == h:
                lines[i] = history
                return
        lines.append(history)

    def add_clocks(self, history, turns):
        """Attach saved turn balances to their last placement, excluding timeouts without a stone."""
        self.capture(history)
        for turn in turns:
            ply, side = turn.get('end_ply', turn['ply']), turn['side']
            # Older tournament logs use the turn's starting ply; freeplay logs use its ending ply.
            if 'end_ply' not in turn and 'clock_spent_ms' in turn:
                ply = min(len(history), ply + (2 if ply % 2 else 1))
            if not 1 < ply <= len(history) or (ply // 2) % 2 != side:
                continue
            balance = turn.get('circle_ms' if side else 'cross_ms')
            if balance is None:
                continue
            prefix = history[:ply]
            key = position_key(prefix)
            saved = self.data['annotations'].get(key, {})
            evaluation = re.search(r'[%#]-?[0-9]+', saved.get('info') or '')
            info = '{@' + str(max(0, round(balance))) + (':' + evaluation[0] if evaluation else '') + '}'
            self.data['annotations'][key] = dict(stone=list(prefix[-1]), info=info, visuals=saved.get('visuals', []))

    @classmethod
    def from_record(cls, record):
        record.history  # Refuse to invent a history for an arbitrary setup.
        study = cls(dict(metadata=record.metadata, setup=record.setup, lines=[], annotations={}))
        if not record.history:
            return study

        def visit(turns, prefix):
            index = len(study.data['lines'])
            study.data['lines'].append(None)
            h = prefix[:]
            for turn in turns:
                before = h[:]
                for move in turn['moves']:
                    if move['stone'] is not None:
                        h.append(move['stone'])
                    if move['stone'] is not None or move['info'] or move['visuals']:
                        key = position_key(h)
                        saved = study.data['annotations'].get(key, {})
                        study.data['annotations'][key] = copy.deepcopy(dict(
                            stone=h[-1], info=move['info'] or saved.get('info'),
                            visuals=move['visuals'] or saved.get('visuals', [])))
                for v in turn['variations']:
                    visit(v, before)
            study.data['lines'][index] = h
        visit(record.turns, [[0, 0]])
        return study

    def record(self, lookup=lambda h: None):
        root = []
        for history in self.data['lines']:
            children = root
            for i in range(1, len(history), 2):
                pair = history[i:i+2]
                node = next((n for n in children if n['pair'] == pair), None)
                if node is None:
                    node = dict(pair=pair, prefix=history[:i], children=[])
                    children.append(node)
                children = node['children']

        def branch(node, alternatives=()):
            result = []
            while True:
                moves = []
                for j, p in enumerate(node['pair']):
                    prefix = node['prefix'] + node['pair'][:j+1]
                    saved = self.data['annotations'].get(position_key(prefix), {})
                    move = dict(stone=p, info=saved.get('info'), visuals=saved.get('visuals', []))
                    ev = lookup(prefix)
                    if ev and ev.get('value') is not None:
                        x = ev['value'] if ((len(prefix)+1)//2) % 2 == 0 else 1-ev['value']
                        clock = re.search(r'@[0-9]+', move['info'] or '')
                        move['info'] = '{' + (clock[0]+':' if clock else '') + '%' + str(round(100*(2*x-1))) + '}'
                    moves.append(move)
                if len(moves) == 1:
                    moves.append(dict(stone=None, info=None, visuals=[]))
                turn = dict(number=(len(node['prefix'])+1)//2, moves=moves,
                            variations=[branch(n) for n in alternatives])
                result.append(turn)
                if not node['children']:
                    return result
                node, alternatives = node['children'][0], node['children'][1:]
        setup = self.data['setup']
        if not root and setup is None:
            setup = dict(first='o' if self.data['lines'] else 'x', x=[[0, 0]] if self.data['lines'] else [], o=[], visuals=[])
        return StudyRecord(self.data['metadata'], setup, branch(root[0], root[1:]) if root else [])
