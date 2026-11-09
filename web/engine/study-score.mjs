/* A readable turn score: each variation replaces the turn beside it, at any nesting depth. */
import {positionKey} from './study.mjs';

export function scoreTree(study) {
  const roots = [];
  for (const [lineIndex, history] of study.lines.entries()) {
    let children = roots;
    for (let start = 1; start < history.length; start += 2) {
      const stones = history.slice(start, start + 2), key = positionKey(stones);
      let node = children.find(n => n.key === key);
      if (!node) {
        node = {key, id: positionKey(history.slice(0, start + stones.length)), start, stones,
          history: history.slice(0, start + stones.length), lineIndex, children: []};
        children.push(node);
      }
      children = node.children;
    }
  }
  return roots;
}

export function annotationText(info) {
  const parts = [], clock = info?.match(/@(\d+)/), evaluation = info?.match(/([%#])(-?\d+)/);
  if (evaluation) {
    const value = Number(evaluation[2]);
    parts.push(evaluation[1] === '%' ? `X ${value > 0 ? '+' : ''}${value}`
      : `${value >= 0 ? 'X' : 'O'} wins in ${Math.abs(value)} turns`);
  }
  if (clock) {
    const ms = Number(clock[1]);
    parts.push(`${Math.floor(ms / 60000)}:${(ms % 60000 / 1000).toFixed(1).padStart(4, '0')}`);
  }
  return parts.join(' · ');
}

export function createScoreView() {
  const collapsed = new Set();
  let lastSelection = null;
  let menu = null, menuEvents = null, menuTarget = null, menuState = null;
  const closeMenu = (restoreFocus = false) => {
    menuEvents?.abort(); menuEvents = null;
    menu?.remove(); menu = null;
    menuState = null;
    if (restoreFocus && menuTarget) {
      const target = menuTarget.isConnected ? menuTarget : [...menuTarget.ownerDocument.querySelectorAll('[data-focus]')]
        .find(element => element.dataset.focus === menuTarget.dataset.focus);
      target?.focus({preventScroll: true});
    }
    menuTarget = null;
  };
  const view = {
    render(box, study, options) {
      const state = JSON.stringify([options.history, options.ply, study.lines]);
      if (menu && menuState !== state) closeMenu();
      const doc = box.ownerDocument, selected = options.history.slice(0, options.ply), selectedKey = positionKey(selected);
      const focus = box.contains(doc.activeElement) ? doc.activeElement.dataset.focus : null;
      const make = (tag, cls, text) => { const e = doc.createElement(tag); e.className = cls; if (text) e.textContent = text; return e; };
      const button = (label, action, cls = 'score-action') => {
        const b = make('button', cls, label); b.type = 'button'; b.onclick = action; b.dataset.focus = cls + ':' + label; return b;
      };
      const roots = scoreTree(study), branches = [];
      let selectedBranch = null;
      const currentLineKey = positionKey(options.history);
      const active = node => selected.length > node.start && (currentLineKey === node.id || currentLineKey.startsWith(node.id + ';'));
      function scan(nodes, depth = 0) {
        for (const [index, node] of nodes.entries()) {
          if (index) { branches.push(node); if (active(node)) selectedBranch = {node, depth: depth + 1}; }
          scan(node.children, depth + (index ? 1 : 0));
        }
      }
      scan(roots);
      // Follow the selected branch automatically, while keeping manual collapse stable on redraws.
      if (lastSelection !== selectedKey) for (const node of branches) if (active(node)) collapsed.delete(node.id);
      lastSelection = selectedKey;
      const toolbar = make('div', 'score-tools');
      toolbar.append(make('span', 'score-location', selectedBranch ? `Variation · depth ${selectedBranch.depth}` : 'Main line'));
      if (selectedBranch) {
        toolbar.append(button('Promote', () => options.onPromote(selectedBranch.node.lineIndex, 'local')),
          button('Make main', () => options.onPromote(selectedBranch.node.lineIndex, true)),
          button('Return to main', () => options.onSelect(0, Math.min(options.ply, study.lines[0].length))));
      }
      if (branches.length) {
        toolbar.append(button('Expand all', () => { collapsed.clear(); redraw(); }),
          button('Collapse all', () => { for (const n of branches) collapsed.add(n.id); redraw(); }));
      }
      function redraw() { view.render(box, study, options); }
      const contextMenu = (event, target, lineIndex, ply, branchNode) => {
        event.preventDefault(); event.stopPropagation(); closeMenu();
        menuState = state;
        menuTarget = target;
        const win = doc.defaultView, viewport = win.visualViewport;
        const left = viewport?.offsetLeft || 0, top = viewport?.offsetTop || 0;
        const width = viewport?.width || win.innerWidth, height = viewport?.height || win.innerHeight;
        menu = make('div', 'score-menu'); menu.setAttribute('role', 'menu'); menu.setAttribute('aria-label', 'Move actions');
        menu.append(make('div', 'score-menu-title', `Turn ${Math.ceil((ply - 1) / 2)} · [${study.lines[lineIndex][ply - 1]}]`));
        const item = (label, action, disabled = false) => {
          const b = button(label, () => { closeMenu(); action(); }, 'score-menu-item'); b.disabled = disabled;
          b.setAttribute('role', 'menuitem'); menu.append(b); return b;
        };
        item('View position', () => options.onSelect(lineIndex, ply));
        item('Promote variation', () => options.onPromote(lineIndex, 'local', ply), lineIndex === 0);
        item('Make main line', () => options.onPromote(lineIndex, true, ply), lineIndex === 0);
        item('Return to main line', () => options.onSelect(0, Math.min(ply, study.lines[0].length)), lineIndex === 0);
        if (branchNode) item('Collapse variation', () => { collapsed.add(branchNode.id); redraw(); });
        doc.body.append(menu);
        menu.style.maxHeight = Math.max(0, height - 16) + 'px';
        menu.style.maxWidth = Math.max(0, width - 16) + 'px';
        const rect = target.getBoundingClientRect(), bounds = menu.getBoundingClientRect();
        const x = event.type === 'keydown' ? rect.left : event.clientX;
        const y = event.type === 'keydown' ? rect.bottom : event.clientY;
        menu.style.left = Math.max(left + 8, Math.min(x, left + width - bounds.width - 8)) + 'px';
        menu.style.top = Math.max(top + 8, Math.min(y, top + height - bounds.height - 8)) + 'px';
        menuEvents = new win.AbortController(); const signal = menuEvents.signal;
        doc.addEventListener('pointerdown', e => { if (!menu?.contains(e.target)) closeMenu(); }, {signal});
        doc.addEventListener('focusin', e => { if (!menu?.contains(e.target)) closeMenu(); }, {signal});
        win.addEventListener('resize', () => closeMenu(), {signal});
        doc.addEventListener('scroll', e => { if (!menu?.contains(e.target)) closeMenu(); }, {capture: true, signal});
        menu.addEventListener('contextmenu', e => e.preventDefault(), {signal});
        menu.addEventListener('keydown', e => {
          if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeMenu(true); return; }
          const items = [...menu.querySelectorAll('button:not(:disabled)')], index = items.indexOf(doc.activeElement);
          const next = e.key === 'ArrowDown' ? (index + 1) % items.length : e.key === 'ArrowUp' ? (index + items.length - 1) % items.length : e.key === 'Home' ? 0 : e.key === 'End' ? items.length - 1 : -1;
          if (next >= 0) { e.preventDefault(); e.stopPropagation(); items[next].focus(); }
        }, {signal});
        menu.querySelector('button:not(:disabled)').focus({preventScroll: true});
      };
      const bindContext = (element, lineIndex, ply, branchNode) => {
        element.oncontextmenu = e => contextMenu(e, element, lineIndex, ply, branchNode);
        element.onkeydown = e => {
          if (e.key === 'ContextMenu' || e.key === 'F10' && e.shiftKey) contextMenu(e, element, lineIndex, ply, branchNode);
        };
      };
      const score = make('div', 'study-score'); score.setAttribute('aria-label', 'Game score with nested variations');
      const move = (point, prefix, index, lineIndex, branchNode) => {
        const key = positionKey(prefix), current = selectedKey === key;
        const b = button(`[${point}]`, () => options.onSelect(lineIndex, prefix.length), 'score-move' + (current ? ' current sel' : ''));
        b.dataset.focus = `move:${lineIndex}:${key}`;
        b.setAttribute('aria-label', `Turn ${index ? Math.ceil(index / 2) : 0}, ${index ? index % 4 < 3 && index % 4 > 0 ? 'O' : 'X' : 'X'}, stone [${point}]`);
        if (current) b.setAttribute('aria-current', 'step');
        bindContext(b, lineIndex, prefix.length, branchNode);
        const wrap = make('span', 'score-stone'); wrap.append(b);
        const saved = study.annotations[key], info = options.info?.(prefix) || saved?.info;
        const text = annotationText(info);
        if (text) wrap.append(make('span', 'score-info', text));
        const grade = options.grade?.(prefix);
        if (grade) { const badge = make('span', 'score-grade', grade.symbol); badge.title = grade.label; badge.style.color = grade.color; wrap.append(badge); }
        if (saved?.visuals?.length) {
          const marks = make('span', 'score-marks', 'marks'); marks.title = 'Cell highlights and labels at this position'; wrap.append(marks);
        }
        return wrap;
      };
      if (study.lines.length) score.append(make('span', 'score-number', '0. X '), move([0, 0], [[0, 0]], 0, 0), doc.createTextNode(' '));
      function line(node, alternatives, container, depth, branchNode = null) {
        while (node) {
          const turn = make('span', 'score-turn' + (depth ? '' : ' score-main'));
          const number = (node.start + 1) / 2, side = number % 2 ? 'O' : 'X';
          turn.style.setProperty('--stone-color', side === 'X' ? 'var(--x)' : 'var(--o)');
          turn.append(make('span', 'score-number', `${number}. ${side} `));
          for (let j = 0; j < node.stones.length; j++) turn.append(move(node.stones[j], node.history.slice(0, node.start + j + 1), node.start + j, node.lineIndex, branchNode));
          if (node.stones.length === 1) turn.append(make('span', 'score-end', ' [/]'));
          const grade = options.turnGrade?.(node.history);
          if (grade) {
            const badge = button(`${grade.symbol} ${grade.label}`, () => options.onSelect(node.lineIndex, node.history.length), 'score-action score-turn-grade');
            badge.title = `Whole turn ${number}: ${grade.label}`;
            badge.dataset.focus = 'turn:' + node.id;
            badge.setAttribute('aria-label', badge.title);
            badge.style.color = grade.color;
            bindContext(badge, node.lineIndex, node.history.length, branchNode);
            turn.append(badge);
          }
          container.append(turn, doc.createTextNode(' '));
          for (const alternative of alternatives) {
            const branch = make('div', 'score-variation' + (depth >= 3 ? ' deep' : '') + (active(alternative) ? ' active-line' : ''));
            branch.dataset.branch = alternative.id;
            const closed = collapsed.has(alternative.id);
            const toggle = button(closed ? '+' : '−', () => {
              closed ? collapsed.delete(alternative.id) : collapsed.add(alternative.id); redraw();
            }, 'score-fold');
            toggle.setAttribute('aria-expanded', String(!closed));
            toggle.dataset.focus = 'fold:' + alternative.id;
            toggle.setAttribute('aria-label', `${closed ? 'Expand' : 'Collapse'} variation at turn ${(alternative.start + 1) / 2}`);
            branch.append(toggle);
            if (closed) { const summary = button(`${(alternative.start + 1) / 2}. ${alternative.stones.map(p => `[${p}]`).join(' ')} …`, () => {
              collapsed.delete(alternative.id); redraw();
            }, 'score-summary'); bindContext(summary, alternative.lineIndex, alternative.history.length, alternative); branch.append(summary); }
            else {
              const content = make('div', 'score-branch'); content.setAttribute('role', 'group');
              content.setAttribute('aria-label', `Variation depth ${depth + 1}`);
              content.append(doc.createTextNode('( ')); line(alternative, [], content, depth + 1, alternative); content.append(doc.createTextNode(' )')); branch.append(content);
            }
            container.append(branch);
          }
          alternatives = node.children.slice(1); node = node.children[0];
        }
      }
      if (roots.length) line(roots[0], roots.slice(1), score, 0);
      box.replaceChildren(toolbar, score);
      if (focus) Array.from(box.querySelectorAll('button')).find(b => b.dataset.focus === focus)?.focus({preventScroll: true});
    }
  };
  return view;
}
