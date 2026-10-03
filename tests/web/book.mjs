// Node runner for tests/test_web_engine.py: reads one JSON job from stdin, writes one JSON answer to stdout.
// {kind: 'follow', enabled, available, stored, steps: [[path, body]], together} -> {requests: [[path, body]], enabled, paused, stored}
//   drives web/engine/book.mjs on a stand-in play server whose page plays 'browser:bubble' seats itself, as seat.mjs does
//   and answers /book later than /seat (turning it on for an empty board resumes the game, as a new game does on both
//   pages);
//   `together` sends every step at once instead of one after another
// {kind: 'offline', steps: [[path, body]]} -> [path] of every request an OfflineSession answered
// {kind: 'static', steps: [[path, body]]} -> {requests: [path], enabled, paused, stones} from the static page's BrowserSession,
//   which starts paused with Human against an engine (one that keeps thinking) and the book on
import {readFileSync} from 'node:fs';
import {followSeats} from '../../web/engine/book.mjs';
import {OfflineSession} from '../../web/engine/offline.mjs';
import {BrowserSession} from '../../web/engine/play-session.mjs';
import {OpeningBook} from '../../web/engine/openings.mjs';
import createModule from '../../web/engine/gumbel.mjs';
import {Native} from '../../web/engine/search.mjs';

const job = JSON.parse(readFileSync(0, 'utf8'));
const requests = [], browser = new Set(), stored = new Map(job.stored ? [['book-touched', job.stored]] : []);
const storage = {getItem: key => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value)};
let page, state, human;

if (job.kind === 'follow') {
  state = {seats: [{engine: 'human'}, {engine: 'six'}], book: {available: job.available, enabled: job.enabled}, history: [], paused: false};
  human = seat => seat.engine === 'human' && !browser.has(state.seats.indexOf(seat));
  page = {post: async (path, body) => {
    requests.push([path, body]);
    if (path === '/seat') {
      const mine = body.engine === 'browser:bubble';
      mine ? browser.add(body.side) : browser.delete(body.side);
      state = {...state, seats: state.seats.map((seat, side) => side === body.side ? {engine: mine ? 'human' : body.engine} : seat)};
    } else if (path === '/book') {
      await new Promise(done => setTimeout(done, 20));
      state = {...state, book: {...state.book, enabled: body.enabled}, paused: state.paused && !body.enabled};
    } else if (path === '/pause') state = {...state, paused: body.paused};
    else if (path === '/new') state = {...state, paused: false};
    return state;
  }};
} else if (job.kind === 'static') {
  const session = new BrowserSession(new Native(await createModule()));
  session.bookData = new OpeningBook(JSON.parse(readFileSync(new URL('../../web/engine/openings.json', import.meta.url))));
  const entry = {id: 'browser:test', name: 'Test', kind: 'bubble', version: 'v1', checkpoints: [], presets: {standard: {simulations: 1, solver_nodes: 0}}};
  session.registerEngine(entry, {turn: () => new Promise(() => {})});
  session.seats = [{engine: 'human'}, session.spec({engine: entry.id, preset: 'standard'})];
  session.book.enabled = true;
  session.paused = true;
  state = session.state();
  human = seat => seat.engine === 'human';
  page = {post: async (path, body) => {
    requests.push(path);
    const [status, data] = await session.request(path, body, 'POST');
    return status === 200 ? state = session.state() : null;
  }};
} else {
  const session = new OfflineSession({game: () => ({winner: -1, player: 0, remaining: 1})}, {engine: 'browser:bubble'});
  state = session.state();
  human = seat => seat.engine === 'human';
  page = {post: async (path, body) => {
    requests.push(path);
    const [status, data] = session.answer(path, body);
    if (status !== 200) return null;
    return state = data;
  }};
}
followSeats(page, () => state, human, storage);
if (job.together) await Promise.all(job.steps.map(([path, body]) => page.post(path, body)));
else for (const [path, body] of job.steps) await page.post(path, body);
process.stdout.write(JSON.stringify(job.kind === 'static' ? {requests, enabled: state.book.enabled, paused: state.paused, stones: state.history.length}
  : job.kind === 'follow' ? {requests, enabled: state.book.enabled, paused: state.paused, stored: stored.get('book-touched') ?? null} : requests));
