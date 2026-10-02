// Node runner for tests/test_web_engine.py: reads one JSON job from stdin, writes one JSON answer to stdout.
// {kind: 'follow', enabled, available, stored, steps: [[path, body]], together} -> {requests: [[path, body]], enabled, paused, stored}
//   drives web/engine/book.mjs on a stand-in play server whose page plays 'browser:bubble' seats itself, as seat.mjs does
//   and answers /book later than /seat (turning it on for an empty board resumes the game, as a new game does);
//   `together` sends every step at once instead of one after another
// {kind: 'offline', steps: [[path, body]]} -> [path] of every request an OfflineSession answered
import {readFileSync} from 'node:fs';
import {followSeats} from '../../web/engine/book.mjs';
import {OfflineSession} from '../../web/engine/offline.mjs';

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
    return state;
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
process.stdout.write(JSON.stringify(job.kind === 'follow' ? {requests, enabled: state.book.enabled, paused: state.paused, stored: stored.get('book-touched') ?? null} : requests));
