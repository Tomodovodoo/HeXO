/* The play server's game state for a page served without it (a static host such as GitHub Pages): answers the
 * page's /state and game requests from this browser, with the rules of gumbel.wasm. Both seats are people to it;
 * seat.mjs plays the browser engines on them. Requests for server-only features answer 501. */
import createModule from './gumbel.mjs';
import {Native} from './search.mjs';

const GAME = new Set(['/state', '/play', '/undo', '/new', '/seat', '/analysis', '/analyse', '/pause', '/retry', '/cancel']);
const SERVER = ['/review', '/rescan', '/import', '/export', '/match', '/matches', '/htttx', '/replay', '/evaluations', '/models', '/book'];
const playerAt = ply => ply === 0 ? 0 : ((ply - 1 >> 1) + 1) % 2;

export class OfflineSession {
  static async create(analysis) {
    return new OfflineSession(new Native(await createModule()), analysis);
  }

  /** `analysis` is the analysis entry the page shows ({engine, preset, budget}). */
  constructor(native, analysis) {
    this.native = native;
    this.analysis = {checkpoint: null, auto: false, ...analysis};
    this.history = [];
    this.paused = false;
    this.revision = 1;
  }

  state() {
    const {winner, player, remaining} = this.native.game(this.history);
    return {instance: 'offline', revision: this.revision, history: this.history.map(p => [...p]), player, remaining, winner,
      paused: this.paused, seats: [{engine: 'human'}, {engine: 'human'}], analysis: this.analysis, engines: [], match: null,
      clock: null, saved_game: null, models_folder: null, book: {available: false, enabled: false}, evaluations: {}, review: [],
      review_preset: 'standard', jobs: []};
  }

  /** Whether `path` (absolute, as the page requests it) is a play server request. */
  static handles(path) {
    path = path.replace(/^\/study/, '');
    return GAME.has(path) || SERVER.some(p => path === p || path.startsWith(p + '/'));
  }

  /** [status, body] for one request of the page. */
  answer(path, body = {}) {
    path = path.replace(/^\/study/, '');
    if (!GAME.has(path)) return [501, {error: 'This needs the Bubble server (python python/bubble.py play)'}];
    try {
      this.apply(path, body);
    } catch (error) {
      return [400, {error: String(error.message || error)}];
    }
    return [200, this.state()];
  }

  apply(path, body) {
    if (path === '/play') {
      const point = [body.q, body.r];
      if (!Number.isInteger(point[0]) || !Number.isInteger(point[1])) throw new Error('Coordinates must be integers');
      if (this.native.game(this.history).winner >= 0) throw new Error('The game has finished');
      this.native.game([...this.history, point]);
      this.history.push(point);
      this.paused = false;
    } else if (path === '/undo' && this.history.length) {
      const people = Array.isArray(body.people) ? body.people : [0, 1];
      this.history.pop();
      while (people.length && this.history.length) {
        const n = this.history.length;
        if (people.includes(playerAt(n)) && (n === 0 || n % 2 === 1)) break;
        this.history.pop();
      }
    } else if (path === '/new' || (path === '/retry' && Number.isInteger(body.ply))) {
      this.history = path === '/new' ? [] : this.history.slice(0, Math.max(0, body.ply));
      this.paused = false;
    }
    else if (path === '/pause') this.paused = Boolean(body.paused);
    else if (path === '/analysis' && body.preset) this.analysis = {...this.analysis, preset: body.preset};
    if (path !== '/state' && path !== '/analyse' && path !== '/cancel') this.revision++;
  }
}
