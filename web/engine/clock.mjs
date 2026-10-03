/* Time controls for the play page, as python/time_control.py and python/play.py clock_spec define them. */

/**
 * A clock request checked and normalized: {mode: 'fixed'}, {mode: 'move', ms} (a complete turn in ms, no banking) or
 * {mode: 'game', tc} with tc '180' (Absolute) or '180+2' (Fischer) in seconds, which carries base_ms and increment_ms.
 */
export function clockSpec(clock = {mode: 'fixed'}) {
  if (clock.mode === 'fixed') return {mode: 'fixed'};
  if (clock.mode === 'move') {
    if (!Number.isFinite(clock.ms) || clock.ms <= 0) throw Error('The time per turn must be positive');
    return {mode: 'move', ms: clock.ms};
  }
  if (clock.mode === 'game') {
    if (Number.isFinite(clock.base_ms)) {
      const increment = clock.increment_ms ?? 0;
      if (clock.base_ms <= 0 || !(increment >= 0)) throw Error('Use a positive base time and no negative increment');
      return {mode: 'game', base_ms: clock.base_ms, increment_ms: increment};
    }
    const m = String(clock.tc).match(/^(\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?$/);
    if (!m || +m[1] <= 0) throw Error('Use seconds or seconds+increment, such as 180+2');
    return {mode: 'game', base_ms: +m[1] * 1000, increment_ms: +(m[2] || 0) * 1000};
  }
  throw Error('Clock mode must be fixed, move or game');
}

/**
 * The time a turn may take, as time_control.allowance gives it: the remaining time and coming increments spread over
 * `horizon` turns, at most three such shares plus the `reserve` and never more than the remaining time. `clock` holds
 * cross_ms, circle_ms and increment_ms. {normal_ms, hard_ms, reserve_ms}.
 */
export function allowance(clock, side, {horizon = 20, reserve = 10} = {}) {
  const remaining = clock[side ? 'circle_ms' : 'cross_ms'], increment = clock.increment_ms || 0;
  const usable = Math.max(0, remaining - reserve), normal = Math.min(usable, (usable + (horizon - 1) * increment) / horizon);
  const hard = Math.min(remaining, 3 * normal + reserve);
  return {normal_ms: Math.min(normal, Math.max(0, hard - reserve)), hard_ms: hard, reserve_ms: reserve};
}

/** The work time for a turn under `control` (a clockSpec) with balances `clock`: a per-turn clock's whole turn less
 * the reserve, else `allowance`'s normal share. */
export function turnTime(control, clock, side) {
  return control.mode === 'move' ? Math.max(1, clock[side ? 'circle_ms' : 'cross_ms'] - 10) : allowance(clock, side).normal_ms;
}
