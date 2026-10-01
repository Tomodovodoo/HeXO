"""Absolute/Fischer clocks and turn allowances. All public times are milliseconds."""
from dataclasses import dataclass
import math
import time


def milliseconds(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} time')
    return float(value)


@dataclass(frozen=True)
class TimeControl:
    base_ms: float
    increment_ms: float = 0

    def __post_init__(self):
        milliseconds(self.base_ms, 'base_ms', positive=True)
        milliseconds(self.increment_ms, 'increment_ms')

    @classmethod
    def parse(cls, value):
        if isinstance(value, dict):
            return cls(value['base_ms'], value.get('increment_ms', 0))
        parts = str(value).split('+')
        if len(parts) not in (1, 2):
            raise ValueError('Time control must be seconds or seconds+increment')
        return cls(float(parts[0])*1000, float(parts[1])*1000 if len(parts) == 2 else 0)

    def text(self):
        base, inc = self.base_ms/1000, self.increment_ms/1000
        return f'{base:g}+{inc:g}' if inc else f'{base:g}'

    def json(self):
        return dict(base_ms=self.base_ms, increment_ms=self.increment_ms)


def allowance(clock=None, side=0, movetime=None, *, horizon=20, reserve_ms=10):
    """Return work and inclusive hard budgets; consumers remove the response reserve once."""
    reserve_ms = milliseconds(reserve_ms, 'reserve_ms')
    if horizon < 1:
        raise ValueError('horizon must be positive')
    if clock is None:
        hard = milliseconds(1000 if movetime is None else movetime, 'movetime')
        return dict(normal_ms=max(0, hard-reserve_ms), hard_ms=hard, reserve_ms=reserve_ms)
    remaining = milliseconds(clock['cross_ms' if side == 0 else 'circle_ms'], 'remaining clock')
    increment = milliseconds(clock.get('increment_ms', 0), 'increment_ms')
    usable = max(0, remaining-reserve_ms)
    normal = min(usable, (usable+(horizon-1)*increment)/horizon)
    hard = min(remaining, 3*normal+reserve_ms)
    if movetime is not None:
        hard = min(hard, milliseconds(movetime, 'movetime'))
    return dict(normal_ms=min(normal, max(0, hard-reserve_ms)), hard_ms=hard, reserve_ms=reserve_ms)


class Clock:
    """Host-owned clock. A placement does not change clocks until the turn completes."""
    def __init__(self, control, now=time.monotonic_ns):
        self.control = TimeControl.parse(control)
        self.now = now
        self.balances = [int(self.control.base_ms*1_000_000)]*2
        self.running = None
        self.started = None

    def start(self, side):
        if self.running is not None:
            raise ValueError('Clock already running')
        self.running, self.started = side, self.now()

    def remaining(self, at=None):
        values = self.balances.copy()
        if self.running is not None:
            values[self.running] -= (self.now() if at is None else at)-self.started
        return values

    def expired(self, at=None):
        return self.running is not None and self.remaining(at)[self.running] <= 0

    def stop(self, *, completed=False, at=None):
        if self.running is None:
            return 0
        at = self.now() if at is None else at
        side = self.running
        elapsed = at-self.started
        self.balances[side] -= elapsed
        if completed and self.balances[side] > 0:
            self.balances[side] += int(self.control.increment_ms*1_000_000)
        self.running = self.started = None
        return elapsed

    def json(self, at=None):
        values = self.remaining(at)
        return dict(cross_ms=max(0, values[0]/1_000_000), circle_ms=max(0, values[1]/1_000_000),
                    increment_ms=self.control.increment_ms,
                    running=None if self.running is None else ('x', 'o')[self.running])
