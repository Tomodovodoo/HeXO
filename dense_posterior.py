"""Posterior over league ratings with per-pair matchup deviations, and the value of information of the next round.

Model: a report is n games of a against b with a's points w (a capped game is half a point). Every id but the
anchor has a rating r (Elo) with a N(0, RATING_PRIOR^2) prior; the anchor is fixed at 0. Every pair that met has
a matchup deviation d_ab ~ N(0, matchup_prior^2) (none when matchup_prior is 0), and a's expected score against b
is 1 / (1 + 10^(-(r_a - r_b + d_ab) / 400)), so a pair's own games outweigh the transitive picture when the two
disagree. `Posterior` finds the mode by Newton's method and approximates the posterior by the Gaussian with the
inverse Hessian as covariance (Laplace). `after` is the posterior variance of a quantity after one more round of
a pairing, from that round's expected Fisher information at the mode: the Laplace form of averaging the updated
variance over the posterior predictive outcomes of the round.
"""
import math

import numpy as np

K = math.log(10)/400
RATING_PRIOR = 1000.


class Posterior:
    """Laplace posterior of `ids` (anchor fixed at 0) from results [(a, b, points of a, games)], summed per pair."""

    def __init__(self, ids, anchor, results, matchup_prior):
        self.anchor, self.sigma = anchor, matchup_prior
        self.ids = list(dict.fromkeys(ids))
        totals = {}
        for a, b, w, n in results:
            key, w = ((a, b), w) if a < b else ((b, a), n-w)
            total = totals.setdefault(key, [0., 0.])
            total[0] += w; total[1] += n
        self.pairs = {k: v for k, v in totals.items() if v[1] > 0}
        free = [i for i in self.ids if i != anchor]
        self.index = {i: k for k, i in enumerate(free)}
        if matchup_prior > 0:
            self.index.update({p: len(free)+k for k, p in enumerate(self.pairs)})
        precision = np.array([RATING_PRIOR**-2]*len(free)+[matchup_prior**-2 if matchup_prior > 0 else 0.]*(len(self.index)-len(free)))
        design = np.array([self.vector(a, b) for a, b in self.pairs]).reshape(len(self.pairs), len(self.index))
        w, n = (np.array([v[i] for v in self.pairs.values()]) for i in (0, 1))
        x = np.zeros(len(self.index))
        for _ in range(100):
            p = 1/(1+np.exp(-np.clip(K*(design@x), -700, 700)))
            gradient = K*design.T@(n*p-w)+precision*x
            hessian = K*K*design.T@(design*(n*p*(1-p))[:, None])+np.diag(precision)
            step = np.linalg.solve(hessian, gradient)
            x -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        p = 1/(1+np.exp(-np.clip(K*(design@x), -700, 700)))
        self.mode = x
        self.cov = np.linalg.inv(K*K*design.T@(design*(n*p*(1-p))[:, None])+np.diag(precision))

    def vector(self, a, b, matchup=True, extra=()):
        """Weights of r_a - r_b (+ d_ab with matchup) over the parameters, then over `extra` new pairs."""
        v = np.zeros(len(self.index)+len(extra))
        for name, sign in ((a, 1.), (b, -1.)):
            if name in self.index:
                v[self.index[name]] += sign
        key = (a, b) if a < b else (b, a)
        if matchup and self.sigma > 0:
            sign = 1. if a < b else -1.
            if key in self.index:
                v[self.index[key]] += sign
            elif key in extra:
                v[len(self.index)+extra.index(key)] += sign
        return v

    def rating(self, name):
        return 0. if name == self.anchor else float(self.mode[self.index[name]])

    def difference(self, a, b, matchup=True):
        """(mean, sd) of r_a - r_b, plus the matchup deviation d_ab with `matchup` (its prior when a and b never
        met)."""
        new = self.new((a, b)) if matchup else ()
        v = self.vector(a, b, matchup, new)
        return float(v[:len(self.mode)]@self.mode), math.sqrt(max(0., float(v@self.covariance(new)@v)))

    def spread(self, name):
        """Posterior sd of r_name minus the mean rating of `ids`, which no choice of anchor changes."""
        v = np.zeros(len(self.index))
        for other in self.ids:
            if other in self.index:
                v[self.index[other]] -= 1/len(self.ids)
        if name in self.index:
            v[self.index[name]] += 1.
        return math.sqrt(max(0., float(v@self.cov@v)))

    def new(self, *pairs):
        """The matchup deviations among `pairs` that are not parameters yet (pairs that never met)."""
        keys = [(a, b) if a < b else (b, a) for a, b in pairs]
        return tuple(dict.fromkeys(k for k in keys if self.sigma > 0 and k not in self.index))

    def covariance(self, new):
        size = len(self.index)
        out = np.zeros((size+len(new), size+len(new)))
        out[:size, :size] = self.cov
        out[size:, size:] = np.eye(len(new))*self.sigma**2
        return out

    def after(self, target, pairing, games):
        """Posterior variance of the target difference (a, b, matchup) after `games` more games of pairing (x, y)."""
        a, b, matchup = target
        new = self.new(*([(a, b)] if matchup else []), pairing)
        cov = self.covariance(new)
        g, v = self.vector(a, b, matchup, new), self.vector(*pairing, True, new)
        mean = float(v[:len(self.mode)]@self.mode)
        p = 1/(1+math.exp(-K*mean))
        info = K*K*games*p*(1-p)
        variance, shared = float(g@cov@g), float(g@cov@v)
        return variance-info*shared**2/(1+info*float(v@cov@v))
