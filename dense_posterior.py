"""Posterior over league ratings with per-pair matchup deviations, and the value of information of the next round.

Model: a result is the pentanomial of a against b: counts of colour-swapped opening pairs by a's points over the
pair (0, 1/2, 1, 3/2, 2; a capped game is half a point). Every id but the anchor has a rating r (Elo) with a
N(r_parent, RATING_PRIOR^2) prior, its parent being the nearest earlier rated checkpoint of its variant (for a
variant `<checkpoint>@name`, that checkpoint), N(0, RATING_PRIOR^2) without one; the anchor is fixed at 0. Every
pair that met has a matchup deviation d_ab ~ N(0, matchup_prior^2) (none when matchup_prior is 0), and a's expected
score against b is 1 / (1 + 10^(-(r_a - r_b + d_ab) / 400)), so a pair's own games outweigh the transitive picture
when the two disagree. An opening pair is one observation: the games of a pair are correlated through their opening,
so each player pair's binomial likelihood of w points in n games is a quasi-likelihood divided by its dispersion
phi, the variance of the pair points over the variance 2p(1-p) two independent games would have at the observed
score p (`dispersion`). Its effective pair count is pairs / phi. `Posterior` finds the mode by Newton's method and
approximates the posterior by the Gaussian with the inverse Hessian as covariance (Laplace). `after` is the
posterior variance of a quantity after one more round of a pairing, from that round's expected Fisher information at
the mode: the Laplace form of averaging the updated variance over the posterior predictive outcomes of the round.
"""
import math

import numpy as np

K = math.log(10)/400
RATING_PRIOR = 1000.
MODEL = 'pentanomial'
DISPERSION_PRIOR_PAIRS = 4.  # pseudo-pairs of independent games that the dispersion estimate is shrunk toward


def dispersion(counts):
    """phi of a pentanomial `counts` (pairs by points 0, 1/2, 1, 3/2, 2; not necessarily whole): the variance of
    the pair points about their mean, shrunk toward the independent 2p(1-p) by DISPERSION_PRIOR_PAIRS pseudo-pairs,
    over 2p(1-p) at p = mean / 2. 1 without pairs or when every pair scored 0 or every pair 2; below 1 when pairs
    split more often than independent games would (the opening decides the colour, not the player), above 1 when
    they sweep more often (the opening favours one player)."""
    counts = np.asarray(counts, float)
    pairs, points = counts.sum(), np.arange(5)/2
    if pairs <= 0:
        return 1.
    mean = float(counts@points)/pairs
    independent = mean*(1-mean/2)
    if independent <= 0:
        return 1.
    variance = float(counts@(points-mean)**2)/pairs
    return (pairs*variance+DISPERSION_PRIOR_PAIRS*independent)/((pairs+DISPERSION_PRIOR_PAIRS)*independent)


def parents(ids):
    """{id: parent id} over league ids (module contract): a variant `<checkpoint>@<name>` -> its checkpoint when
    that is in `ids`; a checkpoint `<variant>/<step>` -> the checkpoint of `ids` of the same variant with the
    highest smaller step. Other ids (Seal) and the first checkpoint of a variant have none."""
    out, lines = {}, {}
    for name in ids:
        checkpoint, at, _ = name.partition('@')
        variant, slash, step = checkpoint.partition('/')
        if at:
            if checkpoint in ids:
                out[name] = checkpoint
        elif slash and step.isdigit():
            lines.setdefault(variant, []).append((int(step), name))
    for line in lines.values():
        line.sort()
        out.update({later: earlier for (_, earlier), (_, later) in zip(line, line[1:])})
    return out


class Posterior:
    """Laplace posterior of `ids` (anchor fixed at 0) from results [(a, b, pentanomial counts of a)], summed per
    player pair (module contract).
    `parents` {id: parent id} centres an id's N(., RATING_PRIOR^2) prior on its parent's rating (the prior is on
    r_id - r_parent), so an id without games sits at its parent's rating; an id without a parent in `ids` keeps
    the prior centred on 0."""

    def __init__(self, ids, anchor, results, matchup_prior, parents=None):
        self.anchor, self.sigma = anchor, matchup_prior
        self.ids = list(dict.fromkeys(ids))
        totals = {}
        for a, b, counts in results:
            key, counts = ((a, b), np.asarray(counts, float)) if a < b else ((b, a), np.asarray(counts, float)[::-1])
            totals[key] = totals.get(key, 0.)+counts
        self.phi = {k: dispersion(c) for k, c in totals.items() if c.sum() > 0}
        self.effective = {k: float(totals[k].sum())/phi for k, phi in self.phi.items()}
        self.pairs = {k: (float(totals[k]@np.arange(5))/2/phi, 2*float(totals[k].sum())/phi)
                      for k, phi in self.phi.items()}
        free = [i for i in self.ids if i != anchor]
        self.index = {i: k for k, i in enumerate(free)}
        if matchup_prior > 0:
            self.index.update({p: len(free)+k for k, p in enumerate(self.pairs)})
        prior = np.diag([0.]*len(free)+[matchup_prior**-2 if matchup_prior > 0 else 0.]*(len(self.index)-len(free)))
        parents = parents or {}
        for name in free:
            parent = parents.get(name)
            v = self.vector(name, parent, False) if parent in self.ids and parent != name else self.vector(name, anchor, False)
            prior += RATING_PRIOR**-2*np.outer(v, v)
        design = np.array([self.vector(a, b) for a, b in self.pairs]).reshape(len(self.pairs), len(self.index))
        w, n = (np.array([v[i] for v in self.pairs.values()]) for i in (0, 1))
        x = np.zeros(len(self.index))
        for _ in range(100):
            p = 1/(1+np.exp(-np.clip(K*(design@x), -700, 700)))
            gradient = K*design.T@(n*p-w)+prior@x
            hessian = K*K*design.T@(design*(n*p*(1-p))[:, None])+prior
            step = np.linalg.solve(hessian, gradient)
            x -= step
            if np.max(np.abs(step)) < 1e-6:
                break
        p = 1/(1+np.exp(-np.clip(K*(design@x), -700, 700)))
        self.mode = x
        self.cov = np.linalg.inv(K*K*design.T@(design*(n*p*(1-p))[:, None])+prior)

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

    def effective_pairs(self, a, b):
        """Opening pairs of a against b divided by their dispersion (module contract); 0 when they never met."""
        return self.effective.get((a, b) if a < b else (b, a), 0.)

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
        """Posterior variance of the target difference (a, b, matchup) after `games` more games of pairing (x, y),
        at the dispersion of the pairing's games so far (1 when it never met)."""
        a, b, matchup = target
        new = self.new(*([(a, b)] if matchup else []), pairing)
        cov = self.covariance(new)
        g, v = self.vector(a, b, matchup, new), self.vector(*pairing, True, new)
        mean = float(v[:len(self.mode)]@self.mode)
        p = 1/(1+math.exp(-K*mean))
        info = K*K*games*p*(1-p)/self.phi.get(tuple(sorted(pairing)), 1.)
        variance, shared = float(g@cov@g), float(g@cov@v)
        return variance-info*shared**2/(1+info*float(v@cov@v))
