"""Paired scores and sequential tests, independent of model inference."""
import math

import numpy as np

# Settings a reused report must share; a report without one of PROTOCOL_DEFAULTS was played at that value.
PROTOCOL = ('sims', 'root_samples', 'max_plies', 'tactics', 'search_graph', 'search_choice', 'q_range_floor', 'opening_suite', 'opening_book', 'seal_ms',
            'external_engine', 'external_name',
            'solver_root_nodes', 'solver_finalists', 'solver_finalist_nodes', 'solver_threat_nodes',
            'solver_defence', 'solver_defence_candidates', 'solver_gate_cap_nodes', 'pipeline')
PROTOCOL_DEFAULTS = dict(opening_book='', search_graph=False, search_choice='gumbel', q_range_floor=0., external_engine='', external_name='seal', solver_root_nodes=0, solver_finalists=0, solver_finalist_nodes=0,
                         solver_threat_nodes=0, solver_defence=False, solver_defence_candidates=8,
                         solver_gate_cap_nodes=0, pipeline=False)


def same_protocol(report_settings, settings):
    """Whether a report with `report_settings` was played under the PROTOCOL of `settings` (a dict, such as the
    evaluator's published status settings). Under the live book that includes opening_book (dense_openings.Book.digest
    of its openings), which changes only at a book refresh: a report is reused while the book keeps its openings,
    and a refresh that changes them starts every comparison afresh. A frozen suite's name fixes its openings
    (opening_book ''); a report without a PROTOCOL_DEFAULTS field was played at its default."""
    return all(report_settings.get(k, PROTOCOL_DEFAULTS.get(k)) == settings.get(k) for k in PROTOCOL)


def pair_scores(records):
    """{pair seed: candidate points / 2} with wins 1, caps 1/2, losses 0 per game; pairs must be intact."""
    pairs = {}
    for g in records:
        pairs.setdefault(g['seed'], []).append(1. if g['winner'] == g['challenger_color'] else .5 if g['winner'] < 0 else 0.)
    if any(len(p) != 2 for p in pairs.values()):
        raise ValueError('Rating requires intact colour-swapped opening pairs')
    return {seed: sum(p)/2 for seed, p in pairs.items()}

def pentanomial(records):
    """[pairs by candidate points 0, 1/2, 1, 3/2, 2] over the colour-swapped opening pairs of `records` (`pair_scores`)."""
    counts = [0]*5
    for score in pair_scores(records).values():
        counts[round(4*score)] += 1
    return counts

def summary(records):
    """Draw-aware match summary: decisive wins/losses, caps, pair score, its paired Hoeffding lower bound
    and an Elo delta from the points with a +1/2 continuity correction."""
    scores = list(pair_scores(records).values())
    wins = sum(g['winner'] == g['challenger_color'] for g in records)
    capped = sum(g['winner'] < 0 for g in records)
    losses = len(records)-wins-capped
    score = sum(scores)/len(scores)
    points = wins+capped/2
    return dict(wins=wins, losses=losses, capped=capped, games=len(records), pair_score=score,
                pair_score_lower=score-math.sqrt(math.log(40)/(2*len(scores))),
                elo_delta=400*math.log10((points+.5)/(len(records)-points+.5)))

def sprt(records, elo0, elo1, alpha, beta):
    """Generalized SPRT (Van den Bergh; Fishtest's pentanomial test) on colour-swapped opening pairs.

    Each pair is one observation x in {0, 1/4, 1/2, 3/4, 1}: the candidate's points over its two games / 2,
    with caps half a point (without caps this is the trinomial 0/1/2 wins per pair), so within-pair
    correlation never inflates the evidence. Let p^ be the empirical category frequencies over N pairs
    (plus 1e-3 pseudo-counts so every category is supported) and s_j = 1/(1 + 10^(-elo_j/400)) the expected
    score under H_j (candidate = champion + elo_j). p_j is the maximum-likelihood category distribution with
    mean s_j: p_j,k = p^_k / (1 + l_j (x_k - s_j)), with l_j solving sum_k p^_k (x_k - s_j)/(1 + l_j (x_k - s_j)) = 0.
    LLR = N sum_k p^_k log(p_1,k / p_0,k). H1 (promote) when LLR >= log((1-beta)/alpha), H0 when
    LLR <= log(beta/(1-alpha)), otherwise decision None.
    """
    x, counts = np.arange(5)/4, np.array(pentanomial(records), float)
    n = counts.sum()
    phat = (counts+1e-3)/(n+5e-3)

    def fit(elo):
        d = x-1/(1+10**(-elo/400))
        low, high = -1/d.max()+1e-12, -1/d.min()-1e-12
        for _ in range(200):
            lam = (low+high)/2
            if np.sum(phat*d/(1+lam*d)) > 0:
                low = lam
            else:
                high = lam
        return phat/(1+lam*d)

    llr = float(n*np.sum(phat*np.log(fit(elo1)/fit(elo0))))
    lower, upper = math.log(beta/(1-alpha)), math.log((1-beta)/alpha)
    return dict(llr=llr, bound_lower=lower, bound_upper=upper, games=int(2*n), elo0=elo0, elo1=elo1, alpha=alpha, beta=beta,
                pair_counts=counts.astype(int).tolist(), decision='H1' if llr >= upper else 'H0' if llr <= lower else None)

def tally(records, test=None):
    """Running score of a comparison's finished games: wins, losses, capped and games over all of them; over its
    complete opening pairs: pairs, pair_score, pair_interval (`summary`'s paired Hoeffding 95% bounds, clipped to
    [0, 1]), elo_delta and elo_interval (`summary`'s continuity-corrected Elo of the points at the score and at its
    bounds), and llr, bound_lower and bound_upper of test(complete records) (an `sprt` result) when given. Pair
    fields are None before the first complete pair."""
    wins = sum(g['winner'] == g['challenger_color'] for g in records)
    capped = sum(g['winner'] < 0 for g in records)
    out = dict(wins=wins, losses=len(records)-wins-capped, capped=capped, games=len(records), pairs=0, pair_score=None,
               pair_interval=None, elo_delta=None, elo_interval=None, llr=None, bound_lower=None, bound_upper=None)
    seeds = [g['seed'] for g in records]
    complete = [g for g in records if seeds.count(g['seed']) == 2]
    if complete:
        s, n = summary(complete), len(complete)
        half = s['pair_score']-s['pair_score_lower']
        bounds = [max(0., s['pair_score']-half), min(1., s['pair_score']+half)]
        elo = lambda p: 400*math.log10((p*n+.5)/(n-p*n+.5))
        out.update(pairs=n//2, pair_score=s['pair_score'], pair_interval=bounds, elo_delta=s['elo_delta'],
                   elo_interval=[elo(p) for p in bounds])
        if test:
            result = test(complete)
            out.update(llr=result['llr'], bound_lower=result['bound_lower'], bound_upper=result['bound_upper'])
    return out
