"""Connected Bradley-Terry ratings from completed internal comparisons."""
import math
import numpy as np

RATING_METHOD = 'Bradley-Terry projection of paired-outcome posterior means; checkpoint 0 fixed at 0'


def solve_ratings(ids, edges):
    ids = sorted(ids)
    connected = {0}
    while True:
        previous = set(connected)
        for a, b, _, _ in edges:
            if a in connected or b in connected:
                connected.update((a, b))
        if connected == previous:
            break
    variables = [i for i in ids if i in connected and i != 0]
    index = {number:i for i,number in enumerate(variables)}
    x = np.zeros(len(variables))
    rows = []
    for a,b,w,n in edges:
        if a not in connected:
            continue
        row = np.zeros(len(variables))
        if a:row[index[a]] = 1.
        if b:row[index[b]] = -1.
        rows.append((row,w,n))
    if rows:
        matrix=np.asarray([row for row,_,_ in rows]);wins=np.asarray([w for _,w,_ in rows]);totals=np.asarray([n for _,_,n in rows])
    def objective(values):
        d=matrix@values
        return float(np.sum(totals*np.logaddexp(0.,d)-wins*d))
    for _ in range(100):
        if not len(x):break
        p=1./(1.+np.exp(-np.clip(matrix@x,-700.,700.)))
        gradient=matrix.T@(totals*p-wins)
        hessian=matrix.T@(matrix*(totals*p*(1.-p))[:,None])
        if np.max(np.abs(gradient))<1e-9:break
        step=np.linalg.solve(hessian,gradient);scale=1.;old=objective(x)
        while objective(x-scale*step)>old and scale>1e-8:scale*=.5
        x-=scale*step
        if np.max(np.abs(scale*step))<1e-9:break
    ratings={number:None for number in ids};ratings[0]=0.
    ratings.update({number:float(x[i]*400./math.log(10.)) for number,i in index.items()})
    return ratings


def paired_posteriors(ids,reports):
    comparisons=[]
    for report in reports:
        if report['metrics']['incomplete'] or report['metrics'].get('pending'):continue
        a,b=report['candidate'],report['opponent']
        if a not in ids or b not in ids:continue
        if a==b:raise ValueError('A checkpoint cannot rate itself')
        pairs={}
        for game in report['games']:pairs.setdefault(game['pair'],[]).append(game)
        counts=np.zeros(3)
        for games in pairs.values():
            if len(games)!=2 or {g['challenger_color'] for g in games}!={0,1} or games[0]['opening']!=games[1]['opening']:
                raise ValueError('Rating requires intact color-swapped opening pairs')
            if any(g['winner'] not in (0,1) for g in games):raise ValueError('Nonterminal game in completed comparison')
            counts[sum(g['winner']==g['challenger_color'] for g in games)]+=1
        if (counts[1]+2*counts[2],counts[1]+2*counts[0])!=(report['metrics']['wins'],report['metrics']['losses']):
            raise ValueError('Paired outcomes disagree with reported score')
        if counts.sum():comparisons.append((a,b,counts+.5,2.*counts.sum()))
    return comparisons


def rate_league(ids,reports,samples=2048,seed=1740):
    """95% Bayesian intervals for the Elo projection, treating each opening pair as one observation.

    Every comparison has a Jeffreys Dirichlet(1/2,1/2,1/2) prior over
    zero, one or two candidate wins per pair. Posterior draws retain
    within-pair dependence, including uncertainty after unanimous results.
    The fixed comparison game counts weight the Bradley-Terry projection.
    These are credible intervals under this model, not frequentist guarantees.
    """
    ids=sorted(ids);comparisons=paired_posteriors(ids,reports)
    def edge(a,b,p,n):return a,b,n*(p[1]+2*p[2])/2.,n
    point=solve_ratings(ids,[edge(a,b,alpha/alpha.sum(),n) for a,b,alpha,n in comparisons])
    rng=np.random.default_rng(seed);draws={number:[] for number in ids if point[number] is not None}
    for _ in range(samples):
        ratings=solve_ratings(ids,[edge(a,b,rng.dirichlet(alpha),n) for a,b,alpha,n in comparisons])
        for number in draws:draws[number].append(ratings[number])
    intervals={number:np.quantile(values,[.025,.975]).tolist() for number,values in draws.items()}
    return point,intervals


def evaluation_schedule(iteration, champion, games, reference_games):
    previous=iteration-1
    older=promotion_older(iteration,champion)
    opponents=[]
    for opponent in (previous,champion,older,0):
        if opponent is not None and opponent not in opponents:opponents.append(opponent)
    required={previous,champion,older}
    return [(opponent, games if opponent in required else reference_games)
            for opponent in opponents]


def promotion_older(iteration, champion):
    """Nearest checkpoint about 20% behind the previous one, distinct from incumbent."""
    previous=iteration-1;target=round(previous*.8)
    choices=[number for number in range(previous) if number!=champion]
    return min(choices,key=lambda number:(abs(number-target),number)) if choices else None
