"""The audited run4-to-run5 CUDA evaluation equivalence, limited to saved ratings.

The two source maps differ only in search_train.py and checkpoint_league.py.
In the reviewed revisions, search_train.play_games, evaluate,
evaluation_protocol, source_identity, verify_artifact and publish have identical
ASTs. The other 13 source_identity files and three native DLLs are byte-identical.
The changes select opponents, decide promotion and report results; they do not
change moves or outcomes. This allowlist does not change artifact identities.
"""
import hashlib
import json


OLD_SOURCE='8e14d6cf48dc23c45941d1d0baf2b1522efc98c315a1d8a52f5d92ee853d0031'
NEW_SOURCE='8482ae437e0496266240c5088c0362dc102de444313a70c70c00ff8c35372c22'
RUNTIME='dae74c86987d6980fc424a5299989a8596cff01819821daa60c09fd891348a28'
CHANGED={'search_train.py','checkpoint_league.py'}
EVAL_SETTINGS=('device','eval_max_plies','eval_tactics','simulations','root_samples',
               'eval_games','reference_games',
               'max_nodes','max_edges','cache_positions','leaf_batch','envs','seed')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def protocol(identity):
    config=identity['config']
    return dict(max_plies=config['eval_max_plies'],tactics=config['eval_tactics'],
                simulations=config['simulations'],root_samples=config['root_samples'],
                device=config['device'],opening_suite='standard-v1',
                source_sha256=fingerprint(identity['sources']),runtime_sha256=fingerprint(identity['runtime']))


def audited_prior(run,config,current_protocol):
    """Return the one eligible migration history, or None for any other run."""
    if fingerprint(config['sources'])!=NEW_SOURCE or fingerprint(config['runtime'])!=RUNTIME or \
       current_protocol!=protocol(config) or config['config']['device']!='cuda':return None
    expected=config.get('history_sha256');path=run/'history.json'
    if not expected:return None
    if digest(path)!=expected:raise ValueError('Rating migration history changed')
    history=json.loads(path.read_text());old=history['previous_identity']
    if fingerprint(old['sources'])!=OLD_SOURCE or fingerprint(old['runtime'])!=RUNTIME or \
       old['config']['device']!='cuda' or \
       any(old['config'][key]!=config['config'][key] for key in EVAL_SETTINGS) or \
       {key for key in old['sources'] if old['sources'][key]!=config['sources'].get(key)}!=CHANGED or \
       set(old['sources'])!=set(config['sources']):return None
    if history['prepared_target']!=dict(config=config['config'],sources=config['sources'],runtime=config['runtime']):
        raise ValueError('Rating migration target changed')
    return history


def scheduled_revision(run,path,report,manifest,hashes,current_protocol,history):
    """Select current or exactly audited prior CUDA report; validate inherited seal."""
    identity=manifest['identity'];saved=identity.get('protocol')
    if saved==current_protocol:
        if report.get('protocol')!=saved:raise ValueError('Scheduled report protocol changed')
        return 'current'
    if history is None or saved!=protocol(history['previous_identity']):return None
    a,b=report['candidate'],report['opponent'];old=history['previous_identity']
    artifact=path.parent.relative_to(run).as_posix()
    if artifact!=f'evaluation/{a:04d}-vs-{b:04d}' or a>history['through'] or \
       history['artifacts'].get(artifact)!=digest(path.parent/'manifest.json') or \
       identity.get('run')!=old or report.get('protocol')!=saved or \
       identity.get('candidate')!=hashes.get(a) or identity.get('opponent')!=hashes.get(b) or \
       identity.get('games')!=report['metrics']['planned_games'] or \
       identity.get('simulations')!=old['config']['simulations'] or \
       identity.get('root_samples')!=old['config']['root_samples'] or \
       identity.get('opening_suite')!='standard-v1' or \
       identity.get('seed')!=old['config']['seed']+100000+a*1000+b*1000003 or \
       manifest['files'].get('report.json')!=digest(path):
        raise ValueError(f'Audited prior evaluation changed: {path}')
    return 'audited compatible'
