"""Transparent cognitive-history extension of frozen MICA v2.

No external KT predictions or architectures enter this module. Four affect
outputs retain their frozen ASSISTments detector supervision and meanings.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(os.environ.get('MICA_ROOT', '/root/autodl-tmp/research'))
DATA = ROOT/'data/processed/eckt_sequences_full'
OUT = ROOT/'reports/mica_transparent_20260923'
SRC = Path(__file__).resolve().parent
LEGACY = ROOT/'reports/mica_history_20260917_v2'
STATES = ('Concentration', 'Confusion', 'Frustration', 'Boredom')
TRACE_NAMES = [f'{g}_{n}' for g in ('student', 'skill', 'item') for n in (
    'rasch_slow', 'rasch_medium', 'rasch_fast', 'evidence_ema2',
    'evidence_ema10', 'evidence_ema50', 'cumulative_difficulty_adjusted_evidence')]
TRACE_NAMES += ['training_item_difficulty', 'student_ability_x_difficulty',
                'student_x_skill_ability', 'same_skill_as_previous',
                'same_item_as_previous', 'same_skill_previous_answer', 'difficulty_change']
PREFIX_NAMES = [f'{g}_{n}' for g in ('student', 'skill', 'item') for n in (
    'log_success', 'log_failure', 'beta_log_odds', 'previous_answer', 'log_gap', 'seen')]

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def positions(split):
    offsets = np.load(DATA/f'{split}_offsets.npy')
    windows = np.load(DATA/f'{split}_windows.npy')
    return np.concatenate([np.arange(offsets[u]+s+1, offsets[u]+s+n) for u,s,n in windows])

def legacy_module():
    # Historical modules set CUDA_VISIBLE_DEVICES during import. Restore the
    # caller's selection before any new CUDA computation.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    sys.path.insert(0, str(ROOT/'src/mica_history_20260917_v2'))
    import develop
    if visible is None: os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else: os.environ['CUDA_VISIBLE_DEVICES'] = visible
    return develop

def state_library():
    lib = OUT/'state_features.so'; digest = sha(SRC/'state_features.cpp')
    marker = OUT/'state_library.json'
    if not lib.exists() or not marker.exists() or json.loads(marker.read_text())['source_sha256'] != digest:
        subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(SRC/'state_features.cpp'), '-o', str(lib)], check=True)
        write(marker, {'source_sha256': digest, 'library_sha256': sha(lib)})
    fn = ctypes.CDLL(str(lib)).mica_states
    fn.argtypes = [ctypes.c_int64, np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS'),
        ctypes.c_int32, ctypes.c_int32, np.ctypeslib.ndpointer(np.float32, flags='C_CONTIGUOUS')]
    fn.restype = ctypes.c_int
    return fn

def trace_features(offsets, skills, items, answers, difficulty, meta):
    arrays = [np.ascontiguousarray(offsets, dtype=np.int64), np.ascontiguousarray(skills, dtype=np.int32),
              np.ascontiguousarray(items, dtype=np.int32), np.ascontiguousarray(answers, dtype=np.float32),
              np.ascontiguousarray(difficulty, dtype=np.float32)]
    result = np.zeros((len(answers), len(TRACE_NAMES)), dtype=np.float32)
    status = state_library()(len(offsets)-1, *arrays, meta['num_skills_with_padding'], meta['num_problems_with_padding'], result)
    if status: raise ValueError('out-of-range skill or item')
    return result

def prepare(split):
    if split not in ('train', 'val', 'test'): raise ValueError(split)
    dest = OUT/'cache'/split
    if (dest/'receipt.json').exists(): return
    dest.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA/'metadata.json').read_text()); idx = positions(split)
    offsets = np.load(DATA/f'{split}_offsets.npy'); skills = np.load(DATA/f'{split}_skills.npy')
    items = np.load(DATA/f'{split}_problem_index.npy'); answer = np.load(DATA/f'{split}_correct.npy')
    difficulty = np.load(DATA/f'{split}_problem_difficulty.npy')
    trace = trace_features(offsets, skills, items, answer, difficulty, meta)[idx]
    np.save(dest/'trace.npy', trace); del trace
    np.save(dest/'prefix.npy', np.load(DATA/f'{split}_prefix_features.npy', mmap_mode='r')[idx])
    np.save(dest/'known.npy', skills[idx] > 0); np.save(dest/'skills.npy', skills[idx])
    np.save(dest/'items.npy', items[idx]); np.save(dest/'target.npy', answer[idx])
    np.save(dest/'users.npy', np.load(DATA/f'{split}_user_ids.npy', mmap_mode='r')[idx])
    dev = legacy_module()
    behavior, history, group, truth, target, users = dev.feature_data(split)
    np.testing.assert_array_equal(target, answer[idx])
    np.save(dest/'behavior.npy', behavior); np.save(dest/'history.npy', history)
    np.save(dest/'truth.npy', truth)
    if (LEGACY/f'{split}_affect_state.npy').exists():
        state = np.load(LEGACY/f'{split}_affect_state.npy')
    else:
        with np.load(LEGACY/'state_selection.npz') as pars: choice = pars['choice']; kinds = pars['kinds']
        outputs = {}
        for kind in set(kinds[choice]):
            with np.load(LEGACY/f'{kind}_model.npz') as pars:
                outputs[kind] = dev.predict(behavior, history, group, kind, pars['weights'])
        state = np.column_stack([outputs[kinds[choice[j]]][:,j] for j in range(4)])
    np.save(dest/'state.npy', state)
    write(dest/'receipt.json', {'split': split, 'targets': len(idx), 'students': len(offsets)-1,
          'trace_names': TRACE_NAMES, 'detector_names': STATES, 'source_sha256': sha(SRC/'state_features.cpp'),
          'timing': 'New traces at target t use outcomes strictly before t; target skill, item and training-derived difficulty are available before response.',
          'difficulty': 'Uses the existing frozen training-only difficulty construction; not independently cross-fitted within training.'})

def data(split, take=None):
    folder = OUT/'cache'/split
    result = {name: np.load(folder/f'{name}.npy', mmap_mode='r') for name in (
        'trace', 'prefix', 'known', 'skills', 'items', 'target', 'users', 'behavior', 'history', 'state')}
    if take is not None: result = {k: v[take] for k,v in result.items()}
    return result

def cognitive(seed, split):
    directory = ROOT/'reports/mica_opt_20260917_v1' if seed == 42 else ROOT/f'reports/mica_meeting_20260917_v1/seed{seed}'
    path = directory/f'{split}_cognitive_logit.npy'
    if not path.exists() and split == 'test':
        path = ROOT/f'reports/mica_meeting_20260917_v1/seed{seed}/test_cognitive_logit.npy'
    return np.load(path, mmap_mode='r')

def specification(train):
    return {'state_knots': np.quantile(train['state'], [.25,.5,.75], axis=0).T,
            'trace_center': np.mean(train['trace'], axis=0, dtype=np.float64),
            'trace_scale': np.maximum(np.std(train['trace'], axis=0, dtype=np.float64), .05),
            'prefix_center': np.mean(train['prefix'], axis=0, dtype=np.float64),
            'prefix_scale': np.maximum(np.std(train['prefix'], axis=0, dtype=np.float64), .05)}

def design(d, spec, kind='trace_joint'):
    parts = []; names = []; groups = []
    def add(x, labels, group):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1: x = x[:,None]
        parts.append(x); names.extend(labels)
        groups.extend([group]*x.shape[1] if isinstance(group, str) else group)
    b = np.asarray(d['behavior'], np.float32)
    for knot in (None, -1., 0., 1.):
        add(b if knot is None else np.maximum(b-knot, 0),
            [f'behavior_{j}_hinge_{knot}' for j in range(b.shape[1])], 'behavior')
    known = np.asarray(d['known'], np.float32)
    add(np.column_stack((known, known*b[:,0], known*b[:,4], known*b[:,8])),
        ['known_skill', 'known_x_response_time', 'known_x_behavior_prior', 'known_x_last_answer'], 'skill_context')
    if kind.endswith('joint'):
        state = np.asarray(d['state'], np.float32)
        for j in (-1, 0, 1, 2):
            add(state if j == -1 else np.maximum(state-spec['state_knots'][:,j], 0),
                [f'affect_{s}_hinge_{j}' for s in STATES], ['affect_'+s for s in STATES])
        add(state*known[:,None], [f'known_x_{s}' for s in STATES], ['affect_'+s for s in STATES])
    p = np.asarray(d['prefix'], np.float32)
    add(p, PREFIX_NAMES, ['history_'+s.split('_')[0] for s in PREFIX_NAMES])
    if kind.startswith('trace'):
        t = np.clip((d['trace']-spec['trace_center'])/spec['trace_scale'], -10, 10).astype(np.float32)
        for knot in (None, -1., 0., 1.):
            add(t if knot is None else np.maximum(t-knot, 0),
                [f'{s}_hinge_{knot}' for s in TRACE_NAMES], ['cognition_'+s.split('_')[0] for s in TRACE_NAMES])
    return np.column_stack(parts).astype(np.float32), names, groups

def metrics(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return {'count': len(y), 'auc': float(roc_auc_score(y,p)),
            'log_loss': float(log_loss(y,p)), 'brier': float(np.mean((y-p)**2))}

def validation_metrics(d, p):
    dev = ((d['users']*1103515245+12345)%2147483647)%2 == 0
    return {name: metrics(d['target'][mask], p[mask]) for name,mask in (
        ('all', np.ones(len(p),bool)), ('development', dev), ('check', ~dev),
        ('known_skill', d['known']), ('unknown_skill', ~d['known']))}

def predict(d, offset, parameters):
    pred = np.empty(len(offset)); reconstruction = 0.
    for a in range(0,len(offset),50000):
        b = min(a+50000,len(offset)); sample = {k:v[a:b] for k,v in d.items()}
        x,names,groups = design(sample, parameters, str(parameters['kind']))
        assert names == list(parameters['names'])
        terms = ((x.astype(np.float64)-parameters['center'])/parameters['scale'])*parameters['theta'][1:]
        z = offset[a:b].astype(np.float64)+parameters['theta'][0]+terms.sum(1)
        summed = sum(terms[:,np.array(groups)==g].sum(1) for g in set(groups))
        reconstruction = max(reconstruction, float(np.max(np.abs(z-offset[a:b]-parameters['theta'][0]-summed))))
        pred[a:b] = expit(z)
    return pred, reconstruction
