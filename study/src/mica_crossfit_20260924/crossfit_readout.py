"""Validation-only student-fold cross-fitted difficulty sensitivity audit.

This branch does not alter or overwrite the frozen MICA model. It recomputes
training item/skill difficulty using other training student folds, propagates
that difficulty through causal trace features, and fits one matched no-affect
transparent readout. The validation split still uses training-only statistics.
No test arrays are opened.
"""
import gc
import json
import time
from pathlib import Path

import numpy as np

from common import ROOT, FROZEN, DEST, DATA, data, positions, cognitive, write, sha, validation_metrics
from core import trace_features
from gam_search import ExplainableBoostingClassifier, export_tables, predict_tables
from reliability_confirm import response_matrix
from uncertainty_search import library, calculate

FOLDS = 10
PRIOR = 20.0


def student_folds(users):
    return (((np.asarray(users, np.int64) * 1103515245 + 12345) % 2147483647) % FOLDS).astype(np.int8)


def crossfit_difficulty(split='train'):
    users = np.load(DATA / f'{split}_user_ids.npy', mmap_mode='r')
    skills = np.load(DATA / f'{split}_skills.npy', mmap_mode='r').astype(np.int32, copy=False)
    items = np.load(DATA / f'{split}_problem_index.npy', mmap_mode='r').astype(np.int32, copy=False)
    answers = np.load(DATA / f'{split}_correct.npy', mmap_mode='r').astype(np.float64, copy=False)
    meta = json.loads((DATA / 'metadata.json').read_text())
    if split != 'train':
        return np.asarray(np.load(DATA / f'{split}_problem_difficulty.npy', mmap_mode='r'))
    folds = student_folds(users)
    n_items = int(meta['num_problems_with_padding'])
    n_skills = int(meta['num_skills_with_padding'])
    out = np.empty(len(answers), np.float32)
    diagnostics = []
    for f in range(FOLDS):
        use = folds != f
        global_p = float(answers[use].mean())
        item_n = np.bincount(items[use], minlength=n_items).astype(np.float64)
        item_y = np.bincount(items[use], weights=answers[use], minlength=n_items)
        skill_n = np.bincount(skills[use], minlength=n_skills).astype(np.float64)
        skill_y = np.bincount(skills[use], weights=answers[use], minlength=n_skills)
        item_p = np.clip((item_y + PRIOR * global_p) / (item_n + PRIOR), 1e-4, 1 - 1e-4)
        skill_p = np.clip((skill_y + PRIOR * global_p) / (skill_n + PRIOR), 1e-4, 1 - 1e-4)
        global_d = float(np.log((1 - global_p) / global_p))
        item_d = np.log((1 - item_p) / item_p)
        skill_d = np.log((1 - skill_p) / skill_p)
        m = folds == f
        vals = np.full(m.sum(), global_d, np.float32)
        known_item = items[m] > 0
        vals[known_item] = item_d[items[m][known_item]]
        fallback_skill = (~known_item) & (skills[m] > 0)
        vals[fallback_skill] = skill_d[skills[m][fallback_skill]]
        out[m] = vals
        diagnostics.append({'fold': f, 'students': int(np.unique(users[m]).size),
                            'events': int(m.sum()), 'global_accuracy_excluding_fold': global_p})
    write(DEST / 'difficulty_diagnostics.json', {
        'folds': diagnostics, 'prior_strength': PRIOR,
        'source': 'training full sequences; each row excludes its own student fold',
        'test_access': False})
    return out


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = DEST / 'protocol.json'
    if not protocol.exists():
        write(protocol, {
            'created_unix': time.time(), 'dataset': 'ASSISTments 2012 only',
            'purpose': 'Audit training-statistic reuse with student-fold cross-fitted item/skill difficulty.',
            'reference': 'reports/mica_evidence_2012_20260924/final_manifest.json',
            'folds': FOLDS, 'prior_strength': PRIOR,
            'train_statistics': 'Each training row uses difficulty computed from the other 9 training student folds; validation uses all training students only.',
            'downstream': 'Causal trace features recomputed; no-affect transparent EBM readout, exact frozen 3M-row sample and budget.',
            'selection': 'Validation all/development/check compared with frozen no-affect and joint controls; no test access or selection.',
            'source_sha256': sha(__file__)})
    result_path = DEST / 'results.json'
    if result_path.exists():
        print(result_path.read_text())
        return
    started = time.time()
    meta = json.loads((DATA / 'metadata.json').read_text())
    idx = positions('train')
    raw_diff = crossfit_difficulty('train')
    raw_sk = np.load(DATA / 'train_skills.npy', mmap_mode='r')
    raw_it = np.load(DATA / 'train_problem_index.npy', mmap_mode='r')
    raw_y = np.load(DATA / 'train_correct.npy', mmap_mode='r')
    offsets = np.load(DATA / 'train_offsets.npy', mmap_mode='r')
    trace = trace_features(offsets, raw_sk, raw_it, raw_y, raw_diff, meta)[idx]
    trace_path = DEST / 'train_trace_crossfit.npy'
    np.save(trace_path, trace)
    del trace, raw_diff
    gc.collect()

    # This first audit isolates the largest difficulty channel. The belief
    # filter stays on its frozen training-only difficulty stream; we record
    # that boundary instead of calling this a full-pipeline cross-fit.
    fn = library()
    belief = calculate(fn, offsets, raw_sk, raw_it, raw_y,
                       np.load(DATA / 'train_problem_difficulty.npy', mmap_mode='r'),
                       meta['num_skills_with_padding'], meta['num_problems_with_padding'])[idx]
    belief_path = DEST / 'train_belief_reference.npy'
    np.save(belief_path, belief)
    del belief
    gc.collect()

    sample = np.load(FROZEN / 'contextual_capacity_validation/auto16_3000k_sample.npy')
    tr, va = data('train', sample), data('val')
    tr['trace'] = np.load(trace_path, mmap_mode='r')[sample]
    bt = np.load(belief_path, mmap_mode='r')[sample]
    bv = np.load(FROZEN / 'uncertainty_validation/val_belief.npy', mmap_mode='r')
    x, names = response_matrix(tr, bt, np.empty((len(tr['target']), 4)), False)
    v, vnames = response_matrix(va, bv, np.empty((len(va['target']), 4)), False)
    assert names == vnames
    ot = np.asarray(cognitive(42, 'train')[sample])
    ov = np.asarray(cognitive(42, 'val'))
    pred = np.empty(len(v))
    rows = []
    for known in (False, True):
        tag = 'known' if known else 'missing'
        tm, vm = tr['known'] == known, va['known'] == known
        path = DEST / f'{tag}_tables.npz'
        folds = student_folds(tr['users'][tm])
        bags = np.column_stack([np.where(folds == j, -1, 1) for j in (0, 1)]).astype(np.int8)
        print('FIT', tag, flush=True)
        model = ExplainableBoostingClassifier(
            feature_names=names, feature_types=['continuous'] * len(names), interactions=64,
            max_bins=256, max_interaction_bins=32, outer_bags=2, inner_bags=0,
            learning_rate=.03, max_rounds=2000, early_stopping_rounds=80,
            min_samples_leaf=100, reg_lambda=10., max_leaves=3, n_jobs=2,
            random_state=42, smoothing_rounds=50, interaction_smoothing_rounds=50)
        model.fit(x[tm], tr['target'][tm], bags=bags, init_score=ot[tm])
        export_tables(model, path)
        table = dict(np.load(path))
        pred[vm] = predict_tables(v[vm], ov[vm], table)
        rows.append({'known': known, 'tables': str(path.relative_to(ROOT)),
                     'sha256': sha(path), 'terms': len(model.term_features_),
                     'best_iteration': np.asarray(model.best_iteration_).tolist()})
        del model
        gc.collect()
    np.save(DEST / 'validation.npy', pred)
    frozen = np.load(FROZEN / 'extended_confirmation/seed42_no_affect_validation.npy', mmap_mode='r')
    joint = np.load(FROZEN / 'extended_confirmation/seed42_joint_validation.npy', mmap_mode='r')
    result = {
        'metrics': validation_metrics(va, pred),
        'frozen_no_affect_metrics': validation_metrics(va, frozen),
        'frozen_joint_metrics': validation_metrics(va, joint),
        'regimes': rows,
        'crossfit_scope': 'Trace difficulty is student-fold cross-fitted; uncertainty belief remains the frozen difficulty stream in this sensitivity audit, so this is not a full-pipeline cross-fit.',
        'promotion_guard': {'reference': 'frozen_no_affect', 'minimum_auc_gain': .0002,
                           'both_validation_halves_non_decrease': True, 'test_used_for_selection': False},
        'test_access': False, 'source_sha256': sha(__file__),
        'seconds': time.time() - started}
    result['auc_deltas_vs_frozen_no_affect'] = {
        k: result['metrics'][k]['auc'] - result['frozen_no_affect_metrics'][k]['auc']
        for k in ('all', 'development', 'check')}
    result['eligible'] = bool(
        result['auc_deltas_vs_frozen_no_affect']['all'] >= .0002
        and result['auc_deltas_vs_frozen_no_affect']['development'] >= 0
        and result['auc_deltas_vs_frozen_no_affect']['check'] >= 0)
    write(result_path, result)
    write(DEST / 'status.json', {'stage': 'complete'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
