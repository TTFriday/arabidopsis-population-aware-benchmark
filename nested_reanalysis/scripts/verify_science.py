import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NEW = Path(__file__).resolve().parents[1]
OUT = NEW / 'results/size_matched'
traits = pd.read_csv(NEW / 'inputs/data_processed/benchmark_traits.csv')
metrics = ['r', 'spearman', 'r2', 'bias', 'abs_bias', 'rmse']


def score(y, p):
    d = (p - y) / y.std()
    return np.array([np.corrcoef(y, p)[0, 1], spearmanr(y, p).statistic,
                     1 - np.square(y - p).sum() / np.square(y - y.mean()).sum(),
                     d.mean(), abs(d.mean()), np.sqrt(np.square(d).mean())])


folds = pd.read_csv(OUT / 'folds_raw.csv')
raw = pd.read_csv(OUT / 'traits_raw.csv')
per_trait = pd.read_csv(OUT / 'per_trait.csv').set_index('trait_id')
assert len(raw) == 135 and len(folds) == 425
assert raw[metrics].notna().all().all()
checked = 0
for trait in traits.trait_id:
    old = np.load(NEW / f'results/nested/{trait}__lopo__r0.npz')
    repetitions = []
    for repeat in range(5):
        d = np.load(OUT / f'{trait}__size_matched__r{repeat}.npz')
        for key in ('ids', 'y', 'clusters'):
            np.testing.assert_array_equal(d[key], old[key])
        seed = zlib.crc32(f'20261125|size-matched|{trait}|{repeat}'.encode())
        perm = np.random.default_rng(seed).permutation(len(d['ids']))
        reconstructed = np.full(len(perm), -1)
        offset = 0
        for fold, size in enumerate(np.bincount(old['fold'])):
            test = np.sort(perm[offset:offset + size])
            train = np.setdiff1d(np.arange(len(perm)), test)
            reconstructed[test] = fold
            row = folds[(folds.trait_id == trait) & (folds.repeat == repeat) & (folds.fold == fold)].iloc[0]
            assert (row.n_train, row.n_test) == (len(train), len(test))
            assert row.train_ids_sha256 == hashlib.sha256('|'.join(d['ids'][train]).encode()).hexdigest()
            np.testing.assert_allclose(score(d['y'][test], d['G'][test])[[0, 1, 2]], row[['r', 'spearman', 'r2']].to_numpy(dtype=float), rtol=0, atol=1e-12)
            offset += size
            checked += 1
        np.testing.assert_array_equal(reconstructed, d['fold'])
        np.testing.assert_allclose(score(d['y'], d['G']), raw[(raw.trait_id == trait) & (raw.repeat == repeat)].iloc[0][metrics].to_numpy(dtype=float), rtol=0, atol=1e-12)
        repetitions.append(score(d['y'], d['G']))
    np.testing.assert_allclose(np.mean(repetitions, axis=0), per_trait.loc[trait, metrics].to_numpy(dtype=float), rtol=0, atol=1e-12)
summary = pd.read_csv(OUT / 'paired_effects.csv')
for _, row in summary.iterrows():
    frame = per_trait.reset_index()
    if row.comparison.startswith('grouped-inner-20'):
        frame = frame[frame.all_lopo_inner_grouped]
    effect = frame[row.metric] - frame[f'lopo_{row.metric}']
    blocks = [effect[frame.study_id == s].to_numpy() for s in sorted(frame.study_id.unique())]
    draw = np.random.default_rng(20261121).integers(0, len(blocks), (5000, len(blocks)))
    samples = [np.median(np.concatenate([blocks[i] for i in ids])) for ids in draw]
    balanced = [np.median([np.median(blocks[i]) for i in ids]) for ids in draw]
    np.testing.assert_allclose([np.median(effect), *np.quantile(samples, [0.025, 0.975]), np.median([np.median(b) for b in blocks]), *np.quantile(balanced, [0.025, 0.975])], row[['median_effect', 'block_ci_low', 'block_ci_high', 'study_balanced_median', 'balanced_ci_low', 'balanced_ci_high']].to_numpy(dtype=float), rtol=0, atol=1e-12)
assert folds.inner_type.eq('grouped').all()
for name, digest in json.loads((NEW / 'inputs/sha256.json').read_text(encoding='utf8')).items():
    assert hashlib.sha256((NEW / name).read_bytes()).hexdigest() == digest
report = {'prediction_bundles_checked': 135, 'fold_assignments_and_train_hashes_checked': checked,
          'finite_repeat_metrics': True, 'paired_and_study_block_summaries_independently_verified': 6,
          'new_random_arm_grouped_inner_folds': 425, 'input_hashes_verified': 8,
          'absolute_tolerance': 1e-12, 'relative_tolerance': 0}
(OUT / 'independent_verification.json').write_text(json.dumps(report, indent=2), encoding='utf8')
print(json.dumps(report, indent=2))
