import argparse
import hashlib
import importlib.util
import json
import time
import zlib
from multiprocessing import Pool

import biosystems_lib as B
import numpy as np
import pandas as pd

OUT = B.NEW / 'results/size_matched'
REPEATS = 5
_P = None


def init():
    global _P
    _P = B.Panel()


def run(trait_id):
    ids, y, cl, X, R, study = _P.trait(trait_id)
    original = np.load(B.NEW / f'results/nested/{trait_id}__lopo__r0.npz')
    np.testing.assert_array_equal(ids, original['ids'])
    np.testing.assert_array_equal(cl, original['clusters'])
    np.testing.assert_allclose(y, original['y'], rtol=0, atol=0)
    sizes = np.bincount(original['fold'])
    assert sizes.sum() == len(ids) and np.all(sizes > 0)
    rows, folds = [], []
    for repeat in range(REPEATS):
        seed = zlib.crc32(f'20261125|size-matched|{trait_id}|{repeat}'.encode())
        perm = np.random.default_rng(seed).permutation(len(ids))
        membership = np.full(len(ids), -1)
        prediction = np.full(len(ids), np.nan)
        offset = 0
        for fold, size in enumerate(sizes):
            test = np.sort(perm[offset:offset + size])
            train = np.setdiff1d(np.arange(len(ids)), test)
            assert len(test) == np.sum(original['fold'] == fold)
            assert len(train) == np.sum(original['fold'] != fold)
            pred, params, details = B.nested_predict(X, R, y, cl, train, test, 'lopo', 100, expression=False)
            membership[test] = fold
            prediction[test] = pred['G']
            folds.append({'trait_id': trait_id, 'study_id': study, 'repeat': repeat,
                          'fold': fold, 'n_train': len(train), 'n_test': len(test), 'seed': seed,
                          'fst': B.fst(X, train, test), **params['G'], **details,
                          'train_ids_sha256': hashlib.sha256('|'.join(ids[train]).encode()).hexdigest(),
                          **B.metrics(y[test], pred['G'], y.std())})
            offset += size
        assert np.all(membership >= 0) and np.all(np.isfinite(prediction))
        np.savez_compressed(OUT / f'{trait_id}__size_matched__r{repeat}.npz',
                            ids=ids, y=y, clusters=cl, fold=membership, G=prediction)
        rows.append({'trait_id': trait_id, 'study_id': study, 'repeat': repeat,
                     'n': len(ids), **B.metrics(y, prediction, y.std())})
    pd.DataFrame(rows).to_csv(OUT / f'{trait_id}__traits.csv', index=False)
    pd.DataFrame(folds).to_csv(OUT / f'{trait_id}__folds.csv', index=False)
    return trait_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    traits = pd.read_csv(B.PROC / 'benchmark_traits.csv').trait_id.tolist()
    start = time.perf_counter()
    with Pool(args.workers, initializer=init) as pool:
        for count, trait in enumerate(pool.imap_unordered(run, traits), 1):
            print(f'Size-matched validation {count}/{len(traits)} {trait}: {time.perf_counter()-start:.1f}s', flush=True)
    raw = pd.concat([pd.read_csv(OUT / f'{t}__traits.csv') for t in traits], ignore_index=True)
    folds = pd.concat([pd.read_csv(OUT / f'{t}__folds.csv') for t in traits], ignore_index=True)
    raw.to_csv(OUT / 'traits_raw.csv', index=False)
    folds.to_csv(OUT / 'folds_raw.csv', index=False)
    averaged = raw.groupby(['trait_id', 'study_id'], as_index=False)[['r', 'spearman', 'r2', 'rmse', 'bias', 'abs_bias']].mean()
    lopo = pd.read_csv(B.NEW / 'results/nested/per_trait.csv')
    lopo = lopo[(lopo.cv == 'lopo') & (lopo.features == 'G')]
    compared = averaged.merge(lopo[['trait_id', 'r', 'r2', 'rmse']].rename(columns={m: f'lopo_{m}' for m in ('r', 'r2', 'rmse')}), on='trait_id', validate='one_to_one')
    existing_folds = pd.read_csv(B.NEW / 'results/nested/folds_raw.csv')
    existing_folds = existing_folds[(existing_folds.cv == 'lopo') & (existing_folds.features == 'G')]
    grouped = existing_folds.groupby('trait_id').inner_type.apply(lambda x: x.eq('grouped').all())
    compared['all_lopo_inner_grouped'] = compared.trait_id.map(grouped)
    compared.to_csv(OUT / 'per_trait.csv', index=False)
    spec = importlib.util.spec_from_file_location('synthesis', B.NEW / 'scripts/26_statistical_synthesis.py')
    synthesis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synthesis)
    summaries, studies, omissions, units = [], [], [], []
    for subset in ('all-27', 'grouped-inner-20'):
        data = compared if subset == 'all-27' else compared[compared.all_lopo_inner_grouped]
        for metric in ('r', 'r2', 'rmse'):
            frame = data[['trait_id', 'study_id']].copy()
            frame['effect'] = data[metric] - data[f'lopo_{metric}']
            label = f'{subset}: size-matched-random-minus-lopo'
            summary, study, omission = synthesis.summarize(frame, label, metric)
            summary.update({'matched_random_median': data[metric].median(), 'lopo_median': data[f'lopo_{metric}'].median()})
            summaries.append(summary)
            studies.append(study)
            omissions.append(omission)
            units.append(frame.assign(comparison=label, metric=metric))
    pd.DataFrame(summaries).to_csv(OUT / 'paired_effects.csv', index=False)
    pd.concat(studies).to_csv(OUT / 'study_effects.csv', index=False)
    pd.concat(omissions).to_csv(OUT / 'study_omission.csv', index=False)
    pd.concat(units).to_csv(OUT / 'unit_effects.csv', index=False)
    folds.groupby('inner_type', as_index=False).size().to_csv(OUT / 'tuning_counts.csv', index=False)
    inputs = json.loads((B.NEW / 'inputs/sha256.json').read_text(encoding='utf8'))
    (OUT / 'analysis_record.json').write_text(json.dumps({
        'outer_repeats': REPEATS, 'workers': args.workers, 'elapsed_seconds': time.perf_counter()-start,
        'outer_seed': 'CRC32(20261125|size-matched|trait_id|repeat)',
        'design': 'Random permutation partition with each outer fold exactly matching the corresponding LOPO fold train/test counts; pooled OOF metrics averaged across five repeats',
        'inner_tuning': 'Grouped where possible, one-group random fallback; same KRR grid, fitted preprocessing and inner seed 100 as LOPO',
        'scope': 'Retrospective sensitivity conditional on the characterized collection; does not isolate a causal population-structure effect or validate a future cohort',
        'effect_block_seed': synthesis.SEED, 'effect_block_resamples': synthesis.BOOTSTRAPS,
        'inputs': inputs,
        'code_sha256': {name: hashlib.sha256((B.NEW / 'scripts' / name).read_bytes()).hexdigest() for name in
                        ('biosystems_lib.py', '26_statistical_synthesis.py', '28_size_matched_validation.py')}
    }, indent=2), encoding='utf8')
    print(pd.DataFrame(summaries).query("metric == 'r'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
