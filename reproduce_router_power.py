#!/usr/bin/env python3
"""Reconstruct the IETE carrier-grade router power prediction experiments.

Source: Lentisco et al. (2025), DOI: 10.5281/zenodo.17282065.
This is a documented RECONSTRUCTION, not the authors' unavailable original script.

Run: python reproduce_router_power.py --input-dir zenodo_data --output-dir results
Dependencies: numpy pandas scipy scikit-learn matplotlib requests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
RNG_SEED = 42
N_BOOTSTRAP = 500
BLOCK = 60
COVERAGE = 0.90
N_FOLDS = 5
FEATURE_NAMES = ['utilization', 'packet_size', 'log_packet_rate', 'load_rate_interaction']
ZENODO_ID = '17282065'
FILE_SPECS = [
    ('RA', 'Multiple traffic flows', 'TC3.2_multiple_traffic_flows_modelA_iteration1.csv',
     'dec360a0d74fa56913df391255e0c7b2', 6583, 3286),
    ('RA', 'Load and packet-length variation',
     'TC3.2_traffic_load_packet_length_variation_modelA_iteration1.csv',
     'b5a2ef4bf36229bc81c8aa9c571a4da1', 5268, 2635),
    ('RB', 'Multiple traffic flows', 'TC3.2_multiple_traffic_flows_modelB_iteration1.csv',
     '5695447b47cc5382d15b7966fe5fad93', 6585, 3283),
    ('RB', 'Load and packet-length variation',
     'TC3.2_traffic_load_packet_length_variation_modelB_iteration1.csv',
     'c3b518b935acf6f26cbfc48406a7f813', 5269, 2635),
]
EXPECTED_PRIMARY = {
    'RA': {'selected': 'Polynomial Ridge', 'cv_rmse': 6.552,
           'test_mae': 5.043, 'test_rmse': 7.151, 'test_r2': 0.645,
           'test_mape': 0.698, 'ci_mae_low': 4.606, 'ci_mae_high': 5.441,
           'coverage': 88.55, 'width': 19.58},
    'RB': {'selected': 'Linear Regression', 'cv_rmse': 1.881,
           'test_mae': 1.383, 'test_rmse': 1.876, 'test_r2': 0.823,
           'test_mape': 1.812, 'ci_mae_low': 1.290, 'ci_mae_high': 1.468,
           'coverage': 90.52, 'width': 6.34},
}


def digest_md5(filename: Path) -> str:
    d = hashlib.md5()
    with filename.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            d.update(block)
    return d.hexdigest()


def get_file(input_dir: Path, filename: str, expected_md5: str, allow_download: bool) -> Path:
    path = input_dir / filename
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f'Missing {path}; remove --no-download to fetch from Zenodo.')
        input_dir.mkdir(parents=True, exist_ok=True)
        urls = [
            f'https://zenodo.org/records/{ZENODO_ID}/files/{filename}?download=1',
            f'https://zenodo.org/api/records/{ZENODO_ID}/files/{filename}/content',
        ]
        errors = []
        for url in urls:
            try:
                with requests.get(url, stream=True, timeout=90,
                                  headers={'User-Agent': 'IETE-Reproduction/1.0'}) as r:
                    r.raise_for_status()
                    with path.open('wb') as fh:
                        for chunk in r.iter_content(1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                if digest_md5(path) == expected_md5:
                    print(f'Downloaded and MD5-checked {filename}')
                    break
                path.unlink(missing_ok=True)
                errors.append(f'checksum failure at {url}')
            except requests.RequestException as err:
                path.unlink(missing_ok=True)
                errors.append(str(err))
        else:
            raise RuntimeError(f'Could not download {filename}: {errors}')
    md5 = digest_md5(path)
    if md5 != expected_md5:
        raise RuntimeError(f'{path} has MD5 {md5}, expected {expected_md5}. '
                           'Use original Zenodo v1 iteration-1 file.')
    return path


def plateau_midpoint(load: np.ndarray) -> tuple[int, tuple[int, int]]:
    """Find center of the contiguous maximum-traffic plateau nearest trace middle."""
    finite = np.isfinite(load)
    if not finite.any():
        raise ValueError('Throughput contains no finite observations.')
    maximum = np.nanmax(load)
    mask = np.isclose(load, maximum, atol=1e-8) & finite
    switches = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(switches == 1)[0]
    ends = np.where(switches == -1)[0]  # exclusive
    # Near trace center first, longest among similar candidates.
    mids = (starts + ends) / 2
    pick = min(range(len(starts)), key=lambda i: (abs(mids[i] - len(load)/2),
                                                   -(ends[i] - starts[i])))
    start, end = int(starts[pick]), int(ends[pick])
    return (start + end) // 2, (start, end)


def prepare_file(router: str, experiment: str, file: Path,
                 expected_n: int, expected_train_n: int,
                 match_published_split: bool) -> tuple[pd.DataFrame, dict]:
    # The Zenodo files are SEMICOLON-delimited. 'Times_s' (plural) is the real header.
    data = pd.read_csv(file, sep=';', encoding='utf-8-sig')
    data.columns = [c.strip() for c in data.columns]
    required = ['Throughput_Percentage', 'PacketSize_B',
                'Packet_Rate', 'Power_Consumption']
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'{file.name}: missing {sorted(missing)}; fields: {list(data.columns)}')
    time_col = 'Times_s' if 'Times_s' in data else 'Time_s'
    if time_col not in data:
        raise ValueError(f'{file.name}: elapsed-time field not found')
    for c in required + [time_col]:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data['_original_row'] = np.arange(len(data))
    # Keep original trace order; sorting on Time_s also guards against re-ordered CSV rows.
    data = data.sort_values([time_col, '_original_row'], kind='stable').reset_index(drop=True)
    raw = len(data)
    # Manuscript's Table 1 reports packet-length experiment range 160-1500 B.
    # This is a reconstruction inference: exclude the original 62-B condition.
    drop_62 = (data['PacketSize_B'] < 160) if 'packet-length' in experiment else pd.Series(False, index=data.index)
    n_drop_packet = int(drop_62.sum())
    data = data.loc[~drop_62].copy().reset_index(drop=True)
    validity = (data['Throughput_Percentage'].between(0, 100) &
                (data['PacketSize_B'] > 0) & (data['Packet_Rate'] >= 0) &
                (data['Power_Consumption'] > 0) &
                np.isfinite(data[required]).all(axis=1))
    dropped_invalid = int((~validity).sum())
    data = data.loc[validity].copy().reset_index(drop=True)
    midpoint, (first_max, last_max) = plateau_midpoint(
        data['Throughput_Percentage'].to_numpy(dtype=float))
    split = midpoint
    used_published_split = False
    if match_published_split and first_max <= expected_train_n <= last_max:
        # Published partition sizes disambiguate the approximate plateau midpoint,
        # using NO target values. Refuse to force a non-plateau split.
        split = expected_train_n
        used_published_split = True
    data['router'] = router
    data['experiment'] = experiment
    data['source_file'] = file.name
    data['cycle'] = np.where(np.arange(len(data)) < split, 'train', 'test')
    data['utilization'] = data['Throughput_Percentage'] / 100.0
    data['packet_size'] = data['PacketSize_B']
    data['log_packet_rate'] = np.log1p(data['Packet_Rate'])
    data['load_rate_interaction'] = data['utilization'] * data['log_packet_rate']
    # Exact operating condition groups, not individual successive time samples.
    data['group'] = (data['experiment'] + '|' +
                     data['Throughput_Percentage'].round(6).astype(str) + '|' +
                     data['packet_size'].round(6).astype(str))
    audit = dict(router=router, experiment=experiment, source_file=file.name,
                 raw_rows=raw, excluded_packet_under_160=n_drop_packet,
                 excluded_invalid=dropped_invalid, retained=len(data),
                 manuscript_retained=expected_n, midpoint_index=midpoint,
                 max_plateau_begin=first_max, max_plateau_end_exclusive=last_max,
                 used_published_split=used_published_split,
                 train_rows=split, manuscript_train=expected_train_n,
                 test_rows=len(data)-split)
    if len(data) != expected_n:
        warnings.warn(f'{file.name}: retained {len(data)}, manuscript says {expected_n}')
    if split != expected_train_n:
        warnings.warn(f'{file.name}: split {split}, manuscript says {expected_train_n}. '
                      'See sample_count_audit.csv; do NOT tune split by test errors.')
    return data, audit


def models() -> dict:
    return {
        'Mean Baseline': DummyRegressor(strategy='mean'),
        'Linear Regression': LinearRegression(),
        'Polynomial Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('ridge', Ridge(alpha=1e-6)),
        ]),
        'Random Forest': RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=1.0,
            random_state=SEED, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=1.0,
            random_state=SEED, n_jobs=-1),
        'Histogram Gradient Boosting': HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=0.1, random_state=SEED),
    }


def metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = predicted - actual
    return {
        'MAE_W': float(mean_absolute_error(actual, predicted)),
        'RMSE_W': float(np.sqrt(mean_squared_error(actual, predicted))),
        'R2': float(r2_score(actual, predicted)),
        'MAPE_percent': float(100 * np.mean(np.abs(error) / actual)),
        'mean_bias_pred_minus_measured_W': float(np.mean(error)),
        'P95_absolute_error_W': float(np.quantile(np.abs(error), 0.95)),
    }


def grouped_oof(estimator, train, features=FEATURE_NAMES):
    group_cv = GroupKFold(n_splits=N_FOLDS)
    x = train[features].to_numpy(dtype=float)
    y = train['Power_Consumption'].to_numpy(dtype=float)
    gp = train['group'].to_numpy()
    if len(np.unique(gp)) < N_FOLDS:
        raise ValueError('Not enough operating-state groups for 5-fold CV.')
    oof = np.full(len(train), np.nan)
    fold_info = []
    for k, (indices_fit, indices_valid) in enumerate(group_cv.split(x, y, gp), 1):
        m = clone(estimator)
        m.fit(x[indices_fit], y[indices_fit])
        oof[indices_valid] = m.predict(x[indices_valid])
        fold_info.append({'fold': k, 'validation_rows': int(len(indices_valid)),
                          'validation_groups': int(len(set(gp[indices_valid]))),
                          'RMSE_W': float(np.sqrt(mean_squared_error(y[indices_valid], oof[indices_valid])))})
    assert np.isfinite(oof).all()
    # Report GLOBAL pooled OOF RMSE, not the unweighted mean of fold RMSEs.
    return oof, metrics(y, oof), fold_info


def bootstrap_intervals(test: pd.DataFrame, predictions: np.ndarray, seed=SEED):
    """Moving block bootstrap, stratified by experimental file/condition."""
    rng = np.random.default_rng(seed)
    test = test.reset_index(drop=True)
    indices_by_file = [np.flatnonzero(test['source_file'].eq(f).to_numpy())
                       for f in test['source_file'].unique()]
    actual = test['Power_Consumption'].to_numpy()
    reps = []
    for _ in range(N_BOOTSTRAP):
        selected = []
        for ids in indices_by_file:
            length = len(ids)
            starts = rng.integers(0, max(1, length - BLOCK + 1),
                                  size=int(np.ceil(length / BLOCK)) + 2)
            sampled = np.concatenate([ids[start:start+BLOCK] for start in starts])[:length]
            if len(sampled) != length:
                raise AssertionError('Bootstrap sampling length was unexpected')
            selected.append(sampled)
        sample = np.concatenate(selected)
        reps.append(metrics(actual[sample], predictions[sample]))
    ci = {}
    for key in reps[0]:
        ci[key + '_lower95'] = float(np.quantile([x[key] for x in reps], 0.025))
        ci[key + '_upper95'] = float(np.quantile([x[key] for x in reps], 0.975))
    return ci


def conformal_from_oof(y: np.ndarray, pred_oof: np.ndarray, test_y, test_pred):
    """Grouped out-of-fold calibration as documented. Not strict split conformal."""
    residual = np.abs(y - pred_oof)
    n = len(residual)
    k = int(np.ceil((n + 1) * COVERAGE))
    k = min(k, n)
    q = float(np.partition(residual, k - 1)[k-1])
    lower = test_pred - q
    upper = test_pred + q
    return dict(half_width_W=q, mean_width_W=2*q,
                empirical_coverage_percent=float(100 * np.mean(
                    (test_y >= lower) & (test_y <= upper))),
                nominal_coverage_percent=100 * COVERAGE), lower, upper


def feature_ablation(chosen, train, test, router):
    sets = {
        'Load only': ['utilization'],
        'Packet size only': ['packet_size'],
        'Official two-feature set': ['utilization', 'packet_size'],
        'Enhanced without interaction': FEATURE_NAMES[:3],
        'Enhanced full set': FEATURE_NAMES,
    }
    rows = []
    for name, features in sets.items():
        est = clone(chosen).fit(train[features].to_numpy(), train['Power_Consumption'].to_numpy())
        p = est.predict(test[features].to_numpy())
        rows.append({'router': router, 'feature_set': name, 'features': ', '.join(features),
                     **metrics(test['Power_Consumption'], p)})
    return rows


def dataset_descriptives(data: pd.DataFrame) -> list:
    rows = []
    for (router, experiment), d in data.groupby(['router', 'experiment'], sort=False):
        idle = d[d['Throughput_Percentage'] == 0]['Power_Consumption']
        full = d[d['Throughput_Percentage'] == 100]['Power_Consumption']
        t = d['Times_s'] if 'Times_s' in d else d['Time_s']
        rows.append({'router': router, 'experiment': experiment,
                     'observations': len(d), 'duration_h': (t.max()-t.min())/3600,
                     'packet_size_mean_B': d['PacketSize_B'].mean(),
                     'packet_size_min_B': d['PacketSize_B'].min(),
                     'packet_size_max_B': d['PacketSize_B'].max(),
                     'power_mean_W': d['Power_Consumption'].mean(),
                     'power_SD_W': d['Power_Consumption'].std(ddof=1),
                     'power_min_W': d['Power_Consumption'].min(),
                     'power_max_W': d['Power_Consumption'].max(),
                     'idle_power_W': idle.mean(), 'full_load_power_W': full.mean(),
                     'dynamic_increase_W': full.mean() - idle.mean()})
    return rows


def save_fig(fig, directory: Path, name: str):
    fig.savefig(directory / (name + '.png'), dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(directory / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_plots(all_data: pd.DataFrame, cv: pd.DataFrame, perf: pd.DataFrame,
               ablation: pd.DataFrame, predictions: dict, out: Path):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    # Fig 1: exact conceptual order from the manuscript.
    fig, ax = plt.subplots(figsize=(12.5, 2.5))
    ax.set_xlim(0, 5); ax.set_ylim(0, 1); ax.axis('off')
    labels = ['Router telemetry\nRA / RB', 'Preprocessing\nand feature design',
              'Grouped CV\nand model selection', 'Held-out downward\nramp evaluation',
              'Uncertainty\nand ablation']
    for i, label in enumerate(labels):
        ax.text(i + 0.43, .55, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', lw=1))
        if i < 4:
            ax.annotate('', (i+.97,.55), (i+.81,.55),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.3))
    save_fig(fig, out, 'Fig_01_Methodology')

    # Figs 2-3: means within the experimental load bands.
    for router, figure_number in [('RA',2), ('RB',3)]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for exp, d in all_data[all_data.router.eq(router)].groupby('experiment'):
            avg = d.groupby('Throughput_Percentage')['Power_Consumption'].mean()
            ax.plot(avg.index, avg.values, marker='o', label=exp, lw=1.8, ms=4)
        ax.set(xlabel='Traffic load (%)', ylabel='Measured mean power (W)',
               title=f'{router}: measured power response to traffic load')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.2)
        save_fig(fig, out, f'Fig_{figure_number:02d}_{router}_Load_Power')

    # Fig 4: grouped CV (exclude dummy from central comparison, include as separately scaled).
    fig, ax = plt.subplots(figsize=(10, 4))
    names = ['Polynomial Ridge', 'Extra Trees', 'Histogram Gradient Boosting',
             'Linear Regression', 'Random Forest', 'Mean Baseline']
    pivot = cv.pivot(index='model', columns='router', values='RMSE_W').reindex(names)
    x = np.arange(len(names)); w=.36
    for shift, router in [(-w/2, 'RA'), (w/2, 'RB')]:
        ax.bar(x+shift, pivot[router], width=w, label=router)
    ax.set_xticks(x, names, rotation=24, ha='right')
    ax.set_ylabel('Grouped CV RMSE (W)')
    ax.legend(frameon=False)
    ax.set_title('Candidate models: grouped cross-validation')
    save_fig(fig, out, 'Fig_04_CV_Comparison')

    for router, number in [('RA',5), ('RB',6)]:
        d = predictions[router]['test']
        fig, ax = plt.subplots(figsize=(6,5))
        lo = min(d['measured_W'].min(), d['predicted_W'].min())
        hi = max(d['measured_W'].max(), d['predicted_W'].max())
        hb = ax.hexbin(d['measured_W'], d['predicted_W'], gridsize=45,
                       bins='log', mincnt=1, cmap='viridis')
        ax.plot([lo,hi],[lo,hi],ls='--', lw=1, color='gray')
        fig.colorbar(hb, ax=ax, label='log observation density')
        ax.set(xlabel='Measured power (W)', ylabel='Predicted power (W)',
               title=f'{router}: held-out ramp measured vs predicted')
        save_fig(fig, out, f'Fig_{number:02d}_{router}_Measured_vs_Predicted')

    for router, number in [('RA',7), ('RB',8)]:
        d = predictions[router]['test']
        d = d[d['experiment'] == 'Multiple traffic flows'].copy()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.plot(d['time_s'], d['measured_W'], lw=.65, alpha=.58, label='Measured')
        ax.plot(d['time_s'], d['predicted_W'], lw=1.3, label='Predicted')
        ax.set(xlabel='Time since experiment start (s)', ylabel='Power (W)',
               title=f'{router}: independent multiple-flow downward-ramp trace')
        ax.legend(frameon=False)
        save_fig(fig, out, f'Fig_{number:02d}_{router}_Power_Trace')

    fig, ax = plt.subplots(figsize=(10, 4))
    items = ['Packet size only', 'Load only', 'Official two-feature set',
             'Enhanced without interaction', 'Enhanced full set']
    p = ablation.pivot(index='feature_set', columns='router', values='MAE_W').reindex(items)
    x = np.arange(len(items)); w=.36
    ax.bar(x-w/2, p['RA'], width=w, label='RA')
    ax.bar(x+w/2, p['RB'], width=w, label='RB')
    ax.set_xticks(x, items, rotation=19, ha='right')
    ax.set_ylabel('Independent-ramp MAE (W)')
    ax.set_title('Post hoc feature-set ablation (descriptive only)')
    ax.legend(frameon=False)
    save_fig(fig, out, 'Fig_09_Feature_Ablation')

    rb = predictions['RB']['test']
    rb = rb[rb['experiment'] == 'Multiple traffic flows'].copy()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(rb['time_s'].to_numpy(), rb['conformal_lower_W'].to_numpy(),
                    rb['conformal_upper_W'].to_numpy(), alpha=.22,
                    label='Nominal 90% calibrated interval')
    ax.plot(rb['time_s'], rb['measured_W'], lw=.65, alpha=.5, label='Measured')
    ax.plot(rb['time_s'], rb['predicted_W'], lw=1.1, label='Predicted')
    ax.set(xlabel='Time since experiment start (s)', ylabel='Power (W)',
           title='RB: grouped-OOF conformal interval, independent trace')
    ax.legend(frameon=False)
    save_fig(fig, out, 'Fig_10_RB_Conformal_Interval')


def comparison_report(cv, performance, uncertainty, selected, out):
    rows = []
    for router, target in EXPECTED_PRIMARY.items():
        c = cv[(cv.router == router) & (cv.model == target['selected'])]
        p = performance[(performance.router == router) & (performance.experiment == 'Overall')]
        u = uncertainty[uncertainty.router == router]
        if c.empty or p.empty or u.empty:
            continue
        actual = {'cv_rmse': c['RMSE_W'].iloc[0],
                  'test_mae': p['MAE_W'].iloc[0],
                  'test_rmse': p['RMSE_W'].iloc[0],
                  'test_r2': p['R2'].iloc[0],
                  'test_mape': p['MAPE_percent'].iloc[0],
                  'ci_mae_low': u['MAE_W_lower95'].iloc[0],
                  'ci_mae_high': u['MAE_W_upper95'].iloc[0],
                  'coverage': u['empirical_coverage_percent'].iloc[0],
                  'width': u['mean_width_W'].iloc[0]}
        for metric, expected in target.items():
            if metric == 'selected':
                continue
            value = float(actual[metric])
            rows.append({'router': router, 'selected_model': selected[router],
                         'manuscript_model': target['selected'],
                         'same_selected_model': selected[router] == target['selected'],
                         'metric': metric,
                         'manuscript': expected, 'reconstruction': value,
                         'difference': value - expected})
    pd.DataFrame(rows).to_csv(out / 'manuscript_vs_reconstructed.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('zenodo_data'))
    parser.add_argument('--output-dir', type=Path, default=Path('router_power_results'))
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--strict-counts', action='store_true',
                        help='Fail if sample counts differ from the manuscript')
    parser.add_argument('--midpoint-only', action='store_true',
                        help='Do not use manuscript training counts to resolve plateau midpoint')
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, audit_rows = [], []
    for router, experiment, filename, md5, expected_n, expected_train in FILE_SPECS:
        path = get_file(args.input_dir, filename, md5, not args.no_download)
        df, audit = prepare_file(router, experiment, path, expected_n,
                                 expected_train, not args.midpoint_only)
        frames.append(df); audit_rows.append(audit)
    data = pd.concat(frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / 'sample_count_audit.csv', index=False)
    print('\nDATA AUDIT\n', audit[['router', 'experiment', 'raw_rows',
        'excluded_packet_under_160', 'excluded_invalid', 'retained',
        'train_rows', 'test_rows', 'used_published_split']].to_string(index=False))
    print('Total retained:', len(data))
    if args.strict_counts:
        assert len(data) == 23705, f'Expected 23705, got {len(data)}'
        assert audit['retained'].eq(audit['manuscript_retained']).all()
        assert audit['train_rows'].eq(audit['manuscript_train']).all()

    pd.DataFrame(dataset_descriptives(data)).to_csv(args.output_dir / 'Table_1_dataset.csv', index=False)
    cv_rows, fold_rows, perf_rows, ci_rows, ablation_rows = [], [], [], [], []
    predictions, selected = {}, {}
    for router in ('RA', 'RB'):
        rd = data[data.router.eq(router)]
        train = rd[rd.cycle.eq('train')].reset_index(drop=True)
        test = rd[rd.cycle.eq('test')].reset_index(drop=True)
        if len(train) < 5 or not len(test):
            raise ValueError(f'{router}: empty train/test split')
        candidate = models()
        oof_results = {}
        print(f'\n{router} CV: train={len(train)} test={len(test)}')
        for model_name, estimator in candidate.items():
            oof, score, folds = grouped_oof(estimator, train)
            cv_rows.append({'router': router, 'model': model_name, **score})
            fold_rows.extend({'router': router, 'model': model_name, **f} for f in folds)
            oof_results[model_name] = oof
            print(f'  {model_name:29s} RMSE={score["RMSE_W"]:.3f} R2={score["R2"]:.3f}')
        row = min((r for r in cv_rows if r['router'] == router), key=lambda r:r['RMSE_W'])
        best_name = row['model']
        selected[router] = best_name
        print('Selected on CV only:', best_name, '(manuscript:', EXPECTED_PRIMARY[router]['selected'] + ')')
        if best_name != EXPECTED_PRIMARY[router]['selected']:
            warnings.warn(f'{router}: selected model differs from manuscript. Differences in test metrics are NOT a like-for-like reproduction.')
        estimator = clone(candidate[best_name])
        estimator.fit(train[FEATURE_NAMES].to_numpy(), train['Power_Consumption'].to_numpy())
        pred = estimator.predict(test[FEATURE_NAMES].to_numpy())
        test_actual = test['Power_Consumption'].to_numpy()
        overall = metrics(test_actual, pred)
        perf_rows.append({'router': router, 'experiment': 'Overall',
                          'test_n': len(test), **overall})
        for experiment in ['Multiple traffic flows', 'Load and packet-length variation']:
            mask = test['experiment'].eq(experiment).to_numpy()
            perf_rows.append({'router': router, 'experiment': experiment,
                              'test_n': int(mask.sum()), **metrics(test_actual[mask], pred[mask])})
        print('  HELD-OUT: ', ', '.join(f'{k}={v:.4f}' for k,v in overall.items()))
        ci = bootstrap_intervals(test, pred, seed=RNG_SEED + (0 if router=='RA' else 1))
        interval, lower, upper = conformal_from_oof(
            train['Power_Consumption'].to_numpy(), oof_results[best_name],
            test_actual, pred)
        ci_rows.append({'router': router, **ci, **interval})
        ablation_rows.extend(feature_ablation(candidate[best_name], train, test, router))
        time_col = 'Times_s' if 'Times_s' in test else 'Time_s'
        pframe = pd.DataFrame({'router': router, 'experiment': test['experiment'],
                               'time_s': test[time_col], 'measured_W': test_actual,
                               'predicted_W': pred, 'residual_pred_minus_measured_W': pred-test_actual,
                               'conformal_lower_W': lower, 'conformal_upper_W': upper})
        pframe.to_csv(args.output_dir / f'{router}_heldout_predictions.csv', index=False)
        predictions[router] = {'test': pframe}
        pd.DataFrame({'router': router, 'experiment': train['experiment'],
                      'group': train['group'], 'measured_W': train['Power_Consumption'],
                      'out_of_fold_prediction_W': oof_results[best_name]}).to_csv(
                          args.output_dir / f'{router}_OOF_calibration.csv', index=False)
    cv = pd.DataFrame(cv_rows)
    perf = pd.DataFrame(perf_rows)
    unc = pd.DataFrame(ci_rows)
    abl = pd.DataFrame(ablation_rows)
    cv.to_csv(args.output_dir / 'Table_2_grouped_CV.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / 'CV_fold_details.csv', index=False)
    perf.to_csv(args.output_dir / 'Table_3_heldout_test.csv', index=False)
    unc.to_csv(args.output_dir / 'Table_4_bootstrap_and_conformal.csv', index=False)
    abl.to_csv(args.output_dir / 'Table_5_feature_ablation.csv', index=False)
    comparison_report(cv, perf, unc, selected, args.output_dir)
    if not args.skip_plots:
        make_plots(data, cv, perf, abl, predictions, args.output_dir)
    (args.output_dir/'run_metadata.json').write_text(json.dumps({
        'source': 'Zenodo 10.5281/zenodo.17282065',
        'library_versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                             'scikit_learn': __import__('sklearn').__version__,
                             'matplotlib': matplotlib.__version__},
        'reconstruction': True, 'not_original_author_source': True,
        'random_seed': SEED, 'bootstrap_repetitions': N_BOOTSTRAP,
        'bootstrap_block_observations': BLOCK, 'nominal_conformal_coverage': COVERAGE,
        'grouped_cv_folds': N_FOLDS, 'selected_models': selected,
        'packet_length_minimum_retained': 160,
        'published_split_disambiguation': not args.midpoint_only,
        'caveats': ['OOF calibration is not strict independent split conformal',
                    'Reported code/library versions and preprocessing script not available',
                    'Additional filtering and CV tie-breaking choices may affect exact metrics'],
    }, indent=2), encoding='utf-8')
    print('\nSaved full analysis to:', args.output_dir.resolve())
    print('READ the manuscript_vs_reconstructed.csv differences; do not substitute manuscript targets for calculations.')


if __name__ == '__main__':
    main()
