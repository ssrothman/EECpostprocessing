import simonplot as splt
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import numpy as np
from correctionlib import CorrectionSet, schemav2
from plotting.load_datasets import build_pq_dataset, build_pq_dataset_stack
from scipy.interpolate import UnivariateSpline
from simonplot.plottables.Datasets import DatasetStack

RUNTAG = 'Mar_01_2026'
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_OUTPUT_DIR = REPO_ROOT / 'ANplots' / 'kinreweight_2D' / RUNTAG
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_1D_CORRECTION_BASE_DIR = REPO_ROOT / 'ANplots' / 'kinreweight' / RUNTAG
INPUT_1D_KEY = 'zpt_reweight'
MC_MODES = ['nominal', 'herwig_signal']

ZPT_BINS = np.asarray([
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    4.0,
])

ABSY_BINS = np.asarray([
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
])

S_FZY = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
S_ALPHA = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
MIN_WEIGHT = 0.05
MAX_WEIGHT = 5.0
ALPHA_MIN = 0.0
FZY_AIC_SMOOTH_TOLERANCE = 2.0
FZY_MIN_SMOOTHING = 16.0
S_JPT = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
JPT_BINS = np.asarray([30.0, 40.0, 50.0, 70.0, 90.0, 120.0, 150.0, 200.0, 250.0, 320.0, 400.0, 500.0, 700.0, 1000.0])


def get_output_dir(mc_mode):
    if mc_mode == 'nominal':
        output_dir = BASE_OUTPUT_DIR
    else:
        output_dir = BASE_OUTPUT_DIR / mc_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_input_1d_correction(mc_mode):
    if mc_mode == 'nominal':
        path = INPUT_1D_CORRECTION_BASE_DIR / 'zpt_reweight.json'
    else:
        path = INPUT_1D_CORRECTION_BASE_DIR / mc_mode / 'zpt_reweight.json'
    return path


def build_mc_stack_for_mode(mc_mode):
    if mc_mode == 'nominal':
        return build_pq_dataset_stack(
            configsuite='BasicConfig',
            runtag=RUNTAG,
            dataset='allMC',
            objsyst='nominal',
            table='events',
            location='scratch-submit',
        )

    if mc_mode == 'herwig_signal':
        bkg = build_pq_dataset_stack(
            configsuite='BasicConfig',
            runtag=RUNTAG,
            dataset='allBKG',
            objsyst='nominal',
            table='events',
            location='scratch-submit',
        )
        herwig_signal = build_pq_dataset(
            configsuite='BasicConfig',
            runtag=RUNTAG,
            dataset='Herwig_inclusive',
            objsyst='nominal',
            table='events',
            location='scratch-submit',
        )
        return DatasetStack(
            key='allMC_herwig_signal',
            color='#D62728',
            label='MC signal+background (Herwig signal)',
            datasets=[bkg, herwig_signal],
            showstack=True,
        )

    raise ValueError(f'Unsupported mc_mode: {mc_mode}')


def weighted_mean(y, yerr):
    finite = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if not np.any(finite):
        return np.nan, np.nan
    w = 1.0 / np.square(yerr[finite])
    mu = np.sum(w * y[finite]) / np.sum(w)
    muerr = np.sqrt(1.0 / np.sum(w))
    return float(mu), float(muerr)


def fit_log_spline(x, y, yerr, smoothing_scales, label):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0) & (y > 0)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    yerr = np.asarray(yerr[finite], dtype=float)

    if len(x) < 4:
        raise ValueError(f'Need >=4 points to fit log-spline for {label}')

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]

    log_y = np.log(y)
    log_yerr = np.clip(yerr / y, 1e-6, None)

    models = []
    k = min(3, len(x) - 1)
    for s in smoothing_scales:
        spline = UnivariateSpline(x, log_y, w=1.0 / log_yerr, s=float(s), k=k)
        pred = np.exp(spline(x))
        chi2 = float(np.sum(np.square((y - pred) / yerr)))
        npar = len(spline.get_coeffs())
        aic = chi2 + 2 * npar
        models.append({
            'smoothing': float(s),
            'spline': spline,
            'chi2': chi2,
            'ndof': max(len(x) - npar, 1),
            'aic': aic,
            'label': label,
        })

    models.sort(key=lambda item: item['aic'])
    return models[0], models


def fit_linear_spline(x, y, yerr, smoothing_scales, label):
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    yerr = np.asarray(yerr[finite], dtype=float)

    if len(x) < 4:
        raise ValueError(f'Need >=4 points to fit spline for {label}')

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]

    models = []
    k = min(3, len(x) - 1)
    for s in smoothing_scales:
        spline = UnivariateSpline(x, y, w=1.0 / yerr, s=float(s), k=k)
        pred = spline(x)
        chi2 = float(np.sum(np.square((y - pred) / yerr)))
        npar = len(spline.get_coeffs())
        aic = chi2 + 2 * npar
        models.append({
            'smoothing': float(s),
            'spline': spline,
            'chi2': chi2,
            'ndof': max(len(x) - npar, 1),
            'aic': aic,
            'label': label,
        })

    models.sort(key=lambda item: item['aic'])
    return models[0], models


def evaluate_log_fit(fit, x):
    x = np.asarray(x, dtype=float)
    return np.exp(fit['spline'](x))


def evaluate_linear_fit(fit, x):
    x = np.asarray(x, dtype=float)
    return fit['spline'](x)


def enforce_monotonic(values, mode='auto'):
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return y.copy(), 'none'

    dy = np.diff(y)
    trend = float(np.nanmedian(dy)) if np.any(np.isfinite(dy)) else 0.0

    if mode == 'auto':
        mode = 'non-decreasing' if trend >= 0 else 'non-increasing'

    if mode == 'non-decreasing':
        ymono = np.maximum.accumulate(y)
    elif mode == 'non-increasing':
        ymono = np.minimum.accumulate(y)
    else:
        raise ValueError(f'Unsupported monotonic mode: {mode}')

    return ymono, mode


def evaluate_result_curve(result, x):
    x = np.asarray(x, dtype=float)
    return np.interp(x, result['curve_x'], result['curve_y'])


def select_smoothest_near_best(models, aic_tolerance):
    if len(models) == 0:
        raise ValueError('No spline models available for selection')
    min_aic = min(item['aic'] for item in models)
    candidates = [item for item in models if item['aic'] <= min_aic + float(aic_tolerance)]
    if len(candidates) == 0:
        return models[0]
    return sorted(candidates, key=lambda item: item['smoothing'], reverse=True)[0]


def compute_ratio_and_uncertainty(h_data, h_mc):
    data = np.asarray(h_data.values(), dtype=float)
    data_var = np.asarray(h_data.variances(), dtype=float)
    mc = np.asarray(h_mc.values(), dtype=float)
    mc_var = np.asarray(h_mc.variances(), dtype=float)

    ratio = np.full_like(data, np.nan, dtype=float)
    ratioerr = np.full_like(data, np.nan, dtype=float)

    valid = (data > 0.0) & (mc > 0.0) & (data_var >= 0.0) & (mc_var >= 0.0)
    ratio[valid] = data[valid] / mc[valid]

    rel_data2 = np.zeros_like(data)
    rel_mc2 = np.zeros_like(mc)
    rel_data2[valid] = data_var[valid] / np.square(data[valid])
    rel_mc2[valid] = mc_var[valid] / np.square(mc[valid])

    ratioerr[valid] = ratio[valid] * np.sqrt(np.clip(rel_data2[valid] + rel_mc2[valid], 0.0, None))
    return ratio, ratioerr, valid


def inclusive_zy_ratio_from_hists(h_data, h_mc):
    data = np.asarray(h_data.values(), dtype=float).sum(axis=0)
    data_var = np.asarray(h_data.variances(), dtype=float).sum(axis=0)
    mc = np.asarray(h_mc.values(), dtype=float).sum(axis=0)
    mc_var = np.asarray(h_mc.variances(), dtype=float).sum(axis=0)

    ratio = np.full_like(data, np.nan, dtype=float)
    ratioerr = np.full_like(data, np.nan, dtype=float)
    valid = (data > 0.0) & (mc > 0.0) & (data_var >= 0.0) & (mc_var >= 0.0)

    ratio[valid] = data[valid] / mc[valid]
    rel_data2 = np.zeros_like(data)
    rel_mc2 = np.zeros_like(mc)
    rel_data2[valid] = data_var[valid] / np.square(data[valid])
    rel_mc2[valid] = mc_var[valid] / np.square(mc[valid])
    ratioerr[valid] = ratio[valid] * np.sqrt(np.clip(rel_data2[valid] + rel_mc2[valid], 0.0, None))
    return ratio, ratioerr, valid


def derive_zy_shape(zy_centers, yvals, yerrs):

    best_raw, all_models = fit_log_spline(zy_centers, yvals, yerrs, S_FZY, label='fzy')
    best = select_smoothest_near_best(all_models, FZY_AIC_SMOOTH_TOLERANCE)
    smooth_floor_candidates = [item for item in all_models if item['smoothing'] >= FZY_MIN_SMOOTHING]
    if len(smooth_floor_candidates) > 0:
        best_floor = sorted(smooth_floor_candidates, key=lambda item: item['aic'])[0]
        best = best_floor

    if best['smoothing'] != best_raw['smoothing']:
        print(
            'f(Zy) smoothness override: '
            f"AIC-best s={best_raw['smoothing']:.3g}, "
            f"selected smoother s={best['smoothing']:.3g} "
            f"within dAIC<={FZY_AIC_SMOOTH_TOLERANCE:.2f}"
        )
    raw_shape = np.clip(evaluate_log_fit(best, zy_centers), 1e-3, None)
    shape, mono_mode = enforce_monotonic(raw_shape, mode='auto')
    return {
        'best': best,
        'all': all_models,
        'points_x': zy_centers,
        'points_y': yvals,
        'points_yerr': yerrs,
        'shape': shape,
        'curve_x': zy_centers,
        'curve_y': shape,
        'mono_mode': mono_mode,
    }


def derive_alpha(logzpt_centers, zy_shape, ratio_map, ratioerr_map, valid_map):
    g = zy_shape - 1.0
    alpha_points = np.full(len(logzpt_centers), np.nan, dtype=float)
    alpha_errors = np.full(len(logzpt_centers), np.nan, dtype=float)

    for ix in range(len(logzpt_centers)):
        mask = valid_map[ix, :] & np.isfinite(g) & (np.abs(g) > 1e-9)
        if np.count_nonzero(mask) < 2:
            continue

        r = ratio_map[ix, mask]
        e = ratioerr_map[ix, mask]
        gk = g[mask]
        w = 1.0 / np.square(e)

        denom = np.sum(w * np.square(gk))
        if denom <= 0:
            continue
        num = np.sum(w * gk * (r - 1.0))
        alpha_points[ix] = num / denom
        alpha_errors[ix] = np.sqrt(1.0 / denom)

    finite = np.isfinite(alpha_points) & np.isfinite(alpha_errors) & (alpha_errors > 0)
    if np.count_nonzero(finite) < 4:
        raise ValueError('Insufficient alpha(Zpt) support points after quality filtering')

    median_err = float(np.median(alpha_errors[finite]))
    stable = finite & (alpha_errors <= 5.0 * median_err)
    if np.count_nonzero(stable) < 4:
        stable = finite

    alpha_points_fit = np.where(stable, alpha_points, np.nan)
    alpha_errors_fit = np.where(stable, alpha_errors, np.nan)

    best, all_models = fit_linear_spline(
        logzpt_centers,
        alpha_points_fit,
        alpha_errors_fit,
        S_ALPHA,
        label='alpha',
    )

    alpha_eval = evaluate_linear_fit(best, logzpt_centers)

    lo = float(np.min(alpha_points_fit[stable]))
    hi = float(np.max(alpha_points_fit[stable]))
    pad = max(0.1, 0.25 * (hi - lo + 1e-6))
    alpha_eval = np.clip(alpha_eval, lo - pad, hi + pad)
    alpha_eval = np.clip(alpha_eval, ALPHA_MIN, 3.0)
    alpha_eval, mono_mode = enforce_monotonic(alpha_eval, mode='auto')

    return {
        'best': best,
        'all': all_models,
        'points_x': logzpt_centers,
        'points_y': alpha_points,
        'points_yerr': alpha_errors,
        'alpha': alpha_eval,
        'curve_x': logzpt_centers,
        'curve_y': alpha_eval,
        'mono_mode': mono_mode,
    }


def build_factorized_surface(alpha_x, zy_shape):
    surface = 1.0 + alpha_x[:, None] * (zy_shape[None, :] - 1.0)
    return np.clip(surface, MIN_WEIGHT, MAX_WEIGHT)


def fit_factorized_variant(
    label,
    ratio_map,
    ratioerr_map,
    valid_map,
    logzpt_centers,
    zy_centers,
    inclusive_zy_ratio,
    inclusive_zy_ratioerr,
):
    fzy = derive_zy_shape(zy_centers, inclusive_zy_ratio, inclusive_zy_ratioerr)
    alpha = derive_alpha(logzpt_centers, fzy['shape'], ratio_map, ratioerr_map, valid_map)
    model = build_factorized_surface(alpha['alpha'], fzy['shape'])

    residual = np.full_like(ratio_map, np.nan, dtype=float)
    mask = valid_map & (model > 0)
    residual[mask] = ratio_map[mask] / model[mask] - 1.0

    return {
        'label': label,
        'fzy': fzy,
        'alpha': alpha,
        'model': model,
        'residual': residual,
    }


def plot_heatmap(values, xedges, yedges, path, title, cbar_label, vmin=None, vmax=None, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(10, 7))
    mesh = ax.pcolormesh(xedges, yedges, values.T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    ax.set_xlabel('log10(Zpt)')
    ax.set_ylabel('|Zy|')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_zy_shape_fit(result, path):
    x = result['points_x']
    y = result['points_y']
    yerr = result['points_yerr']
    xx = np.linspace(np.min(x), np.max(x), 400)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(x, y, yerr=yerr, fmt='o', ms=5, capsize=2, label='Per-bin |Zy| ratio points')
    ax.plot(
        xx,
        evaluate_result_curve(result, xx),
        linewidth=2.3,
        label=(
            f"Monotonic model s={result['best']['smoothing']:.3g}, "
            f"mode={result['mono_mode']}"
        ),
    )
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('|Zy|')
    ax.set_ylabel('Data / MC')
    ax.set_title('Continuous |Zy| model f(|Zy|)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_alpha_fit(result, path):
    x = result['points_x']
    y = result['points_y']
    yerr = result['points_yerr']
    xx = np.linspace(np.min(x), np.max(x), 400)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(x, y, yerr=yerr, fmt='o', ms=5, capsize=2, label='Per-bin alpha points')
    ax.plot(
        xx,
        evaluate_result_curve(result, xx),
        linewidth=2.3,
        label=(
            f"Monotonic model s={result['best']['smoothing']:.3g}, "
            f"mode={result['mono_mode']}"
        ),
    )
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('log10(Zpt)')
    ax.set_ylabel('alpha(Zpt)')
    ax.set_title('Zpt-controlled Zy strength alpha(Zpt)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_slice_overlays(logzpt_centers, zy_centers, ratio_map, ratioerr_map, model_map, valid_map, path):
    quantiles = [0.2, 0.5, 0.8]
    slice_indices = [int(np.clip(np.floor(q * (len(logzpt_centers) - 1)), 0, len(logzpt_centers) - 1)) for q in quantiles]

    fig, axes = plt.subplots(1, len(slice_indices), figsize=(16, 5), sharey=True)
    for ax, idx in zip(axes, slice_indices):
        mask = valid_map[idx, :]
        ax.errorbar(
            zy_centers[mask],
            ratio_map[idx, :][mask],
            yerr=ratioerr_map[idx, :][mask],
            fmt='o',
            ms=4,
            capsize=2,
            label='Data/MC',
        )
        ax.plot(zy_centers, model_map[idx, :], linewidth=2, label='Model')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'log10(Zpt) ~= {logzpt_centers[idx]:.2f}')
        ax.set_xlabel('|Zy|')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Data / MC')
    axes[0].legend()
    fig.suptitle('|Zy| slices at representative Zpt values')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def best_fit_alpha_for_slice(ratio_slice, ratioerr_slice, valid_slice, zy_shape):
    g = zy_shape - 1.0
    mask = valid_slice & np.isfinite(g) & (np.abs(g) > 1e-9)
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan

    r = ratio_slice[mask]
    e = ratioerr_slice[mask]
    gk = g[mask]
    w = 1.0 / np.square(e)
    denom = np.sum(w * np.square(gk))
    if denom <= 0:
        return np.nan, np.nan
    alpha = np.sum(w * gk * (r - 1.0)) / denom
    alpha = max(float(alpha), ALPHA_MIN)
    alpha_err = np.sqrt(1.0 / denom)
    return float(alpha), float(alpha_err)


def plot_alpha_bin_validation(logzpt_centers, zy_centers, ratio_map, ratioerr_map, valid_map, zy_shape, path):
    quantiles = [0.2, 0.5, 0.8]
    slice_indices = [int(np.clip(np.floor(q * (len(logzpt_centers) - 1)), 0, len(logzpt_centers) - 1)) for q in quantiles]

    fig, axes = plt.subplots(1, len(slice_indices), figsize=(16, 5), sharey=True)
    if len(slice_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, slice_indices):
        valid_slice = valid_map[idx, :]
        alpha, alpha_err = best_fit_alpha_for_slice(
            ratio_map[idx, :],
            ratioerr_map[idx, :],
            valid_slice,
            zy_shape,
        )
        model = np.clip(1.0 + alpha * (zy_shape - 1.0), MIN_WEIGHT, MAX_WEIGHT)

        ax.errorbar(
            zy_centers[valid_slice],
            ratio_map[idx, :][valid_slice],
            yerr=ratioerr_map[idx, :][valid_slice],
            fmt='o',
            ms=4,
            capsize=2,
            label='Data/MC',
        )
        ax.plot(zy_centers, model, linewidth=2.2, label='Best-fit alpha model')
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f'log10(Zpt)~{logzpt_centers[idx]:.2f}\nalpha={alpha:.3f}+-{alpha_err:.3f}')
        ax.set_xlabel('|Zy|')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Data / MC')
    axes[0].legend()
    fig.suptitle('Factorization check with per-slice best-fit alpha')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_zy_sign_symmetry_validation(
    abs_zy,
    ratio_pos,
    err_pos,
    ratio_neg_mirror,
    err_neg_mirror,
    path,
):
    valid = (
        np.isfinite(abs_zy)
        & np.isfinite(ratio_pos)
        & np.isfinite(err_pos)
        & np.isfinite(ratio_neg_mirror)
        & np.isfinite(err_neg_mirror)
        & (err_pos > 0)
        & (err_neg_mirror > 0)
    )

    x = abs_zy[valid]
    rp = ratio_pos[valid]
    ep = err_pos[valid]
    rn = ratio_neg_mirror[valid]
    en = err_neg_mirror[valid]

    chi2 = float(np.sum(np.square((rp - rn) / np.sqrt(np.square(ep) + np.square(en))))) if len(x) > 0 else np.nan
    ndof = max(len(x), 1)

    denom = 0.5 * (rp + rn)
    frac_asym = np.full_like(rp, np.nan)
    frac_asym_err = np.full_like(rp, np.nan)
    nonzero = np.abs(denom) > 1e-9
    frac_asym[nonzero] = (rp[nonzero] - rn[nonzero]) / denom[nonzero]
    frac_asym_err[nonzero] = np.sqrt(np.square(ep[nonzero]) + np.square(en[nonzero])) / np.abs(denom[nonzero])

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2.4, 1.4]},
    )

    ax_top.errorbar(x, rp, yerr=ep, fmt='o', ms=5, capsize=2, label='Data/MC at +Zy')
    ax_top.errorbar(x, rn, yerr=en, fmt='s', ms=5, capsize=2, label='Data/MC at -Zy (mirrored)')
    ax_top.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax_top.set_ylabel('Data / MC')
    ax_top.set_title(f'Zy sign symmetry check (inclusive in Zpt): chi2/ndof = {chi2:.2f}/{ndof}')
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    ax_bot.errorbar(x, frac_asym, yerr=frac_asym_err, fmt='o', ms=4, capsize=2)
    ax_bot.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax_bot.set_xlabel('|Zy|')
    ax_bot.set_ylabel('Frac. asym')
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def extract_pos_neg_mirrored(zy_signed_centers, ratio_slice, err_slice, valid_slice):
    pos_mask = (zy_signed_centers > 0) & valid_slice
    neg_mask = (zy_signed_centers < 0) & valid_slice

    if np.count_nonzero(pos_mask) < 2 or np.count_nonzero(neg_mask) < 2:
        return None

    x_pos = zy_signed_centers[pos_mask]
    ratio_pos = ratio_slice[pos_mask]
    err_pos = err_slice[pos_mask]

    x_neg_abs = np.abs(zy_signed_centers[neg_mask])
    neg_order = np.argsort(x_neg_abs)
    x_neg_abs = x_neg_abs[neg_order]
    ratio_neg = ratio_slice[neg_mask][neg_order]
    err_neg = err_slice[neg_mask][neg_order]

    ratio_neg_interp = np.interp(x_pos, x_neg_abs, ratio_neg)
    err_neg_interp = np.interp(x_pos, x_neg_abs, err_neg)
    return x_pos, ratio_pos, err_pos, ratio_neg_interp, err_neg_interp


def plot_zy_sign_symmetry_validation_zpt_bins(
    logzpt_centers,
    zy_signed_centers,
    ratio_map,
    ratioerr_map,
    valid_map,
    path,
):
    quantiles = [0.2, 0.5, 0.8]
    slice_indices = [int(np.clip(np.floor(q * (len(logzpt_centers) - 1)), 0, len(logzpt_centers) - 1)) for q in quantiles]

    fig, axes = plt.subplots(2, len(slice_indices), figsize=(16, 8), sharex='col')
    if len(slice_indices) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for col, idx in enumerate(slice_indices):
        packed = extract_pos_neg_mirrored(
            zy_signed_centers,
            ratio_map[idx, :],
            ratioerr_map[idx, :],
            valid_map[idx, :],
        )

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        if packed is None:
            ax_top.text(0.5, 0.5, 'Insufficient +/- Zy bins', transform=ax_top.transAxes, ha='center', va='center')
            ax_top.set_title(f'log10(Zpt)~{logzpt_centers[idx]:.2f}')
            ax_bot.set_xlabel('|Zy|')
            continue

        x, rp, ep, rn, en = packed
        sigma = np.sqrt(np.square(ep) + np.square(en))
        chi2 = float(np.sum(np.square((rp - rn) / sigma)))
        ndof = max(len(x), 1)

        denom = 0.5 * (rp + rn)
        frac_asym = np.full_like(rp, np.nan)
        frac_asym_err = np.full_like(rp, np.nan)
        nonzero = np.abs(denom) > 1e-9
        frac_asym[nonzero] = (rp[nonzero] - rn[nonzero]) / denom[nonzero]
        frac_asym_err[nonzero] = sigma[nonzero] / np.abs(denom[nonzero])

        ax_top.errorbar(x, rp, yerr=ep, fmt='o', ms=4, capsize=2, label='+Zy')
        ax_top.errorbar(x, rn, yerr=en, fmt='s', ms=4, capsize=2, label='-Zy mirrored')
        ax_top.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        ax_top.set_title(f'log10(Zpt)~{logzpt_centers[idx]:.2f}\nchi2/ndof={chi2:.2f}/{ndof}')
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel('Data / MC')
            ax_top.legend()

        ax_bot.errorbar(x, frac_asym, yerr=frac_asym_err, fmt='o', ms=4, capsize=2)
        ax_bot.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        ax_bot.set_xlabel('|Zy|')
        ax_bot.grid(True, alpha=0.3)
        if col == 0:
            ax_bot.set_ylabel('Frac. asym')

    fig.suptitle('Zy sign symmetry by Zpt bin')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_multibinning_correction(name, x_edges, y_edges, values):
    correction = schemav2.Correction(
        name=name,
        description='2D factorized kinematic reweighting in (log10(Zpt), |Zy|)',
        version=1,
        inputs=[
            schemav2.Variable(name='log10zpt', type='real'),
            schemav2.Variable(name='abszy', type='real'),
        ],
        output=schemav2.Variable(name='weight', type='real'),
        data=schemav2.MultiBinning(
            nodetype='multibinning',
            inputs=['log10zpt', 'abszy'],
            edges=[
                [float(edge) for edge in x_edges],
                [float(edge) for edge in y_edges],
            ],
            content=[float(v) for v in values.ravel(order='C')],
            flow='clamp',
        ),
    )
    return correction


def build_1d_binning_correction(name, edges, values, input_name='jpt'):
    correction = schemav2.Correction(
        name=name,
        description='1D Jpt reweighting after full 2D Z kinematic reweighting',
        version=1,
        inputs=[schemav2.Variable(name=input_name, type='real')],
        output=schemav2.Variable(name='weight', type='real'),
        data=schemav2.Binning(
            nodetype='binning',
            input=input_name,
            edges=[float(edge) for edge in edges],
            content=[float(v) for v in values],
            flow='clamp',
        ),
    )
    return correction


def plot_jpt_ratio_fit(x, y, yerr, fit_dict, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, yerr=yerr, fmt='o', ms=4, capsize=2, alpha=0.6, label='Data/MC')

    xgrid = np.linspace(np.min(x), np.max(x), 400)
    colors = {'nominal': 'C0', 'up': 'C2', 'down': 'C3'}
    for variant in ['nominal', 'up', 'down']:
        curve = np.clip(evaluate_log_fit(fit_dict[variant], xgrid), MIN_WEIGHT, MAX_WEIGHT)
        ax.plot(
            xgrid,
            curve,
            linewidth=2,
            color=colors.get(variant),
            label=f"{variant}: s={fit_dict[variant]['smoothing']:.3g}",
        )

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Jpt [GeV]')
    ax.set_ylabel('Data / MC')
    ax.set_title('Post-2D Jpt reweighting fit')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_post2d_jpt_reweighting(
    dset_mc,
    dset_data,
    cut,
    var_logzpt,
    var_abszy,
    data_weight,
    input_1d_correction,
    input_2d_correction,
    output_dir,
):
    var_jpt = splt.variable.BasicVariable('Jpt')
    mc_weight_post2d = splt.variable.ProductVariable(
        'wt_nominal',
        splt.variable.CorrectionlibVariable(
            [var_logzpt],
            str(input_1d_correction),
            INPUT_1D_KEY,
        ),
        splt.variable.CorrectionlibVariable(
            [var_logzpt, var_abszy],
            str(input_2d_correction),
            'zptabsy_reweight_nominal',
        ),
    )

    ax_jpt = hist.axis.Variable(JPT_BINS)
    h_mc_jpt = dset_mc.fill_hist(var_jpt, cut, mc_weight_post2d, ax_jpt)
    h_data_jpt = dset_data.fill_hist(var_jpt, cut, data_weight, ax_jpt)

    ratio_jpt, ratioerr_jpt, valid_jpt = compute_ratio_and_uncertainty(h_data_jpt, h_mc_jpt)
    jpt_edges = np.asarray(h_data_jpt.axes[0].edges, dtype=float)
    jpt_centers = np.asarray(h_data_jpt.axes[0].centers, dtype=float)

    shifted = {
        'nominal': ratio_jpt,
        'up': np.clip(ratio_jpt + ratioerr_jpt, 1e-3, None),
        'down': np.clip(ratio_jpt - ratioerr_jpt, 1e-3, None),
    }

    jpt_results = {}
    for variant in ['nominal', 'up', 'down']:
        best, all_models = fit_log_spline(
            jpt_centers,
            shifted[variant],
            ratioerr_jpt,
            S_JPT,
            label=f'jpt_{variant}',
        )
        values = np.full_like(jpt_centers, np.nan, dtype=float)
        values[valid_jpt] = np.clip(evaluate_log_fit(best, jpt_centers[valid_jpt]), MIN_WEIGHT, MAX_WEIGHT)
        values[~valid_jpt] = 1.0

        jpt_results[variant] = {
            'best': best,
            'all': all_models,
            'values': values,
        }

        print(f'[{variant}] post-2D Jpt spline candidates ranked by AIC:')
        for candidate in all_models:
            print(
                f"  s={candidate['smoothing']:.3f} chi2={candidate['chi2']:.3f} "
                f"ndof={candidate['ndof']} aic={candidate['aic']:.3f}"
            )

    plot_jpt_ratio_fit(
        jpt_centers,
        ratio_jpt,
        ratioerr_jpt,
        {key: jpt_results[key]['best'] for key in ['nominal', 'up', 'down']},
        output_dir / 'jpt_post2d_ratio_fit.png',
    )

    corrections = [
        build_1d_binning_correction('jpt_reweight_post2d_nominal', jpt_edges, jpt_results['nominal']['values']),
        build_1d_binning_correction('jpt_reweight_post2d_up', jpt_edges, jpt_results['up']['values']),
        build_1d_binning_correction('jpt_reweight_post2d_down', jpt_edges, jpt_results['down']['values']),
    ]
    cset = schemav2.CorrectionSet(schema_version=2, corrections=corrections)

    json_path = output_dir / 'jpt_reweight_post2d.json'
    json_path.write_text(cset.model_dump_json(indent=2) + '\n')

    loaded = CorrectionSet.from_file(str(json_path))
    probe_points = [jpt_edges[0] - 20.0, jpt_centers[len(jpt_centers) // 2], jpt_edges[-1] + 20.0]
    print('\nPost-2D Jpt CorrectionLib round-trip checks:')
    for point in probe_points:
        val_nom = loaded['jpt_reweight_post2d_nominal'].evaluate(float(point))
        val_up = loaded['jpt_reweight_post2d_up'].evaluate(float(point))
        val_dn = loaded['jpt_reweight_post2d_down'].evaluate(float(point))
        print(f'  (Jpt={point:.4f}) -> nom={val_nom:.6f} up={val_up:.6f} down={val_dn:.6f}')


def run_for_mode(mc_mode):
    output_dir = get_output_dir(mc_mode)
    input_1d_correction = get_input_1d_correction(mc_mode)

    if not input_1d_correction.exists():
        raise FileNotFoundError(
            f'Missing input 1D correction at {input_1d_correction}. Run plotting/kinreweight_1D.py first.'
        )

    print(f'\n=== Running 2D Z reweighting for mode: {mc_mode} ===')

    dset_mc = build_mc_stack_for_mode(mc_mode)

    dset_data = build_pq_dataset_stack(
        configsuite='BasicConfig',
        runtag=RUNTAG,
        dataset='DATA',
        objsyst='DATA',
        table='events',
        location='scratch-submit',
        no_count=True,
    )

    dset_mc.compute_weight(dset_data.lumi)
    dset_data.compute_weight(dset_data.lumi)

    cut = splt.cut.AndCuts([
        splt.cut.EqualsCut('nMu', 2),
        splt.cut.EqualsCut('nEle', 0),
        splt.cut.LessThanCut('numMediumB', 2),
        splt.cut.TwoSidedCut('Zmass', 91.1876 - 15, 91.1876 + 15),
    ])

    var_logzpt = splt.variable.LogVariable('Zpt', 10)
    var_abszy = splt.variable.AbsVariable('Zy')
    var_zy_signed = splt.variable.BasicVariable('Zy')

    mc_weight = splt.variable.ProductVariable(
        'wt_nominal',
        splt.variable.CorrectionlibVariable(
            [var_logzpt],
            str(input_1d_correction),
            INPUT_1D_KEY,
        ),
    )
    data_weight = splt.variable.BasicVariable('wt_nominal')

    ax_zpt = hist.axis.Variable(ZPT_BINS)
    ax_zy = hist.axis.Variable(ABSY_BINS)

    h_mc = dset_mc.fill_hist_2D(var_logzpt, var_abszy, cut, mc_weight, ax_zpt, ax_zy)
    h_data = dset_data.fill_hist_2D(var_logzpt, var_abszy, cut, data_weight, ax_zpt, ax_zy)

    signed_zy_edges = np.concatenate((-ABSY_BINS[::-1], ABSY_BINS[1:]))
    ax_zy_signed = hist.axis.Variable(signed_zy_edges)
    h_mc_zy_signed = dset_mc.fill_hist(var_zy_signed, cut, mc_weight, ax_zy_signed)
    h_data_zy_signed = dset_data.fill_hist(var_zy_signed, cut, data_weight, ax_zy_signed)
    h_mc_2d_zy_signed = dset_mc.fill_hist_2D(var_logzpt, var_zy_signed, cut, mc_weight, ax_zpt, ax_zy_signed)
    h_data_2d_zy_signed = dset_data.fill_hist_2D(var_logzpt, var_zy_signed, cut, data_weight, ax_zpt, ax_zy_signed)

    ratio_zy_signed, ratioerr_zy_signed, valid_zy_signed = compute_ratio_and_uncertainty(h_data_zy_signed, h_mc_zy_signed)
    ratio_2d_zy_signed, ratioerr_2d_zy_signed, valid_2d_zy_signed = compute_ratio_and_uncertainty(
        h_data_2d_zy_signed,
        h_mc_2d_zy_signed,
    )
    zy_signed_centers = np.asarray(h_data_zy_signed.axes[0].centers, dtype=float)
    packed_inclusive = extract_pos_neg_mirrored(zy_signed_centers, ratio_zy_signed, ratioerr_zy_signed, valid_zy_signed)
    if packed_inclusive is not None:
        common_abs_zy, ratio_pos, err_pos, ratio_neg_interp, err_neg_interp = packed_inclusive
        plot_zy_sign_symmetry_validation(
            common_abs_zy,
            ratio_pos,
            err_pos,
            ratio_neg_interp,
            err_neg_interp,
            output_dir / 'zy_sign_symmetry_validation.png',
        )

    zpt_centers = np.asarray(h_data_2d_zy_signed.axes[0].centers, dtype=float)
    plot_zy_sign_symmetry_validation_zpt_bins(
        zpt_centers,
        zy_signed_centers,
        ratio_2d_zy_signed,
        ratioerr_2d_zy_signed,
        valid_2d_zy_signed,
        output_dir / 'zy_sign_symmetry_validation_zpt_bins.png',
    )

    ratio, ratioerr, valid = compute_ratio_and_uncertainty(h_data, h_mc)
    ratio_zy_inclusive, ratioerr_zy_inclusive, valid_zy_inclusive = inclusive_zy_ratio_from_hists(h_data, h_mc)

    x_edges = np.asarray(h_data.axes[0].edges, dtype=float)
    y_edges = np.asarray(h_data.axes[1].edges, dtype=float)
    x_centers = np.asarray(h_data.axes[0].centers, dtype=float)
    y_centers = np.asarray(h_data.axes[1].centers, dtype=float)

    plot_heatmap(
        np.asarray(h_data.values(), dtype=float),
        x_edges,
        y_edges,
        output_dir / 'data_heatmap.png',
        'DATA yield in (log10(Zpt), |Zy|)',
        'Yield',
    )
    plot_heatmap(
        np.asarray(h_mc.values(), dtype=float),
        x_edges,
        y_edges,
        output_dir / 'mc_1dpreweighted_heatmap.png',
        'MC yield after 1D Zpt reweighting',
        'Yield',
    )
    plot_heatmap(
        ratio,
        x_edges,
        y_edges,
        output_dir / 'data_mc_ratio_heatmap.png',
        'DATA/MC ratio before 2D fit',
        'Data / MC',
        vmin=0.9,
        vmax=1.1,
        cmap='coolwarm',
    )
    plot_heatmap(
        ratioerr,
        x_edges,
        y_edges,
        output_dir / 'data_mc_ratio_uncertainty_heatmap.png',
        'Uncertainty on DATA/MC ratio',
        'sigma(Data / MC)',
    )

    shifted_ratio = {
        'nominal': ratio,
        'up': np.clip(ratio + ratioerr, 1e-3, None),
        'down': np.clip(ratio - ratioerr, 1e-3, None),
    }
    shifted_ratio_zy_inclusive = {
        'nominal': ratio_zy_inclusive,
        'up': np.clip(ratio_zy_inclusive + ratioerr_zy_inclusive, 1e-3, None),
        'down': np.clip(ratio_zy_inclusive - ratioerr_zy_inclusive, 1e-3, None),
    }

    results = {}
    for variant in ['nominal', 'up', 'down']:
        results[variant] = fit_factorized_variant(
            label=variant,
            ratio_map=shifted_ratio[variant],
            ratioerr_map=ratioerr,
            valid_map=valid,
            logzpt_centers=x_centers,
            zy_centers=y_centers,
            inclusive_zy_ratio=shifted_ratio_zy_inclusive[variant][valid_zy_inclusive],
            inclusive_zy_ratioerr=ratioerr_zy_inclusive[valid_zy_inclusive],
        )

        print(f'[{variant}] f(Zy) spline candidates ranked by AIC:')
        for candidate in results[variant]['fzy']['all']:
            print(
                f"  s={candidate['smoothing']:.3f} chi2={candidate['chi2']:.3f} "
                f"ndof={candidate['ndof']} aic={candidate['aic']:.3f}"
            )

        print(f'[{variant}] alpha(Zpt) spline candidates ranked by AIC:')
        for candidate in results[variant]['alpha']['all']:
            print(
                f"  s={candidate['smoothing']:.3f} chi2={candidate['chi2']:.3f} "
                f"ndof={candidate['ndof']} aic={candidate['aic']:.3f}"
            )

        plot_zy_shape_fit(results[variant]['fzy'], output_dir / f'fzy_fit_{variant}.png')
        plot_alpha_fit(results[variant]['alpha'], output_dir / f'alpha_fit_{variant}.png')
        plot_heatmap(
            results[variant]['model'],
            x_edges,
            y_edges,
            output_dir / f'model_heatmap_{variant}.png',
            f'Factorized model ({variant})',
            'Weight',
            vmin=0.90,
            vmax=1.10,
            cmap='coolwarm',
        )
        plot_heatmap(
            results[variant]['residual'],
            x_edges,
            y_edges,
            output_dir / f'residual_heatmap_{variant}.png',
            f'Residual ratio/model - 1 ({variant})',
            'Residual',
            vmin=-0.1,
            vmax=0.1,
            cmap='coolwarm',
        )

    plot_slice_overlays(
        x_centers,
        y_centers,
        ratio,
        ratioerr,
        results['nominal']['model'],
        valid,
        output_dir / 'slice_overlays_nominal.png',
    )

    plot_alpha_bin_validation(
        x_centers,
        y_centers,
        ratio,
        ratioerr,
        valid,
        results['nominal']['fzy']['shape'],
        output_dir / 'factorization_alpha_bins_nominal.png',
    )

    corrected_mc = np.asarray(h_mc.values(), dtype=float) * results['nominal']['model']
    closure = np.full_like(ratio, np.nan, dtype=float)
    closure_mask = corrected_mc > 0
    closure[closure_mask] = np.asarray(h_data.values(), dtype=float)[closure_mask] / corrected_mc[closure_mask]
    plot_heatmap(
        closure,
        x_edges,
        y_edges,
        output_dir / 'closure_heatmap_nominal.png',
        'Closure DATA / (MC * w2D)',
        'Closure',
        vmin=0.90,
        vmax=1.10,
        cmap='coolwarm',
    )

    corrections = [
        build_multibinning_correction('zptabsy_reweight_nominal', x_edges, y_edges, results['nominal']['model']),
        build_multibinning_correction('zptabsy_reweight_up', x_edges, y_edges, results['up']['model']),
        build_multibinning_correction('zptabsy_reweight_down', x_edges, y_edges, results['down']['model']),
    ]
    cset = schemav2.CorrectionSet(schema_version=2, corrections=corrections)

    json_path = output_dir / 'zptabsy_reweight_2d.json'
    json_path.write_text(cset.model_dump_json(indent=2) + '\n')

    loaded = CorrectionSet.from_file(str(json_path))
    probe_x = [x_edges[0] - 0.5, x_centers[len(x_centers) // 2], x_edges[-1] + 0.5]
    probe_y = [y_edges[0] - 0.5, 0.0, y_edges[-1] + 0.5]

    print('\nCorrectionLib round-trip checks:')
    for px in probe_x:
        for py in probe_y:
            val_nom = loaded['zptabsy_reweight_nominal'].evaluate(float(px), float(py))
            val_up = loaded['zptabsy_reweight_up'].evaluate(float(px), float(py))
            val_dn = loaded['zptabsy_reweight_down'].evaluate(float(px), float(py))
            print(
                f'  (log10Zpt={px:.4f}, |Zy|={py:.4f}) -> '
                f'nom={val_nom:.6f} up={val_up:.6f} down={val_dn:.6f}'
            )

    run_post2d_jpt_reweighting(
        dset_mc=dset_mc,
        dset_data=dset_data,
        cut=cut,
        var_logzpt=var_logzpt,
        var_abszy=var_abszy,
        data_weight=data_weight,
        input_1d_correction=input_1d_correction,
        input_2d_correction=json_path,
        output_dir=output_dir,
    )


def main():
    for mc_mode in MC_MODES:
        run_for_mode(mc_mode)


if __name__ == '__main__':
    main()
