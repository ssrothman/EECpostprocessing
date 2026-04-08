import simonplot as splt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import hist
from correctionlib import CorrectionSet, schemav2
from plotting.load_datasets import build_pq_dataset_stack
from scipy.interpolate import UnivariateSpline
from simonplot.util.histplot import simon_histplot_ratio

RUNTAG = 'Mar_01_2026'
OUTPUT_DIR = Path('ANplots') / 'kinreweight' / RUNTAG
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAIL_REGULARIZATION_LAST_N_POINTS = 3
TAIL_REGULARIZATION_FRACTION_FALLBACK = 0.2

def fit_ratio_splines(x, y, yerr, smoothing_scales=None, label='nominal'):
    """Fit smoothing splines to a ratio with optional label for diagnostics."""
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0) & (y > 0)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    yerr = np.asarray(yerr[finite], dtype=float)

    if len(x) < 4:
        raise ValueError('Need at least four finite ratio points to fit a smoothing spline')

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]

    log_y = np.log(y)
    log_yerr = yerr / y

    if smoothing_scales is None:
        smoothing_scales = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    models = []
    spline_degree = min(3, len(x) - 1)
    for smoothness in smoothing_scales:
        spline = UnivariateSpline(
            x,
            log_y,
            w=1.0 / log_yerr,
            s=float(smoothness),
            k=spline_degree,
        )

        pred = np.exp(spline(x))
        chi2 = float(np.sum(np.square((y - pred) / yerr)))
        npar = len(spline.get_coeffs())
        aic = chi2 + 2 * npar
        models.append({
            'smoothing': float(smoothness),
            'spline': spline,
            'chi2': chi2,
            'ndof': max(len(x) - npar, 1),
            'aic': aic,
            'label': label,
        })

    models.sort(key=lambda item: item['aic'])
    return models[0], models


def fit_shifted_ratio_splines(x, y, yerr, smoothing_scales=None):
    """Fit splines to nominal, up (y+yerr), and down (y-yerr) shifted ratios."""
    best_nom, all_nom = fit_ratio_splines(x, y, yerr, smoothing_scales, label='nominal')
    best_up, all_up = fit_ratio_splines(x, np.clip(y + yerr, 0.001, None), yerr, smoothing_scales, label='up')
    best_down, all_down = fit_ratio_splines(x, np.clip(y - yerr, 0.001, None), yerr, smoothing_scales, label='down')

    # sort the all_* lists by smoothing scale for consistent diagnostics
    all_nom.sort(key=lambda item: item['smoothing'])
    all_up.sort(key=lambda item: item['smoothing'])
    all_down.sort(key=lambda item: item['smoothing'])

    return {'nominal': best_nom, 'up': best_up, 'down': best_down}, {'nominal': all_nom, 'up': all_up, 'down': all_down}


def evaluate_fit(fit, x):
    """Evaluate a fit using its stabilized evaluator if present."""
    x = np.asarray(x, dtype=float)
    if 'eval' in fit:
        return fit['eval'](x)
    return np.exp(fit['spline'](x))


def add_tail_regularization_to_fit(
    fit,
    x_support,
    tail_last_n_points=TAIL_REGULARIZATION_LAST_N_POINTS,
    tail_fraction_fallback=TAIL_REGULARIZATION_FRACTION_FALLBACK,
):
    """Suppress edge wiggles by enforcing monotonic behavior in the right tail.

    By default, regularization starts at the x-value of the N-th point from the right.
    """
    x_support = np.asarray(x_support, dtype=float)
    x_sorted = np.sort(x_support)
    xmin = float(np.min(x_support))
    xmax = float(np.max(x_support))
    xgrid = np.linspace(xmin, xmax, 1200)
    ygrid = np.exp(fit['spline'](xgrid))

    if tail_last_n_points is not None and len(x_sorted) >= 2:
        n = int(np.clip(tail_last_n_points, 2, len(x_sorted)))
        tail_start = float(x_sorted[-n])
        tail_spec = f'last-{n}-points'
    else:
        tail_start = xmin + (1.0 - tail_fraction_fallback) * (xmax - xmin)
        tail_spec = f'fraction-{tail_fraction_fallback:.3g}'

    tail_mask = xgrid >= tail_start
    ytail = ygrid[tail_mask]

    if len(ytail) > 2:
        dy = np.diff(ytail)
        trend = float(np.median(dy))
        if trend <= 0:
            ytail = np.minimum.accumulate(ytail)
            tail_mode = 'non-increasing'
        else:
            ytail = np.maximum.accumulate(ytail)
            tail_mode = 'non-decreasing'
        ygrid[tail_mask] = ytail
    else:
        tail_mode = 'none'

    ygrid = np.clip(ygrid, 1e-6, None)

    def stabilized_eval(x):
        return np.interp(x, xgrid, ygrid)

    out = fit.copy()
    out['eval'] = stabilized_eval
    out['tail_mode'] = tail_mode
    out['tail_start'] = tail_start
    out['tail_spec'] = tail_spec
    return out


def add_tail_regularization_to_fit_dict(fit_dict, x_support):
    return {
        key: add_tail_regularization_to_fit(value, x_support)
        for key, value in fit_dict.items()
    }


def add_tail_regularization_to_all_fit_dict(all_fit_dict, x_support):
    return {
        key: [add_tail_regularization_to_fit(item, x_support) for item in fits]
        for key, fits in all_fit_dict.items()
    }


def build_dense_binning_correction(name, fit, xmin, xmax, n_bins=250):
    edges = np.linspace(xmin, xmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = evaluate_fit(fit, centers)
    correction = schemav2.Correction(
        name=name,
        description='Smooth Zpt reweighting from data/MC ratio sampled from a smoothing spline',
        version=1,
        inputs=[schemav2.Variable(name='x', type='real')],
        output=schemav2.Variable(name='weight', type='real'),
        data=schemav2.Binning(
            nodetype='binning',
            input='x',
            edges=[float(edge) for edge in edges],
            content=[float(value) for value in values],
            flow='clamp',
        ),
    )
    return correction, edges, values

def save_fit_diagnostic(x, y, yerr, fit_dict, path):
    """Plot nominal, up, down fits with error band and binned corrections."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, yerr=yerr, fmt='o', ms=4, capsize=2, label='Data/MC ratio', alpha=0.6)

    xgrid = np.linspace(np.min(x), np.max(x), 400)
    
    # Plot nominal and systematic variants
    colors = {'nominal': 'C0', 'up': 'C2', 'down': 'C3'}
    for variant in ['nominal', 'up', 'down']:
        if variant in fit_dict:
            spline_curve = evaluate_fit(fit_dict[variant], xgrid)
            label = f"{variant}: s={fit_dict[variant]['smoothing']:.3g}"
            ax.plot(xgrid, spline_curve, label=label, color=colors.get(variant), linewidth=2)
    
    ax.set_xlabel('log10(Zpt)')
    ax.set_ylabel('Data / MC')
    ax.set_title('Smooth Zpt reweighting fit with systematic variants')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_smoothing_scan_diagnostic(x, y, yerr, all_fit_dict, best_fit_dict, path):
    """Plot all smoothing-parameter curves and highlight the selected best one."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    variants = ['nominal', 'up', 'down']
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, max(len(all_fit_dict[v]) for v in variants)))
    xgrid = np.linspace(np.min(x), np.max(x), 400)

    for axis, variant in zip(axes, variants):
        fits = all_fit_dict[variant]
        axis.errorbar(x, y, yerr=yerr, fmt='o', ms=3, capsize=2, alpha=0.4, color='black', label='Data/MC ratio')

        for idx, fit in enumerate(fits):
            spline_curve = evaluate_fit(fit, xgrid)
            axis.plot(
                xgrid,
                spline_curve,
                color=colors[idx],
                alpha=0.85,
                linewidth=1.5,
                label=f"s={fit['smoothing']:.3g}",
            )

        best = best_fit_dict[variant]
        best_curve = evaluate_fit(best, xgrid)
        axis.plot(
            xgrid,
            best_curve,
            color='black',
            linewidth=2.6,
            linestyle='--',
            label=f"selected best s={best['smoothing']:.3g}",
        )

        axis.set_ylabel('Data / MC')
        axis.set_title(f'{variant} variation: all smoothing values')
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=4, fontsize=8)

    axes[-1].set_xlabel('log10(Zpt)')
    fig.suptitle('Smoothing scan for nominal/up/down reweighting curves', y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def smoothing_tag(smoothing):
    """Build a filesystem-friendly tag for smoothing values."""
    return f"{smoothing:.3g}".replace('.', 'p')


def main():
    dset_MC = build_pq_dataset_stack(
        configsuite='BasicConfig',
        runtag=RUNTAG,
        dataset='allMC',
        objsyst='nominal',
        table='events',
        location='scratch-submit',
    )

    dset_DATA = build_pq_dataset_stack(
        configsuite='BasicConfig',
        runtag=RUNTAG,
        dataset='DATA',
        objsyst='DATA',
        table='events',
        location='scratch-submit',
        no_count=True,
    )

    dset_MC.compute_weight(dset_DATA.lumi)
    dset_DATA.compute_weight(dset_DATA.lumi)

    cut = splt.cut.AndCuts([
        splt.cut.EqualsCut('nMu', 2),
        splt.cut.EqualsCut('nEle', 0),
        splt.cut.LessThanCut('numMediumB', 2),
        splt.cut.TwoSidedCut('Zmass', 91.1876 - 15, 91.1876 + 15),
    ])

    var1 = splt.variable.LogVariable('Zpt', 10)
    wt = splt.variable.BasicVariable('wt_nominal')

    bins = np.asarray([
        0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1,
        1.05,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.45,
        1.5,
        1.55,
        1.6,
        1.65,
        1.7,
        1.75,
        1.8,
        1.85,
        1.9,
        1.95,
        2,
        2.1,
        2.2,
        2.3,
        2.4,
        2.6,
        2.8,
        3,
        4,
    ])
    ax = hist.axis.Variable(bins)

    H_MC = dset_MC.fill_hist(var1, cut, wt, ax)
    H_DATA = dset_DATA.fill_hist(var1, cut, wt, ax)

    _, ratio, ratioerr = simon_histplot_ratio(H_DATA, H_MC)
    ratio_plot_path = OUTPUT_DIR / 'data_mc_ratio.png'
    plt.savefig(ratio_plot_path)
    plt.clf()

    centers = H_DATA.axes[0].centers
    best_fit, all_fits = fit_ratio_splines(centers, ratio, ratioerr)

    print('Spline candidates ranked by AIC:')
    for candidate in all_fits:
        print(
            f"  s={candidate['smoothing']:.3f} chi2={candidate['chi2']:.3f} "
            f"ndof={candidate['ndof']} aic={candidate['aic']:.3f}"
        )

    xmin, xmax = float(np.min(centers)), float(np.max(centers))
    
    # Build nominal correction
    correction_nom, edges_nom, values_nom = build_dense_binning_correction(
        name='zpt_reweight',
        xmin=xmin,
        xmax=xmax,
        fit=add_tail_regularization_to_fit(best_fit, centers),
    )
    
    # Option 2: Shifted ratio refitting
    print('\n=== Option 2: Shifted Ratio Refitting ===')
    shifted_best_fits, shifted_all_fits = fit_shifted_ratio_splines(centers, ratio, ratioerr)
    shifted_best_fits = add_tail_regularization_to_fit_dict(shifted_best_fits, centers)
    shifted_all_fits = add_tail_regularization_to_all_fit_dict(shifted_all_fits, centers)
    print('Shifted ratio spline candidates:')
    for variant in ['nominal', 'up', 'down']:
        fit = shifted_best_fits[variant]
        print(
            f'  {variant}: s={fit["smoothing"]:.3f} chi2={fit["chi2"]:.3f} '
            f'aic={fit["aic"]:.3f} tail={fit["tail_mode"]}'
        )
    
    # Build corrections for shifted method
    shift_corrections = {}
    for variant in ['nominal', 'up', 'down']:
        corr, edges, values = build_dense_binning_correction(
            name=f'zpt_reweight_{variant}',
            xmin=xmin,
            xmax=xmax,
            fit=shifted_best_fits[variant],
        )
        shift_corrections[variant] = {'correction': corr, 'edges': edges, 'values': values}
        
    # Plot diagnostics with all variants
    fit_plot_path = OUTPUT_DIR / 'data_mc_ratio_fit_options.png'
    save_fit_diagnostic(centers, ratio, ratioerr, shifted_best_fits, fit_plot_path)

    # Export one fit-options style figure per smoothing value.
    smoothings = sorted(
        set(item['smoothing'] for item in shifted_all_fits['nominal'])
        & set(item['smoothing'] for item in shifted_all_fits['up'])
        & set(item['smoothing'] for item in shifted_all_fits['down'])
    )
    for smoothing in smoothings:
        per_smoothing_fits = {
            variant: next(
                item for item in shifted_all_fits[variant]
                if np.isclose(item['smoothing'], smoothing)
            )
            for variant in ['nominal', 'up', 'down']
        }
        per_smoothing_path = OUTPUT_DIR / f"data_mc_ratio_fit_options_s{ smoothing_tag(smoothing) }.png"
        save_fit_diagnostic(
            centers,
            ratio,
            ratioerr,
            per_smoothing_fits,
            per_smoothing_path,
        )
    
    # Export option 2: shifted ratio
    corrections_list_shift = [shift_corrections[v]['correction'] for v in ['nominal', 'up', 'down']]
    correction_set_shift = schemav2.CorrectionSet(schema_version=2, corrections=corrections_list_shift)
    json_path_shift = OUTPUT_DIR / 'zpt_reweight_shifted_ratio.json'
    json_path_shift.write_text(correction_set_shift.model_dump_json(indent=2) + '\n')
        
    # Export original nominal for reference
    correction_set_nom = schemav2.CorrectionSet(schema_version=2, corrections=[correction_nom])
    json_path_nom = OUTPUT_DIR / 'zpt_reweight.json'
    json_path_nom.write_text(correction_set_nom.model_dump_json(indent=2) + '\n')
    
    # Test round-trip
    loaded = CorrectionSet.from_file(str(json_path_shift))
    sample_points = np.asarray([-4, -2, 0, np.min(centers), float(np.mean(centers)), np.max(centers), np.max(centers) + 1, np.max(centers) + 2])
    print('\nCorrectionLib round-trip (shifted ratio):')
    for point in sample_points:
        try:
            val_nom = loaded['zpt_reweight_nominal'].evaluate(float(point))
            val_up = loaded['zpt_reweight_up'].evaluate(float(point))
            val_dn = loaded['zpt_reweight_down'].evaluate(float(point))
            print(f'  x={point:.4f} -> nom={val_nom:.6f} up={val_up:.6f} dn={val_dn:.6f}')
        except Exception as e:
            print(f'  x={point:.4f} -> ERROR: {e}')


if __name__ == '__main__':
    main()


