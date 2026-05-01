import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re

def extract_curves(log_dir, pattern):
    curves = {}
    files = sorted(Path(log_dir).glob(pattern))
    if not files:
        return curves
    for f in files:
        seed_m = re.search('_s(\\d+)\\.log$', f.name)
        if seed_m:
            seed = int(seed_m.group(1))
            (steps, returns) = ([], [])
            with open(f) as fh:
                for line in fh:
                    m = re.search('Env Step:\\s+(\\d+).*Returns:\\s+([\\d.]+)', line)
                    if m:
                        steps.append(int(m.group(1)))
                        returns.append(float(m.group(2)))
            if len(steps) > 10:
                curves[seed] = (np.array(steps), np.array(returns))
        else:
            current_seed = None
            seed_data = {}
            with open(f) as fh:
                for line in fh:
                    sm = re.search('seed[=\\s]+(\\d+)\\s+start', line)
                    if sm:
                        current_seed = int(sm.group(1))
                        if current_seed not in seed_data:
                            seed_data[current_seed] = ([], [])
                        continue
                    m = re.search('Env Step:\\s+(\\d+).*Returns:\\s+([\\d.]+)', line)
                    if not m:
                        m = re.search('step=\\s*(\\d+)\\s*\\|\\s*ret=\\s*([\\d.]+)', line)
                    if m and current_seed is not None:
                        seed_data[current_seed][0].append(int(m.group(1)))
                        seed_data[current_seed][1].append(float(m.group(2)))
            for (seed, (steps, rets)) in seed_data.items():
                if len(steps) > 10:
                    curves[seed] = (np.array(steps), np.array(rets))
    return curves

def compute_mean_std(curves, num_points=200):
    if not curves:
        return (None, None, None)
    all_steps = [c[0] for c in curves.values()]
    max_step = min((s[-1] for s in all_steps))
    min_step = max((s[0] for s in all_steps))
    x = np.linspace(min_step, max_step, num_points)
    ys = []
    for (steps, rets) in curves.values():
        ys.append(np.interp(x, steps, rets))
    ys = np.array(ys)
    return (x, ys.mean(axis=0), ys.std(axis=0))

def smooth(y, window=5):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, y[0]), y, np.full(window // 2, y[-1])])
    return np.convolve(padded, kernel, mode='valid')[:len(y)]
plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'mathtext.fontset': 'dejavusans', 'axes.linewidth': 0.8, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'grid.alpha': 0.3, 'legend.fontsize': 7, 'figure.dpi': 150, 'svg.fonttype': 'none'})
log_dir = Path('logs') / 'logs' / 'overnight'
ablation_dir = Path('logs') / 'logs' / 'smax_ablation'
baseline_dir = Path('logs') / 'logs' / 'smax_baselines'
extra_dir = Path('logs') / 'logs' / 'smax_extra_seeds'
methods = [('RCB-only', [(ablation_dir, '27m-rcb-only_s*.log')], {'color': '#2ca02c', 'linestyle': '--', 'lw': 1.3, 'zorder': 1}), ('MAPPO', [(log_dir, 'smax_mappo_noexplore_s*.log'), (extra_dir, 'gpu0_mappo_seeds5-9.log')], {'color': '#888888', 'linestyle': '--', 'lw': 1.3, 'zorder': 2}), ('Linear', [(log_dir, 'smax_mappo_lin_fair_v3_s*.log'), (extra_dir, 'gpu1_mappo-lin_seeds5-9.log')], {'color': '#ff7f0e', 'linestyle': '-.', 'lw': 1.3, 'zorder': 3}), ('RSQ-only', [(ablation_dir, '27m-rsq-only_s*.log')], {'color': '#d62728', 'linestyle': ':', 'lw': 1.3, 'zorder': 4}), ('RCB+RSQ (Ours)', [(log_dir, 'smax_ours_v2a_s*.log'), (extra_dir, 'gpu6_ours_seeds5-9.log')], {'color': '#1f77b4', 'linestyle': '-', 'lw': 2.0, 'zorder': 10}), ('MAVEN', [(baseline_dir, 'maven_smax27m.log')], {'color': '#9467bd', 'linestyle': '-.', 'lw': 1.3, 'zorder': 5}), ('Lagrangian', [(baseline_dir, 'lagrangian_smax27m.log')], {'color': '#17becf', 'linestyle': '--', 'lw': 1.3, 'zorder': 6}), ('COIN', [(baseline_dir, 'coin_smax27m.log')], {'color': '#bcbd22', 'linestyle': '-.', 'lw': 1.3, 'zorder': 8})]
(fig, ax) = plt.subplots(1, 1, figsize=(3.5, 2.6))
window = 7
for (name, sources, style) in methods:
    print(f'Loading {name}...')
    curves = {}
    for (ldir, pattern) in sources:
        new_curves = extract_curves(ldir, pattern)
        curves.update(new_curves)
    print(f'  Found {len(curves)} seeds')
    (x, mean, std) = compute_mean_std(curves)
    if x is None:
        print(f'  Skipping {name} (no data)')
        continue
    x_m = x / 1000000.0
    mean_s = smooth(mean, window)
    std_s = smooth(std, window)
    ax.plot(x_m, mean_s, label=name, color=style['color'], linestyle=style['linestyle'], linewidth=style['lw'], zorder=style['zorder'])
    ax.fill_between(x_m, mean_s - std_s, mean_s + std_s, alpha=0.12, color=style['color'], zorder=style['zorder'] - 0.5)
ax.set_xlabel('Environment Steps ($\\times 10^6$)')
ax.set_ylabel('Mean Episode Return')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)
ax.set_ylim(-0.02, 0.52)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0, ncol=3, framealpha=0.95, edgecolor='0.85', borderpad=0.3, labelspacing=0.3, handlelength=1.5, columnspacing=1.0, fontsize=7)
fig.tight_layout(pad=0.3)
fig.subplots_adjust(top=0.78)
out_path = Path(__file__).resolve().parent.parent.parent / 'figures' / 'learning_curves_smax.pdf'
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
print(f'\nSaved to {out_path}')
png_path = out_path.with_suffix('.png')
fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02, dpi=200)
print(f'Preview: {png_path}')
svg_dir = out_path.parent / 'paper-final'
svg_dir.mkdir(parents=True, exist_ok=True)
svg_path = svg_dir / 'learning_curves_smax.svg'
fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.02)
print(f'SVG: {svg_path}')
