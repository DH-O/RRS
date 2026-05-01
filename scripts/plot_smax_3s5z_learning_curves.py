import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re
LOG_DIR = Path('logs') / 'logs' / 'smax_3s5z_10seed'
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / 'figures'

def extract_curves_multiseed(filepath, config_filter=None):
    curves = {}
    current_seed = None
    current_config = None
    seed_data = {}
    with open(filepath, errors='replace') as f:
        for line in f:
            sm = re.search('seed[=\\s]+(\\d+)\\s+start', line)
            if sm:
                current_seed = int(sm.group(1))
                cfg = re.search('config([A-Z])', line)
                if cfg:
                    current_config = cfg.group(1)
                elif re.search('Phase\\s*1.*MAPPO-Lin|mappo_lin', line):
                    current_config = 'mappo_lin'
                elif re.search('Phase\\s*2.*Lagrangian|lagrangian', line):
                    current_config = 'lagrangian'
                else:
                    current_config = 'default'
                if config_filter and current_config != config_filter:
                    current_seed = None
                    continue
                if current_seed not in seed_data:
                    seed_data[current_seed] = ([], [])
                continue
            phase_m = re.search('Phase\\s*(\\d+).*?(\\w+)', line)
            if phase_m and 'seed' not in line:
                if 'MAPPO-Lin' in line or 'mappo_lin' in line:
                    current_config = 'mappo_lin'
                elif 'Lagrangian' in line or 'lagrangian' in line:
                    current_config = 'lagrangian'
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

def extract_curves_perseed(filepath):
    (steps, rets) = ([], [])
    with open(filepath, errors='replace') as f:
        for line in f:
            m = re.search('Env Step:\\s+(\\d+).*Returns:\\s+([\\d.]+)', line)
            if not m:
                m = re.search('step=\\s*(\\d+)\\s*\\|\\s*ret=\\s*([\\d.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                rets.append(float(m.group(2)))
    if len(steps) > 10:
        return (np.array(steps), np.array(rets))
    return None

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

def smooth(y, window=7):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, y[0]), y, np.full(window // 2, y[-1])])
    return np.convolve(padded, kernel, mode='valid')[:len(y)]
methods = [('RCB+RSQ (Ours)', {'sources': [('gpu2_hmin08_s0_2.log', None), ('gpu4_hmin08.log', None), ('gpu5_hmin08.log', None), ('gpu6_hmin08.log', None), ('gpu7_hmin08.log', None)], 'style': {'color': '#1f77b4', 'linestyle': '-', 'lw': 2.0, 'zorder': 10}}), ('MAPPO', {'sources': [('gpu4_ours_mappo.log', None), ('gpu5_ours_mappo.log', None), ('gpu4_mappo_fix.log', None), ('gpu2_mappo_remaining.log', None)], 'style': {'color': '#888888', 'linestyle': '--', 'lw': 1.3, 'zorder': 2}}), ('Linear', {'sources': [('gpu5_baselines_overnight.log', 'mappo_lin')], 'style': {'color': '#ff7f0e', 'linestyle': '-.', 'lw': 1.3, 'zorder': 3}}), ('RSQ-only', {'sources': [('gpu7_rsq_only.log', None)], 'style': {'color': '#d62728', 'linestyle': ':', 'lw': 1.3, 'zorder': 4}}), ('RCB-only', {'sources': [('gpu4_rcb_only.log', None)], 'style': {'color': '#2ca02c', 'linestyle': '--', 'lw': 1.3, 'zorder': 1}}), ('MAVEN', {'sources': [('gpu4_maven.log', None)], 'style': {'color': '#9467bd', 'linestyle': '-.', 'lw': 1.3, 'zorder': 5}}), ('Lagrangian', {'sources': [('gpu5_baselines_overnight.log', 'lagrangian')], 'style': {'color': '#17becf', 'linestyle': '--', 'lw': 1.3, 'zorder': 6}}), ('COIN', {'sources': [('gpu5_coin_3s5z.log', None)], 'style': {'color': '#bcbd22', 'linestyle': '-.', 'lw': 1.3, 'zorder': 8}})]
plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'mathtext.fontset': 'dejavusans', 'axes.linewidth': 0.8, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'grid.alpha': 0.3, 'legend.fontsize': 7, 'figure.dpi': 150, 'svg.fonttype': 'none'})
(fig, ax) = plt.subplots(1, 1, figsize=(3.5, 2.6))
window = 7
for (name, cfg) in methods:
    print(f'Loading {name}...')
    curves = {}
    for (filename, config_filter) in cfg['sources']:
        filepath = LOG_DIR / filename
        if not filepath.exists():
            print(f'  WARNING: {filepath} not found')
            continue
        if config_filter:
            new_curves = extract_curves_multiseed(filepath, config_filter)
        else:
            new_curves = extract_curves_multiseed(filepath)
        for (seed, data) in new_curves.items():
            if seed not in curves:
                curves[seed] = data
    print(f'  Found {len(curves)} seeds: {sorted(curves.keys())}')
    (x, mean, std) = compute_mean_std(curves)
    if x is None:
        print(f'  Skipping {name} (no data)')
        continue
    style = cfg['style']
    x_m = x / 1000000.0
    mean_s = smooth(mean, window)
    std_s = smooth(std, window)
    ax.plot(x_m, mean_s, label=name, color=style['color'], linestyle=style['linestyle'], linewidth=style['lw'], zorder=style['zorder'])
    ax.fill_between(x_m, mean_s - std_s, mean_s + std_s, alpha=0.12, color=style['color'], zorder=style['zorder'] - 0.5)
ax.set_xlabel('Environment Steps ($\\times 10^6$)')
ax.set_ylabel('Mean Episode Return')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)
ax.set_ylim(-0.02, 0.7)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0, ncol=3, framealpha=0.95, edgecolor='0.85', borderpad=0.3, labelspacing=0.3, handlelength=1.5, columnspacing=1.0, fontsize=7)
fig.tight_layout(pad=0.3)
fig.subplots_adjust(top=0.78)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_DIR / 'learning_curves_3s5z.pdf'
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
print(f'\nSaved to {out_path}')
png_path = out_path.with_suffix('.png')
fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02, dpi=200)
print(f'Preview: {png_path}')
svg_dir = OUTPUT_DIR / 'paper-final'
svg_dir.mkdir(parents=True, exist_ok=True)
svg_path = svg_dir / 'learning_curves_3s5z.svg'
fig.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0.02)
print(f'SVG: {svg_path}')
