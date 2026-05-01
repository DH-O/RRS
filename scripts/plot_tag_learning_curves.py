import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / 'scripts' / 'logs' / 'experiments'
TAG_SWEEP_DIR = PROJECT_ROOT / 'scripts' / 'logs' / 'tag_sweep'
OUTPUT_DIR = PROJECT_ROOT / 'paper_ijcas_submit' / 'figures'

def parse_learning_curve(log_path, patterns=None):
    if patterns is None:
        patterns = [re.compile('Env Step:\\s*([\\d]+)\\s*\\|\\s*Returns:\\s*([-\\d.]+)'), re.compile('step=([\\d]+)\\s*\\|\\s*ret=([-\\d.]+)')]
    (steps, returns) = ([], [])
    try:
        with open(log_path, 'r', errors='replace') as f:
            for line in f:
                for p in patterns:
                    m = p.search(line)
                    if m:
                        steps.append(int(m.group(1)))
                        returns.append(float(m.group(2)))
                        break
    except Exception:
        pass
    return (np.array(steps), np.array(returns))

def load_method_curves(log_dir, pattern, num_seeds=10):
    all_curves = []
    for seed in range(num_seeds):
        log_path = log_dir / pattern.format(seed=seed)
        if log_path.exists():
            (steps, returns) = parse_learning_curve(log_path)
            if len(steps) > 0:
                all_curves.append((steps, returns))
    return all_curves

def interpolate_curves(curves, num_points=100):
    if not curves:
        return (None, None, None)
    max_step = min((curve[0][-1] for curve in curves))
    min_step = max((curve[0][0] for curve in curves))
    common_steps = np.linspace(min_step, max_step, num_points)
    interpolated = []
    for (steps, returns) in curves:
        interp = np.interp(common_steps, steps, returns)
        interpolated.append(interp)
    interpolated = np.array(interpolated)
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0)
    return (common_steps, mean, std)
methods = {'RCB+RSQ (Ours)': {'color': '#E53935', 'linewidth': 2.5, 'zorder': 10}, 'MAVEN': {'dir': EXPERIMENTS_DIR, 'pattern': 'maven_tag_s{seed}.log', 'color': '#9C27B0', 'linewidth': 1.5, 'zorder': 5}, 'MAPPO-Linear': {'dir': EXPERIMENTS_DIR, 'pattern': 'eval_MAPPO-Linear_tag_s{seed}.log', 'color': '#4CAF50', 'linewidth': 1.5, 'zorder': 5}, 'IPPO': {'dir': EXPERIMENTS_DIR, 'pattern': 'eval_IPPO_tag_s{seed}.log', 'color': '#FF9800', 'linewidth': 1.5, 'zorder': 5}, 'COIN': {'dir': EXPERIMENTS_DIR, 'pattern': 'eval_COIN_tag_s{seed}.log', 'color': '#607D8B', 'linewidth': 1.0, 'zorder': 3}, 'Lagrangian': {'dir': EXPERIMENTS_DIR, 'pattern': 'eval_Lagrangian_tag_s{seed}.log', 'color': '#795548', 'linewidth': 1.0, 'zorder': 3}}
(fig, ax) = plt.subplots(1, 1, figsize=(7, 4.5))
for (method_name, info) in methods.items():
    if method_name == 'RCB+RSQ (Ours)':
        curves = []
        for seed in range(15):
            log_path = TAG_SWEEP_DIR / f'maven_bmin015_ent05_s{seed}.log'
            if log_path.exists():
                (steps, returns) = parse_learning_curve(log_path)
                if len(steps) > 0:
                    curves.append((steps, returns))
    else:
        curves = load_method_curves(info['dir'], info['pattern'])
    if not curves:
        print(f'  {method_name}: No data found')
        continue
    (common_steps, mean, std) = interpolate_curves(curves)
    if common_steps is None:
        continue
    print(f'  {method_name}: {len(curves)} seeds, final_mean={mean[-1]:.1f}')
    ax.plot(common_steps / 1000000.0, mean, label=method_name, color=info['color'], linewidth=info['linewidth'], zorder=info['zorder'])
    ax.fill_between(common_steps / 1000000.0, mean - std, mean + std, alpha=0.15, color=info['color'], zorder=info['zorder'] - 1)
ax.set_xlabel('Environment Steps (M)', fontsize=11)
ax.set_ylabel('Team Return', fontsize=11)
ax.set_title('MPE-tag 6v2 (6 adversaries)', fontsize=12)
ax.legend(fontsize=8, loc='lower right', ncol=2)
ax.grid(alpha=0.3)
ax.set_xlim(0, 30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'learning_curves_tag.pdf', bbox_inches='tight', dpi=300)
plt.savefig(OUTPUT_DIR / 'learning_curves_tag.png', bbox_inches='tight', dpi=300)
print(f"\nSaved to {OUTPUT_DIR / 'learning_curves_tag.pdf'}")
