import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
EXP_DIR = BASE / 'logs' / 'experiments'
MASTER_DIR = BASE / 'logs' / 'master' / 'seeds'
TAG_SWEEP = BASE / 'logs' / 'tag_sweep'
RSQ_ONLY_DIR = BASE / 'logs' / 'rsq_only_mpe'
OUTPUT_DIR = BASE / 'figures'
METHOD_STYLES = {'Linear': {'color': '#888888', 'linestyle': '--', 'label': 'Linear', 'zorder': 1, 'lw': 1.3}, 'COIN': {'color': '#bcbd22', 'linestyle': '-.', 'label': 'COIN', 'zorder': 3, 'lw': 1.3}, 'Lagrangian': {'color': '#17becf', 'linestyle': '--', 'label': 'Lagrangian', 'zorder': 4, 'lw': 1.3}, 'MAPPO': {'color': '#ff7f0e', 'linestyle': ':', 'label': 'MAPPO', 'zorder': 5, 'lw': 1.3}, 'IPPO': {'color': '#ff7f0e', 'linestyle': ':', 'label': 'IPPO', 'zorder': 5, 'lw': 1.3}, 'MAVEN': {'color': '#9467bd', 'linestyle': '-.', 'label': 'MAVEN', 'zorder': 6, 'lw': 1.3}, 'RCB-only': {'color': '#2ca02c', 'linestyle': '--', 'label': 'RCB-only', 'zorder': 8, 'lw': 1.3}, 'RSQ-only': {'color': '#d62728', 'linestyle': ':', 'label': 'RSQ-only', 'zorder': 9, 'lw': 1.3}, 'RCB+RSQ': {'color': '#1f77b4', 'linestyle': '-', 'label': 'RCB+RSQ (Ours)', 'zorder': 10, 'lw': 2.0}}
METHOD_ORDER_MPE = ['Linear', 'COIN', 'Lagrangian', 'MAPPO', 'MAVEN', 'RCB-only', 'RSQ-only', 'RCB+RSQ']
METHOD_ORDER_MABRAX = ['Linear', 'COIN', 'Lagrangian', 'IPPO', 'MAVEN', 'RCB-only', 'RCB+RSQ']
LOG_PATTERNS = {('Linear', 'corridor'): [str(EXP_DIR / 'eval_MAPPO-Linear_corridor_s*.log')], ('COIN', 'corridor'): [str(EXP_DIR / 'eval_COIN_corridor_s*.log')], ('Lagrangian', 'corridor'): [str(EXP_DIR / 'eval_Lagrangian_corridor_s*.log')], ('MAPPO', 'corridor'): [str(BASE / 'logs' / 'overnight' / 'mappo_corridor_s*.log')], ('MAVEN', 'corridor'): [str(EXP_DIR / 'maven_corridor_s*.log')], ('RCB-only', 'corridor'): [str(EXP_DIR / 'rerun_RCB-only_corridor_s*.log')], ('RSQ-only', 'corridor'): [str(RSQ_ONLY_DIR / 'corridor_s*.log')], ('RCB+RSQ', 'corridor'): [str(MASTER_DIR / f'rcb_lsq_seed{i}.log') for i in range(10)], ('Linear', 'tag'): [str(EXP_DIR / 'eval_MAPPO-Linear_tag_s*.log')], ('COIN', 'tag'): [str(EXP_DIR / 'eval_COIN_tag_s*.log')], ('Lagrangian', 'tag'): [str(EXP_DIR / 'eval_Lagrangian_tag_s*.log')], ('MAPPO', 'tag'): [str(BASE / 'logs' / 'overnight' / 'mappo_tag_gpu2_s*.log')], ('MAVEN', 'tag'): [str(EXP_DIR / 'maven_tag_s*.log')], ('RCB-only', 'tag'): [str(EXP_DIR / 'rerun_RCB-only_tag_s*.log')], ('RSQ-only', 'tag'): [str(RSQ_ONLY_DIR / 'tag_s*.log')], ('RCB+RSQ', 'tag'): [str(BASE / 'logs' / 'tag_ours_rerun' / f'tag_ours_s{i}.log') for i in range(10)], ('Linear', 'ant_4x2'): [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / f'mappo_lin_s{i}.log') for i in range(5)] + [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / 'mappo_lin_s5_9_multi.log')], ('COIN', 'ant_4x2'): [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / 'coin_qr_s0_4_multi.log'), str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / 'coin_qr_s5_9_multi.log')], ('Lagrangian', 'ant_4x2'): [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / f'lagrangian_s{i}.log') for i in range(5)] + [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / 'lagrangian_s5_9_multi.log')], ('IPPO', 'ant_4x2'): [str(BASE / '.' / 'paper_used_results' / 'logs' / 'ant_4x2' / 'ippo_rerun' / f'ippo_s{i}.log') for i in range(10)], ('MAVEN', 'ant_4x2'): [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / 'maven_qr_s0_4_multi.log'), str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / 'maven_qr_s5_9_multi.log')], ('RCB-only', 'ant_4x2'): [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / f'rcb_only_s{i}.log') for i in range(5)] + [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / 'rcb_only_s5_9_multi.log')], ('RCB+RSQ', 'ant_4x2'): [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / f'ours_s{i}.log') for i in range(5)] + [str(BASE / '.' / 'exp_final_paper' / 'ant_4x2' / 'ours_s5_9_multi.log')]}
ENV_TITLES = {'corridor': '(a) MPE-corridor (8 agents)', 'tag': '(a) MPE-Tag (6 predators, 2 prey)', 'ant_4x2': '(b) MABrax ant_4x2 (4 agents, 2 actuators)'}
SMOOTHING_WINDOW = 5
N_INTERP_POINTS = 200
RETURN_PATTERNS = [re.compile('Env Step:\\s*(\\d+)\\s*\\|\\s*Returns:\\s*([-\\d.e+]+)'), re.compile('step=\\s*(\\d+)\\s*\\|\\s*ret=\\s*([-\\d.e+]+)'), re.compile('ts=\\s*([\\d,]+)\\s*\\|\\s*ep_return=([-\\d.e+]+)')]

def parse_log(filepath, env_filter=None):
    data = []
    active = True
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            if env_filter:
                lo = line.lower()
                if 'environment:' in lo or ('seed=' in lo and '===' in line):
                    if env_filter.lower() in lo:
                        active = True
                    elif 'environment:' in lo:
                        active = False
                if not active:
                    continue
            for pat in RETURN_PATTERNS:
                m = pat.search(line)
                if m:
                    try:
                        data.append((int(m.group(1).replace(',', '')), float(m.group(2))))
                    except (ValueError, IndexError):
                        pass
                    break
    data.sort(key=lambda x: x[0])
    return data

def smooth(values, window=SMOOTHING_WINDOW):
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window // 2, values[0]), values, np.full(window // 2, values[-1])])
    return np.convolve(padded, kernel, mode='valid')[:len(values)]

def collect_all():
    all_curves = {}
    for ((method, env), patterns) in LOG_PATTERNS.items():
        files = []
        for pat in patterns:
            files.extend(sorted(glob.glob(pat)))
        curves = []
        max_step = 50000000.0 if env == 'ant_4x2' else 30000000.0
        for f in files:
            is_multi = '_multi' in Path(f).name or '_qr_' in Path(f).name
            env_filter = env if is_multi else None
            data = parse_log(f, env_filter=env_filter)
            if len(data) >= 10:
                steps = np.array([d[0] for d in data])
                rets = np.array([d[1] for d in data])
                mask = steps <= max_step
                curves.append((steps[mask], rets[mask]))
        if curves:
            all_curves[method, env] = curves
            print(f'  {method:>15s} / {env:>8s}: {len(curves)} seeds, {len(curves[0][0])} points each')
        else:
            print(f'  {method:>15s} / {env:>8s}: NO DATA')
    return all_curves

def plot_env(ax, env, all_curves):
    method_order = METHOD_ORDER_MABRAX if env == 'ant_4x2' else METHOD_ORDER_MPE
    for method in method_order:
        key = (method, env)
        if key not in all_curves:
            continue
        curves = all_curves[key]
        style = METHOD_STYLES[method]
        min_step = max((c[0][0] for c in curves))
        max_step = min((c[0][-1] for c in curves))
        common_steps = np.linspace(min_step, max_step, N_INTERP_POINTS)
        interpolated = []
        for (steps, rets) in curves:
            interp = np.interp(common_steps, steps, rets)
            interpolated.append(interp)
        matrix = np.array(interpolated)
        mean = smooth(np.mean(matrix, axis=0))
        std = smooth(np.std(matrix, axis=0))
        x = common_steps / 1000000.0
        ax.plot(x, mean, color=style['color'], linestyle=style['linestyle'], linewidth=style['lw'], label=style['label'], zorder=style['zorder'])
        ax.fill_between(x, mean - std, mean + std, color=style['color'], alpha=0.12, zorder=style['zorder'] - 0.5)
    ax.set_xlabel('Environment Steps ($\\times 10^6$)')
    ax.set_ylabel('Mean Episode Return')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), mode='expand', borderaxespad=0, ncol=3, framealpha=0.95, edgecolor='0.85', borderpad=0.3, labelspacing=0.3, handlelength=1.5, columnspacing=1.0, fontsize=7)
    ax.grid(True, alpha=0.3)
    xlim_map = {'corridor': 30, 'tag': 30, 'ant_4x2': 50}
    ax.set_xlim(0, xlim_map.get(env, 30))

def main():
    print('Collecting learning curve data...')
    all_curves = collect_all()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'mathtext.fontset': 'dejavusans', 'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10, 'axes.linewidth': 0.8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7, 'grid.alpha': 0.3, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'svg.fonttype': 'none'})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for env in ['corridor', 'tag', 'ant_4x2']:
        (fig, ax) = plt.subplots(figsize=(3.5, 2.6))
        plot_env(ax, env, all_curves)
        fig.tight_layout(pad=0.3)
        fig.subplots_adjust(top=0.78)
        fname_map = {'ant_4x2': 'ant'}
        fname = fname_map.get(env, env)
        out_path = OUTPUT_DIR / f'learning_curves_{fname}.pdf'
        fig.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
        print(f'\nSaved: {out_path}')
        out_png = OUTPUT_DIR / f'learning_curves_{fname}.png'
        fig.savefig(out_png, format='png', dpi=200, bbox_inches='tight', pad_inches=0.02)
        print(f'Saved: {out_png}')
        svg_dir = OUTPUT_DIR / 'paper-final'
        svg_dir.mkdir(parents=True, exist_ok=True)
        out_svg = svg_dir / f'learning_curves_{fname}.svg'
        fig.savefig(out_svg, format='svg', bbox_inches='tight', pad_inches=0.02)
        print(f'Saved: {out_svg}')
        plt.close(fig)
    print('\nDone!')
if __name__ == '__main__':
    main()
