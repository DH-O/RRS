import json
import os
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'mathtext.fontset': 'dejavusans', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 6.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.4, 'lines.linewidth': 1.0, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.framealpha': 0.85, 'legend.edgecolor': '0.8', 'svg.fonttype': 'none'}
AGENT_COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, 'r') as fh:
        for (lineno, line) in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f'Warning: skipping malformed line {lineno}: {exc}')
    if not records:
        raise ValueError(f'No valid JSON records found in {path}')
    return records

def extract_arrays(records: List[Dict]) -> Dict[str, np.ndarray]:
    T = len(records)
    n_agents = None
    for rec in records:
        if 'rsq' in rec and isinstance(rec['rsq'], list):
            n_agents = len(rec['rsq'])
            break
    if n_agents is None:
        raise ValueError('No per-agent RSQ data found in JSONL log.')
    result = {'env_steps': np.array([r.get('env_step', i) for (i, r) in enumerate(records)]), 'mean_return': np.array([r.get('mean_return', 0.0) for r in records]), 'rcb_beta': np.array([r.get('rcb_beta', 0.5) for r in records]), 'n_agents': n_agents}
    per_agent_keys = {'rsq': ('rsq', np.nan), 'g_individual': ('g_individual', 1.0)}
    for (out_key, (json_key, default_val)) in per_agent_keys.items():
        arr = np.full((T, n_agents), default_val, dtype=np.float64)
        for (t, rec) in enumerate(records):
            vals = rec.get(json_key)
            if vals is not None and isinstance(vals, list):
                arr[t, :len(vals)] = vals[:n_agents]
        result[out_key] = arr
    return result

def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    alpha = 2.0 / (window + 1)
    if arr.ndim == 1:
        out = np.empty_like(arr)
        out[0] = arr[0]
        for t in range(1, len(arr)):
            out[t] = alpha * arr[t] + (1 - alpha) * out[t - 1]
        return out
    else:
        out = np.empty_like(arr)
        out[0] = arr[0]
        for t in range(1, arr.shape[0]):
            out[t] = alpha * arr[t] + (1 - alpha) * out[t - 1]
        return out

def plot_combined(corridor_path: str, tag_path: str, output_path: str, smooth_window: int=10) -> None:
    plt.rcParams.update(STYLE)
    corridor_data = extract_arrays(load_jsonl(corridor_path))
    tag_data = extract_arrays(load_jsonl(tag_path))
    sw = smooth_window
    datasets = {'corridor': {'x': corridor_data['env_steps'] / 1000000.0, 'rsq': smooth(corridor_data['rsq'], sw), 'g': smooth(corridor_data['g_individual'], sw), 'beta': smooth(corridor_data['rcb_beta'], sw), 'ret': smooth(corridor_data['mean_return'], sw), 'n_agents': corridor_data['n_agents'], 'title': 'MPE-Corridor (8 agents)'}, 'tag': {'x': tag_data['env_steps'] / 1000000.0, 'rsq': smooth(tag_data['rsq'], sw), 'g': smooth(tag_data['g_individual'], sw), 'beta': smooth(tag_data['rcb_beta'], sw), 'ret': smooth(tag_data['mean_return'], sw), 'n_agents': tag_data['n_agents'], 'title': 'MPE-Tag 6v2 (6 agents)'}}
    (fig, axes) = plt.subplots(3, 2, figsize=(7.0, 5.2), gridspec_kw={'hspace': 0.5, 'wspace': 0.38})
    env_order = ['corridor', 'tag']
    panel_captions = [['(a) ' + 'MPE-Corridor (8 agents)', '(b) ' + 'MPE-Tag 6v2 (6 agents)'], ['(c) Modulation weight $h_i$', '(d) Modulation weight $h_i$'], ['(e) $\\beta$ and Team Return', '(f) $\\beta$ and Team Return']]
    for (col_idx, env_key) in enumerate(env_order):
        d = datasets[env_key]
        x = d['x']
        n_agents = d['n_agents']
        colors = AGENT_COLORS[:n_agents]
        ax = axes[0, col_idx]
        for i in range(n_agents):
            ax.plot(x, d['rsq'][:, i], color=colors[i], alpha=0.85, label=f'Agent {i}', linewidth=0.9)
        ax.set_ylabel('RSQ$_i$')
        ax.set_xlabel(panel_captions[0][col_idx], fontsize=8)
        y_lo = max(0.0, np.nanmin(d['rsq']) - 0.02)
        y_hi = min(1.05, np.nanmax(d['rsq']) + 0.03)
        ax.set_ylim([y_lo, y_hi])
        ax = axes[1, col_idx]
        for i in range(n_agents):
            ax.plot(x, d['g'][:, i], color=colors[i], alpha=0.85, label=f'Agent {i}', linewidth=0.9)
        ax.set_ylabel('$h_i$')
        ax.set_xlabel(panel_captions[1][col_idx], fontsize=8)
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax_beta = axes[2, col_idx]
        ax_beta.spines['right'].set_visible(False)
        color_beta = '#d62728'
        color_return = '#1f77b4'
        ln1 = ax_beta.plot(x, d['beta'], color=color_beta, linewidth=1.1, label='$\\beta$ (RCB)')
        ax_beta.set_ylabel('')
        ax_beta.text(0.0, 1.02, '$\\beta$ (RCB)', transform=ax_beta.transAxes, fontsize=7, color=color_beta, ha='left', va='bottom')
        ax_beta.tick_params(axis='y', labelcolor=color_beta)
        beta_min = np.nanmin(d['beta'])
        beta_max = np.nanmax(d['beta'])
        beta_margin = max(0.02, (beta_max - beta_min) * 0.15)
        ax_beta.set_ylim([max(0, beta_min - beta_margin), beta_max + beta_margin])
        ax_beta.set_xlabel('Environment Steps (M)\n' + panel_captions[2][col_idx], fontsize=8)
        ax_ret = ax_beta.twinx()
        ax_ret.spines['top'].set_visible(False)
        ax_ret.spines['right'].set_visible(True)
        ax_ret.spines['right'].set_linewidth(0.6)
        ax_ret.grid(False)
        ln2 = ax_ret.plot(x, d['ret'], color=color_return, linewidth=1.1, linestyle='--', label='Team Return')
        ax_ret.set_ylabel('')
        ax_ret.text(1.0, 1.02, 'Team Return', transform=ax_ret.transAxes, fontsize=7, color=color_return, ha='right', va='bottom')
        ax_ret.tick_params(axis='y', labelcolor=color_return)
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    (handles_agents, labels_agents) = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles_agents, labels_agents, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=len(handles_agents), fontsize=7.5, handlelength=1.5, columnspacing=1.0, frameon=True, edgecolor='0.8')
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path)
    print(f'Figure saved to {output_path}')
    if output_path.endswith('.pdf'):
        svg_dir = os.path.join(os.path.dirname(output_path), 'paper-final')
        os.makedirs(svg_dir, exist_ok=True)
        svg_path = os.path.join(svg_dir, os.path.basename(output_path).replace('.pdf', '.svg'))
        fig.savefig(svg_path, format='svg')
        print(f'SVG saved to {svg_path}')
    if output_path.endswith('.pdf'):
        png_path = output_path.replace('.pdf', '.png')
        fig.savefig(png_path)
        print(f'PNG companion saved to {png_path}')
    plt.close(fig)
    for env_key in env_order:
        d = datasets[env_key]
        print(f"\n--- {d['title']} ---")
        print(f"  Steps: {d['x'][0]:.1f}M -> {d['x'][-1]:.1f}M ({len(d['x'])} points)")
        print(f"  Final RSQ range: [{np.nanmin(d['rsq'][-1]):.4f}, {np.nanmax(d['rsq'][-1]):.4f}]")
        print(f"  Final g_i range: [{np.nanmin(d['g'][-1]):.4f}, {np.nanmax(d['g'][-1]):.4f}]")
        print(f"  Final beta:  {d['beta'][-1]:.4f}")
        print(f"  Final return: {d['ret'][-1]:.1f}")
if __name__ == '__main__':
    BASE = '.'
    corridor_path = os.path.join(BASE, 'scripts/fig1_data/dynamics/corridor_rsq_dynamics.jsonl')
    tag_path = os.path.join(BASE, 'scripts/fig1_data/dynamics/tag_rsq_dynamics_new.jsonl')
    output_path = os.path.join(BASE, 'figures/rsq_dynamics_combined.pdf')
    plot_combined(corridor_path=corridor_path, tag_path=tag_path, output_path=output_path, smooth_window=10)
