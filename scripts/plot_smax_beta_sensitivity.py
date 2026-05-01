import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'mathtext.fontset': 'dejavusans', 'font.size': 9, 'axes.labelsize': 10, 'axes.linewidth': 0.8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7, 'figure.dpi': 150, 'text.usetex': False, 'svg.fonttype': 'none'})
betas = [0.1, 0.15, 0.2, 0.3, 0.5]
means = [0.449, 0.44, 0.366, 0.214, 0.096]
stds = [0.003, 0.009, 0.123, 0.11, 0.022]
failures = [0, 0, 20, 80, 100]
ours_mean = 0.447
ours_std = 0.005
ours_fail = 0
mappo_mean = 0.43
mappo_std = 0.015
(fig, ax1) = plt.subplots(1, 1, figsize=(3.5, 2.6))
x = np.arange(len(betas))
width = 0.55
colors = []
for f in failures:
    if f == 0:
        colors.append('#4DBEEE')
    elif f <= 20:
        colors.append('#EDB120')
    else:
        colors.append('#D95319')
bars = ax1.bar(x, means, width, yerr=stds, color=colors, edgecolor='black', linewidth=0.5, capsize=3, zorder=3, alpha=0.85)
ax1.text(2.35, 0.48, '20%', ha='left', va='bottom', fontsize=6, color='#A0200A', fontweight='bold')
ax1.text(2.35, 0.455, 'fail', ha='left', va='bottom', fontsize=6, color='#A0200A', fontweight='bold')
ax1.text(3, 0.34, '80%\nfail', ha='center', va='bottom', fontsize=6, color='#A0200A', fontweight='bold')
ax1.text(4.35, 0.1, '100%\nfail', ha='left', va='bottom', fontsize=6, color='#A0200A', fontweight='bold')
ax1.axhspan(ours_mean - ours_std, ours_mean + ours_std, color='#77AC30', alpha=0.25, zorder=1)
ax1.axhline(y=ours_mean, color='#77AC30', linewidth=2, linestyle='-', label=f'Ours ($\\beta_{{\\min}}$=0.3+RSQ)', zorder=2)
ax1.axhline(y=mappo_mean, color='gray', linewidth=1.2, linestyle='--', label=f'MAPPO (no expl.)', zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels([f'{b}' for b in betas])
ax1.set_xlabel('Fixed $\\beta_{\\min}$ (without RSQ)')
ax1.set_ylabel('Mean Episode Return')
ax1.set_ylim(-0.02, 0.52)
ax1.set_xlim(-0.6, len(betas) - 0.2)
ax1.legend(loc='lower left', framealpha=0.9, edgecolor='0.85', borderpad=0.3, labelspacing=0.25, handlelength=1.5)
ax1.yaxis.grid(True, alpha=0.3, zorder=0)
ax1.set_axisbelow(True)
fig.tight_layout(pad=0.3)
plt.savefig('figures/smax_beta_sensitivity.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig('figures/smax_beta_sensitivity.png', bbox_inches='tight', pad_inches=0.02, dpi=200)
import os
os.makedirs('figures/paper-final', exist_ok=True)
plt.savefig('figures/paper-final/smax_beta_sensitivity.svg', bbox_inches='tight', pad_inches=0.02)
plt.close()
print('Saved PDF, PNG, SVG')
