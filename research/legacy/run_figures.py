"""Generate 5 NeurIPS publication figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTDIR = '/home/kec23008/docker-sys/dllm/research/figures'
os.makedirs(OUTDIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'figure.dpi': 150,
    'font.family': 'serif',
})
C_BF16 = '#d62728'  # red
C_FP32 = '#2ca02c'  # green
C_ACCENT = '#1f77b4'  # blue

# ===== Figure 1: GEMM Batch Variance =====
fig, ax = plt.subplots(figsize=(7, 4.5))
M = [1, 16, 32, 64, 256, 1024]
bf16_vals = [0, 0, 0.5, 1.0, 1.0, 1.0]
fp32_vals = [0, 0, 0.5, 0.5, 0.5, 0.5]
ax.plot(range(len(M)), bf16_vals, 'o-', color=C_BF16, linewidth=2, markersize=8, label='BF16 (default)')
ax.plot(range(len(M)), fp32_vals, 's--', color=C_FP32, linewidth=2, markersize=8, label='FP32 accum')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='BF16 rounding quantum')
ax.set_xticks(range(len(M)))
ax.set_xticklabels([str(m) for m in M])
ax.set_xlabel('Batch Size M')
ax.set_ylabel('Max Element-wise Diff (BF16 ULP scale)')
ax.set_title('GEMM Batch Variance: Q-Projection (K=4096, N=4096)')
ax.legend(loc='upper left')
ax.set_ylim(-0.05, 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/figure1_gemm_variance.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/figure1_gemm_variance.png', bbox_inches='tight', dpi=300)
plt.close()
print("Figure 1 done")

# ===== Figure 2: 1000-Run Hash Distribution =====
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

# BF16
axes[0].bar(['Hash A', 'Hash B'], [900, 100], color=[C_BF16, '#ff9896'], edgecolor='black', linewidth=0.5)
axes[0].set_title('BF16 (default)', fontsize=14)
axes[0].set_ylabel('Number of Runs')
axes[0].text(0, 920, '900', ha='center', fontsize=12, fontweight='bold')
axes[0].text(1, 120, '100', ha='center', fontsize=12, fontweight='bold')
axes[0].annotate('2 unique outputs', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=11, color=C_BF16, fontweight='bold')

# FP32
axes[1].bar(['Hash A'], [1000], color=[C_FP32], edgecolor='black', linewidth=0.5)
axes[1].set_title('FP32 accum', fontsize=14)
axes[1].text(0, 1020, '1000', ha='center', fontsize=12, fontweight='bold')
axes[1].annotate('1 unique output', xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', fontsize=11, color=C_FP32, fontweight='bold')
axes[1].set_xlim(-0.6, 1.4)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1100)

fig.suptitle('Generation Determinism: 1000 Runs, 10 Batch Sizes (Llama-3.1-8B)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/figure2_hash_distribution.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/figure2_hash_distribution.png', bbox_inches='tight', dpi=300)
plt.close()
print("Figure 2 done")

# ===== Figure 3: Op-Level Effectiveness Heatmap =====
fig, ax = plt.subplots(figsize=(8, 4.5))
ops = ['GEMM\n(Q-proj)', 'GEMM\n(gate)', 'RMSNorm', 'Softmax', 'Attention\n(fixed split)', 'Attention\n(varying split)']
# 0=already det, 1=FP32 fixes, -1=FP32 doesn't help
# columns: bs=1, bs=8, bs=32, bs=128, bs=1024
data = np.array([
    [0, 1, 1, 1, 1],    # GEMM Q-proj
    [0, 1, 1, 1, 1],    # GEMM gate
    [0, 1, 1, 1, 1],    # RMSNorm
    [0, 0, 0, 0, 0],    # Softmax (already det)
    [0, 0, 0, 0, 0],    # Attention fixed split (already det)
    [0, -1, -1, -1, -1], # Attention varying split
])
bs_labels = ['1', '8', '32', '128', '1024']

cmap = matplotlib.colors.ListedColormap(['#d62728', '#f0f0f0', '#2ca02c'])
norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

ax.set_xticks(range(len(bs_labels)))
ax.set_xticklabels(bs_labels)
ax.set_yticks(range(len(ops)))
ax.set_yticklabels(ops)
ax.set_xlabel('Batch Size')
ax.set_title('FP32 Accumulation Effectiveness by Operation')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2ca02c', label='FP32 fixes it'),
    mpatches.Patch(facecolor='#f0f0f0', edgecolor='gray', label='Already deterministic'),
    mpatches.Patch(facecolor='#d62728', label='FP32 insufficient'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Add text annotations
for i in range(len(ops)):
    for j in range(len(bs_labels)):
        v = data[i, j]
        txt = {0: '-', 1: 'Y', -1: 'X'}[v]
        color = {0: 'gray', 1: 'white', -1: 'white'}[v]
        ax.text(j, i, txt, ha='center', va='center', fontsize=14, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTDIR}/figure3_op_heatmap.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/figure3_op_heatmap.png', bbox_inches='tight', dpi=300)
plt.close()
print("Figure 3 done")

# ===== Figure 4: MoE Amplification Chain =====
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

boxes = [
    (0.5, 1.5, 'Gate\nGEMM', '#1f77b4'),
    (3.0, 1.5, 'Softmax', '#ff7f0e'),
    (5.5, 1.5, 'Top-k', '#d62728'),
    (8.0, 1.5, 'Expert\nSelection', '#9467bd'),
]
for x, y, txt, color in boxes:
    rect = mpatches.FancyBboxPatch((x-0.7, y-0.5), 1.4, 1.0,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color, alpha=0.8, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y, txt, ha='center', va='center', fontsize=12, color='white', fontweight='bold')

# Arrows and error labels
arrow_props = dict(arrowstyle='->', lw=2, color='black')
errors = [('0.5 ULP\ndiff', 1.2, 2.25), ('amplified\nnear boundary', 3.7, 2.25),
          ('hard\nthreshold', 6.2, 2.25)]
for i, (x1, x2) in enumerate([(1.2, 2.3), (3.7, 4.8), (6.2, 7.3)]):
    ax.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5), arrowprops=arrow_props)
    ax.text((x1+x2)/2, 2.3, errors[i][0], ha='center', va='center', fontsize=10,
            color='#555', style='italic')

# Final annotation
ax.text(8.0, 0.5, 'EXPERT FLIP', ha='center', va='center', fontsize=13,
        color='#d62728', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#ffcccc', edgecolor='#d62728'))

ax.set_title('MoE Routing: Error Amplification Chain', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/figure4_moe_chain.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/figure4_moe_chain.png', bbox_inches='tight', dpi=300)
plt.close()
print("Figure 4 done")

# ===== Figure 5: Dense vs MoE Comparison =====
fig, ax = plt.subplots(figsize=(7, 4.5))
x_pos = [0, 1, 3, 4]
heights = [2, 1, 3, 3]
colors = [C_BF16, C_FP32, C_BF16, C_FP32]
labels = ['BF16', 'FP32', 'BF16', 'FP32']
bars = ax.bar(x_pos, heights, color=colors, edgecolor='black', linewidth=0.5, width=0.7)

# Annotations
ax.text(0, 2.1, '2', ha='center', fontsize=14, fontweight='bold')
ax.text(1, 1.1, '1', ha='center', fontsize=14, fontweight='bold')
ax.text(3, 3.15, '3', ha='center', fontsize=14, fontweight='bold')
ax.text(4, 3.15, '3', ha='center', fontsize=14, fontweight='bold')

# Status labels
ax.annotate('SOLVED', xy=(1, 1.35), fontsize=12, ha='center', color=C_FP32,
            fontweight='bold')
ax.annotate('OPEN', xy=(4, 3.4), fontsize=12, ha='center', color=C_BF16,
            fontweight='bold')

ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(['Dense\n(Llama-3.1-8B)', 'MoE\n(DeepSeek-V2-Lite)'], fontsize=12)
ax.set_ylabel('Unique Outputs Across Batch Sizes')
ax.set_title('Dense vs MoE: Generation Determinism (200+ Runs)')
ax.set_ylim(0, 4.2)
ax.set_yticks([0, 1, 2, 3, 4])

# Legend
bf16_patch = mpatches.Patch(color=C_BF16, label='BF16 (default)')
fp32_patch = mpatches.Patch(color=C_FP32, label='FP32 accum')
ax.legend(handles=[bf16_patch, fp32_patch], loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/figure5_dense_vs_moe.pdf', bbox_inches='tight')
plt.savefig(f'{OUTDIR}/figure5_dense_vs_moe.png', bbox_inches='tight', dpi=300)
plt.close()
print("Figure 5 done")

print(f"\nAll figures saved to {OUTDIR}/")
