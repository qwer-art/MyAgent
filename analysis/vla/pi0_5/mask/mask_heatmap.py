"""
Pi0.5 Action Expert Attention Mask 可视化
suffix→prefix: 全True (cross attend)  [50, 712]
suffix↔suffix: causal下三角           [50, 50]
完整 mask: concat → [50, 762]

所有子图 aspect='equal', 单位1一致

运行: python mask_heatmap.py
输出: mask_heatmap.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ---- 维度 ----
N_PREFIX = 712
N_SUFFIX = 50

# ---- 构造 mask ----
cross_mask = np.ones((N_SUFFIX, N_PREFIX), dtype=np.int8)
causal_mask = np.tril(np.ones((N_SUFFIX, N_SUFFIX), dtype=np.int8))
full_mask = np.concatenate([cross_mask, causal_mask], axis=1)

# ---- 可视化 ----
# 三幅图按列数比例分配物理宽度, aspect='equal' 保证1格=1单位
# 列数比: 712 : 50 : 762 ≈ 14.2 : 1 : 15.2
# 每行的像素高度相同(50行), 单位1自然一致
total_cols = N_PREFIX + N_SUFFIX + N_PREFIX + N_SUFFIX  # 712+50+762 = 1524
# figsize 按比例: 高度固定, 宽度按列数
fig_height = 6  # inches, enough for 50 rows
cell_size = fig_height / N_SUFFIX  # each row height in inches ≈ 0.12
fig_width = cell_size * (N_PREFIX + N_SUFFIX + N_PREFIX + N_SUFFIX + 30)  # 30 for spacing
# 更简洁: 用 gridspec_kw 按列数比例分宽度

fig, axes = plt.subplots(1, 3, figsize=(30, 6),
                          gridspec_kw={'width_ratios': [N_PREFIX, N_SUFFIX, N_PREFIX + N_SUFFIX],
                                       'wspace': 0.05})

cmap = mcolors.ListedColormap(['#E53935', '#43A047'])  # 0=红(False), 1=绿(True)

# 1) suffix→prefix (cross attend)
im0 = axes[0].imshow(cross_mask, aspect='equal', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
axes[0].set_title('suffix → prefix\n(cross attend, all True)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('prefix key position', fontsize=11)
axes[0].set_ylabel('suffix query position', fontsize=11)
axes[0].text(N_PREFIX/2, N_SUFFIX/2, f'ALL = 1 (True)\nshape: [{N_SUFFIX}, {N_PREFIX}]',
             ha='center', va='center', fontsize=14, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#2E7D32', alpha=0.85))

# 2) suffix↔suffix (causal)
im1 = axes[1].imshow(causal_mask, aspect='equal', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
axes[1].set_title('suffix ↔ suffix\n(causal, lower-tri)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('suffix key position', fontsize=11)
axes[1].set_ylabel('suffix query position', fontsize=11)
for i in range(0, N_SUFFIX, 10):
    for j in range(0, min(i + 1, N_SUFFIX), 10):
        val = int(causal_mask[i, j])
        axes[1].text(j, i, str(val), ha='center', va='center', fontsize=7,
                     color='white' if val == 1 else 'black', fontweight='bold')
# 也标注0的区域
for i in range(0, N_SUFFIX, 10):
    for j in range(i + 1, N_SUFFIX, 10):
        axes[1].text(j, i, '0', ha='center', va='center', fontsize=7,
                     color='black', fontweight='bold')

# 3) 完整 mask [50, 762]
im2 = axes[2].imshow(full_mask, aspect='equal', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
axes[2].set_title('Full Attention Mask\n[50, 762]', fontsize=14, fontweight='bold')
axes[2].set_xlabel('key position (0~711 prefix | 712~761 suffix)', fontsize=11)
axes[2].set_ylabel('suffix query position', fontsize=11)
axes[2].axvline(x=N_PREFIX - 0.5, color='white', linewidth=1.5, linestyle='--')

# 颜色条
cbar = fig.colorbar(im2, ax=axes, orientation='horizontal', fraction=0.03, pad=0.08,
                     ticks=[0, 1], aspect=40)
cbar.ax.set_xticklabels(['0 = False (masked)', '1 = True (attend)'], fontsize=11)

fig.suptitle('Pi0.5 Action Expert — Attention Mask Heatmap\n'
             'Q = suffix [50]  |  KV = prefix [712] + suffix [50] = [762]  |  1 cell = 1 unit',
             fontsize=16, fontweight='bold')

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'mask_heatmap.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"已生成: {out_path}")