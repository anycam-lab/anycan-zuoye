import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================= 1. 数据加载与预处理 =================
trajectory = np.load("trajectory.npy")
depths = np.load("depths.npy").squeeze()
uncertainties = np.load("uncertainties.npy").squeeze()

# 既然你实际只走了几米，我们设定一个较小的缩放系数，或者仅展示相对形状
SCALE_FACTOR = 10.0
translations = trajectory[:, :3, 3] * SCALE_FACTOR
x, y, z = translations[:, 0], translations[:, 1], translations[:, 2]

# ================= 2. 视觉风格配置 =================
plt.rcParams['font.family'] = 'sans-serif'
fig = plt.figure(figsize=(22, 10), facecolor='white', dpi=120)

# -------- 图 1：极简 3D 轨迹 (隐藏数值) --------
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# 绘制渐变轨迹
norm = plt.Normalize(0, len(x))
for i in range(len(x)-1):
    ax1.plot(x[i:i+2], y[i:i+2], z[i:i+2],
             color=plt.cm.plasma(norm(i)), linewidth=4, alpha=0.9)

# 关键：隐藏坐标轴上的数值和刻度
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])

# 去掉背景填充，保留极简网格线
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('white')
ax1.yaxis.pane.set_edgecolor('white')
ax1.zaxis.pane.set_edgecolor('white')

# 标注起终点（使用更简约的符号）
ax1.scatter(x[0], y[0], z[0], color='#2ecc71', s=200, edgecolors='black', label='Start', alpha=1)
ax1.scatter(x[-1], y[-1], z[-1], color='#e74c3c', s=200, marker='X', edgecolors='black', label='End', alpha=1)

ax1.set_title("Spatio-Temporal Trajectory Analysis", fontsize=18, fontweight='bold', pad=10)
ax1.set_xlabel("Relative Lateral", fontsize=12, labelpad=-10)
ax1.set_ylabel("Relative Vertical", fontsize=12, labelpad=-10)
ax1.set_zlabel("Forward Motion", fontsize=12, labelpad=-10)

# 调整视角：更具纵深感的透视
ax1.view_init(elev=20, azim=-35)
ax1.legend(frameon=False, loc='upper left', fontsize=12)

# -------- 图 2：高对比度深度图 (极简) --------
mid_idx = len(depths) // 2
ax2 = fig.add_subplot(2, 2, 2)
# 使用 'inferno' 色带，呈现从黑到金的质感
d_im = ax2.imshow(depths[mid_idx], cmap='inferno')
ax2.set_title("Estimated Depth Geometry", fontsize=15, fontweight='bold')
ax2.axis('off') # 彻底隐藏坐标轴
plt.colorbar(d_im, ax=ax2, shrink=0.7, aspect=20, label="Near → Far")

# -------- 图 3：误差不确定度图 (极简) --------
ax3 = fig.add_subplot(2, 2, 4)
# 使用 'hot' 色带，白色/橙色代表极高不确定度（如树冠、漂移点）
u_im = ax3.imshow(uncertainties[mid_idx], cmap='hot', vmax=np.percentile(uncertainties[mid_idx], 92))
ax3.set_title("Model Confidence/Uncertainty map", fontsize=15, fontweight='bold')
ax3.axis('off')
plt.colorbar(u_im, ax=ax3, shrink=0.7, aspect=20, label="Stable → Uncertain")

# ================= 3. 完美展示 =================
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.show()