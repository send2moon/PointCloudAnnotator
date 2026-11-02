"""检查 e57 里的标注点并生成可视化图 + 明细清单。

用法: python _plot_labels.py <file.e57> [输出目录]
输出:
  <文件名>_标注分布.png   六宫格：俯视/侧视/三维/类别统计/实例统计/站点统计
  <文件名>_标注点清单.csv 每一个标注点的坐标、类别、实例、颜色、强度
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np

from e57io import E57Document, check_e57_structure

FONT = r"C:\Windows\Fonts\simhei.ttf"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

LABEL_NAMES = {0: "未标注", 1: "穗", 2: "叶", 3: "茎", 4: "其他"}
LABEL_COLORS = {0: "#969696", 1: "#ff6060", 2: "#60dc78",
                3: "#6eaaff", 4: "#ffcd5a"}

path = sys.argv[1] if len(sys.argv) > 1 else ""
if not path:
    print("用法: python _plot_labels.py <file.e57> [输出目录]")
    sys.exit(1)
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(outdir, exist_ok=True)
base = os.path.splitext(os.path.basename(path))[0]

problems = check_e57_structure(path)
print("结构体检:", problems or "OK")
if problems:
    print("!! 文件结构有问题，读取可能不完整")

doc = E57Document.read(path)
print(f"扫描 {len(doc.scans)} 个，共 {doc.total_points:,} 点")

xyz_parts, lab_parts, ins_parts, sid_parts, rgba_parts, inten_parts = [], [], [], [], [], []
counts = []
for i, s in enumerate(doc.scans):
    n = s.count
    lab = np.asarray(s.points.get("annotationLabel", np.zeros(n, np.uint8)))
    ins = np.asarray(s.points.get("annotationInstance", np.zeros(n, np.uint16)))
    xyz_parts.append(np.column_stack([
        s.points["cartesianX"], s.points["cartesianY"], s.points["cartesianZ"]]))
    lab_parts.append(lab.astype(np.int16))
    ins_parts.append(ins.astype(np.int32))
    sid_parts.append(np.full(n, i, np.int32))
    rgba_parts.append(np.column_stack([
        s.points.get("colorRed", np.zeros(n, np.uint8)),
        s.points.get("colorGreen", np.zeros(n, np.uint8)),
        s.points.get("colorBlue", np.zeros(n, np.uint8))]))
    inten_parts.append(np.asarray(s.points.get("intensity", np.zeros(n, np.float32))))
    u, c = np.unique(lab, return_counts=True)
    counts.append(dict(zip(u.tolist(), c.tolist())))

xyz = np.vstack(xyz_parts)
lab = np.concatenate(lab_parts)
ins = np.concatenate(ins_parts)
sid = np.concatenate(sid_parts)
rgb = np.vstack(rgba_parts)
inten = np.concatenate(inten_parts)
m = lab > 0

total_by_label = {k: int((lab == k).sum()) for k in LABEL_NAMES if (lab == k).any()}
print("类别统计:", {LABEL_NAMES[k]: v for k, v in total_by_label.items()})
print(f"标注点 {int(m.sum()):,} / {len(lab):,} 点 ({100*m.mean():.2f}%)")
print("标注点包围盒:", xyz[m].min(0), xyz[m].max(0))
print("全部点云包围盒:", xyz.min(0), xyz.max(0))
inst_ids = sorted(set(int(v) for v in np.unique(ins[m]) if v != 0))
print(f"实例（笔刷/框选）数 {len(inst_ids)}:", inst_ids)

# ------------------------------------------------------------------ 明细 CSV
csv_path = os.path.join(outdir, base + "_标注点清单.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["扫描序号", "X_m", "Y_m", "Z_m", "类别编码", "类别",
                "实例编号", "R", "G", "B", "强度"])
    for k in np.nonzero(m)[0]:
        w.writerow([int(sid[k]), f"{xyz[k,0]:.6f}", f"{xyz[k,1]:.6f}",
                    f"{xyz[k,2]:.6f}", int(lab[k]), LABEL_NAMES.get(int(lab[k]), "?"),
                    int(ins[k]), int(rgb[k,0]), int(rgb[k,1]), int(rgb[k,2]),
                    f"{float(inten[k]):.6f}"])
print("明细:", csv_path)

# ------------------------------------------------------------------ 作图
fig = plt.figure(figsize=(16.5, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.34, wspace=0.26)
fig.suptitle(f"{os.path.basename(path)}  标注点分布   "
             f"共 {len(lab):,} 点 / 已标注 {int(m.sum()):,} 点 ({100*m.mean():.2f}%)",
             fontsize=17, fontweight="bold", y=0.97)
stat = "　".join(f"{LABEL_NAMES[k]} {v:,}" for k, v in total_by_label.items())
fig.text(0.5, 0.935, f"类别：{stat}　|　实例（笔刷/框选）{len(inst_ids)} 个",
         ha="center", fontsize=11, color="#444")

cmap_inst = plt.get_cmap("tab20")
inst_color = {u: cmap_inst(j % 20) for j, u in enumerate(inst_ids)}


def scatter_panel(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x[~m], y[~m], s=1.2, c="#c9ced6", alpha=0.35, linewidths=0,
               label=f"未标注 {int((~m).sum()):,}")
    for k in LABEL_NAMES:
        mm = lab == k
        if k == 0 or not mm.any():
            continue
        ax.scatter(x[mm], y[mm], s=9, c=LABEL_COLORS[k], alpha=0.9,
                   linewidths=0, label=f"{LABEL_NAMES[k]} {int(mm.sum()):,}")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, markerscale=2)


ax0 = fig.add_subplot(gs[0, 0])
scatter_panel(ax0, xyz[:, 0], xyz[:, 1], "俯视 XY：标注点落在冠层顶部一小簇", "X / m", "Y / m")

ax1 = fig.add_subplot(gs[0, 1])
scatter_panel(ax1, xyz[:, 0], xyz[:, 2], "侧视 XZ：标注集中在最高处", "X / m", "Z / m")

# ---- 三维：只画标注点，按实例着色
ax2 = fig.add_subplot(gs[0, 2], projection="3d")
for u in inst_ids:
    mm = m & (ins == u)
    ax2.scatter(xyz[mm, 0], xyz[mm, 1], xyz[mm, 2], s=6,
                color=inst_color[u], depthshade=False, label=f"#{u}")
ax2.set_title(f"标注点三维分布（{len(inst_ids)} 个实例）", fontsize=12)
ax2.set_xlabel("X / m", labelpad=-2); ax2.set_ylabel("Y / m", labelpad=-2)
ax2.set_zlabel("Z / m", labelpad=-2)
ax2.view_init(elev=22, azim=-62)
ax2.tick_params(labelsize=7)
sp = xyz[m].max(0) - xyz[m].min(0) + 1e-6
ax2.set_box_aspect((sp[0], sp[1], sp[2] * 0.9))

# ---- 标注点俯视放大，按实例着色
ax3 = fig.add_subplot(gs[1, 0])
sub = xyz[m]
for u in inst_ids:
    mm = m & (ins == u)
    ax3.scatter(xyz[mm, 0], xyz[mm, 1], s=26, color=inst_color[u],
                edgecolors="white", linewidths=0.3, label=f"#{u}")
ax3.set_title(f"标注点俯视放大（{len(sub):,} 点，按实例着色）", fontsize=12)
ax3.set_xlabel("X / m"); ax3.set_ylabel("Y / m")
ax3.set_aspect("equal", adjustable="datalim")
ax3.grid(alpha=0.25)
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=6.5,
           ncol=8, markerscale=1.2, frameon=False, handletextpad=0.2,
           columnspacing=0.8)

# ---- 每个实例点数，按站点堆叠
ax4 = fig.add_subplot(gs[1, 1])
bottom = np.zeros(len(inst_ids))
scan_colors = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, max(len(doc.scans), 2)))
for si in range(len(doc.scans)):
    vals = np.array([int(((ins == u) & m & (sid == si)).sum()) for u in inst_ids])
    if not vals.any():
        continue
    ax4.bar(range(len(inst_ids)), vals, bottom=bottom,
            color=scan_colors[min(si, len(scan_colors) - 1)],
            label=f"扫描 #{si+1}")
    bottom += vals
ax4.set_xticks(range(len(inst_ids)))
ax4.set_xticklabels([str(u) for u in inst_ids], rotation=60, fontsize=7)
ax4.set_title(f"每个实例的点数（共 {len(inst_ids)} 个，按站点堆叠）", fontsize=12)
ax4.set_xlabel("实例编号"); ax4.set_ylabel("点数")
ax4.legend(fontsize=8, frameon=False)
ax4.grid(axis="y", alpha=0.3)

# ---- 高度分布：全部 vs 标注
ax5 = fig.add_subplot(gs[1, 2])
bins = np.linspace(float(xyz[:, 2].min()), float(xyz[:, 2].max()), 60)
ax5.hist(xyz[~m, 2], bins=bins, color="#c9ced6", label="全部点云")
ax5.hist(xyz[m, 2], bins=bins, color="#ff6060", label="标注点")
ax5.set_title("高度（Z）分布：标注点在冠层顶部", fontsize=12)
ax5.set_xlabel("Z / m"); ax5.set_ylabel("点数")
ax5.legend(fontsize=9, frameon=False)
ax5.grid(axis="y", alpha=0.3)

png_path = os.path.join(outdir, base + "_标注分布.png")
fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("图片:", png_path)
