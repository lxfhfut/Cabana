"""
Before/after visualization for analyzer.py fixes #1-4 and batch.py fix #5.
Saves all figures to the viz_results/ folder.
"""

import os
import sys
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.ndimage as ndi
import imageio.v3 as iio
from pathlib import Path
from skimage.morphology import skeletonize, remove_small_holes
from skimage.draw import circle_perimeter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
VIZ_DIR = PROJECT_ROOT / "viz_results"

from cabana.log import Log
Log.init_log_path(str(VIZ_DIR))

from cabana.analyzer import SkeletonAnalyzer as NewAnalyzer

# ── Load the pre-fix analyzer ────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "analyzer_before", "/tmp/analyzer_before_fixed.py"
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
OldAnalyzer = _mod.SkeletonAnalyzer


# ── Helper: run both analyzers on a numpy binary image ──────────────────────

def run_both(binary_img, dark_line=False, skel_thresh=0, branch_thresh=0):
    results = []
    for Cls in (OldAnalyzer, NewAnalyzer):
        sa = Cls(skel_thresh=skel_thresh, branch_thresh=branch_thresh,
                 dark_line=dark_line)
        sa.analyze_image(binary_img.copy())
        sa.calc_curve_all()
        results.append(sa)
    return tuple(results)


def run_both_preskel(skel_img, dark_line=False, skel_thresh=0, branch_thresh=0):
    """Run both analyzers on a pre-built skeleton (bypasses skeletonize step)."""
    FOREGROUND = 255
    results = []
    for Cls in (OldAnalyzer, NewAnalyzer):
        sa = Cls(skel_thresh=skel_thresh, branch_thresh=branch_thresh,
                 dark_line=dark_line)
        sa.raw_image = skel_img.copy()
        sa.FOREGROUND = FOREGROUND
        sa.BACKGROUND = 0
        sa.skel_image = skel_img.copy()
        sa.construct_graphs()
        sa.draw_key_points()
        sa.calc_len_map_all()
        sa.calc_total_len()
        sa.calc_proj_area()
        sa.calc_growth_unit()
        sa.calc_frac_dim()
        sa.calc_lacunarity()
        sa.calc_curve_all()
        results.append(sa)
    return tuple(results)


def metrics_text(sa):
    return (f"branches : {sa.num_branches}\n"
            f"tips     : {sa.num_tips}\n"
            f"length   : {sa.total_length:.1f} px")


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic skeletons (direct pixel arrays, no skeletonize ambiguity)
# ═══════════════════════════════════════════════════════════════════════════

def make_missing_pattern_junction():
    """
    Branch point at (15,15) with neighbors {NW, N, S} — one of the 40
    3-neighbor patterns missing from selems_3.  Old code misses it,
    so the component is misclassified as >2-endpoints (after the fix
    it is a proper T-junction with 1 branch point).
    """
    img = np.zeros((30, 30), dtype=np.uint8)
    cx, cy = 15, 15
    img[cy, cx] = 255
    for i in range(1, 12):
        img[cy - i, cx - i] = 255   # NW arm
    for i in range(1, 12):
        img[cy - i, cx] = 255       # N arm
    for i in range(1, 12):
        img[cy + i, cx] = 255       # S arm
    return img


def make_covered_pattern_junction():
    """
    Standard T-junction: branch pixel has N, W, E neighbors —
    covered by selems_3[0]. Both old and new code handle this.
    """
    img = np.zeros((30, 35), dtype=np.uint8)
    img[15, 5:30] = 255   # horizontal bar
    img[4:15, 15] = 255   # vertical arm
    return img


def make_multi_junction():
    """
    Three T-junctions connected in a chain.  Mix of covered and
    uncovered patterns; new code detects all, old code may miss some.
    """
    img = np.zeros((50, 120), dtype=np.uint8)
    img[25, 10:110] = 255           # horizontal backbone
    img[5:25, 30] = 255             # vertical arm 1 (standard T)
    img[5:25, 60] = 255             # vertical arm 2
    img[5:25, 90] = 255             # vertical arm 3
    # Add short diagonal stubs to create non-standard patterns
    for i in range(1, 8):
        img[25 + i, 30 + i] = 255  # SE stub from junction 1
        img[25 + i, 60 - i] = 255  # SW stub from junction 2
    return img


def make_loop():
    """Pure circular skeleton — old code gives length 0, new code measures it."""
    img = np.zeros((80, 80), dtype=np.uint8)
    rr, cc = circle_perimeter(40, 40, 28)
    valid = (rr >= 0) & (rr < 80) & (cc >= 0) & (cc < 80)
    img[rr[valid], cc[valid]] = 255
    return img


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Synthetic cases
# ═══════════════════════════════════════════════════════════════════════════

cases_raw = [
    ("{NW,N,S} junction\n(uncovered pattern, fix #1)", make_missing_pattern_junction()),
    ("Standard T-junction\n(covered — reference)", make_covered_pattern_junction()),
    ("Multi-junction chain\n(mixed patterns, fix #1)", make_multi_junction()),
    ("Pure loop\n(fix #4: 0 → full length)", make_loop()),
]

# Use pre-skeletonized analysis so patterns are exact
cases = []
for label, raw in cases_raw:
    # For the "loop", analyze_image will produce a clean circle skeleton
    if "loop" in label.lower():
        old_sa, new_sa = run_both(raw, dark_line=False, skel_thresh=0, branch_thresh=0)
    else:
        old_sa, new_sa = run_both_preskel(raw, dark_line=False, skel_thresh=0, branch_thresh=0)
    cases.append((label, raw, old_sa, new_sa))

fig1, axes = plt.subplots(len(cases), 4, figsize=(16, 4.5 * len(cases)))
fig1.suptitle("Synthetic skeleton tests: Before vs After fixes #1 & #4",
              fontsize=13, fontweight='bold', y=1.005)

col_titles = ["Binary input", "Key points — BEFORE", "Key points — AFTER", "Metrics Δ"]
for j, ct in enumerate(col_titles):
    axes[0, j].set_title(ct, fontsize=10, fontweight='bold')

for i, (label, raw, old_sa, new_sa) in enumerate(cases):
    axes[i, 0].imshow(raw, cmap='gray')
    axes[i, 0].set_ylabel(label, fontsize=8, labelpad=4)

    for j_off, sa in enumerate((old_sa, new_sa)):
        ax = axes[i, j_off + 1]
        if sa.key_pts_image is not None:
            ax.imshow(sa.key_pts_image)
        else:
            ax.imshow(raw, cmap='gray')
            ax.text(0.5, 0.5, "no graph", ha='center', va='center',
                    transform=ax.transAxes, color='red', fontsize=9)

    ax = axes[i, 3]
    ax.axis('off')
    delta_len = new_sa.total_length - old_sa.total_length
    delta_brh = new_sa.num_branches - old_sa.num_branches
    delta_tip = new_sa.num_tips - old_sa.num_tips
    color = 'lightgreen' if abs(delta_len) > 0.5 or delta_brh != 0 or delta_tip != 0 else 'lightyellow'
    txt = (f"BEFORE\n{metrics_text(old_sa)}\n\n"
           f"AFTER\n{metrics_text(new_sa)}\n\n"
           f"Δ length  : {delta_len:+.1f} px\n"
           f"Δ branches: {delta_brh:+d}\n"
           f"Δ tips    : {delta_tip:+d}")
    ax.text(0.05, 0.97, txt, transform=ax.transAxes, va='top',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

    for j in range(4):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

red_p    = mpatches.Patch(color=(1, 0, 0),   label='Endpoint (red)')
yellow_p = mpatches.Patch(color=(1, 1, 0),   label='Branch point (yellow)')
fig1.legend(handles=[red_p, yellow_p], loc='lower center', ncol=2,
            fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig1.tight_layout()
out1 = VIZ_DIR / "fig1_synthetic_fixes1_4.png"
fig1.savefig(out1, dpi=120, bbox_inches='tight')
print(f"Saved {out1}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Real image: sduiwew0hkdsa.png
# ═══════════════════════════════════════════════════════════════════════════

real_path = PROJECT_ROOT / "data/Synthetic/sduiwew0hkdsa.png"
real_img  = iio.imread(str(real_path))
if real_img.ndim > 2:
    real_img = real_img[..., 0]

old_real, new_real = run_both(real_img, dark_line=True, skel_thresh=20, branch_thresh=10)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.suptitle("Real image (sduiwew0hkdsa.png, dark fibres): Before vs After fixes #1–4",
              fontsize=13, fontweight='bold')

def show_map(ax, data, title, cmap='hot'):
    if data is not None and data.max() > 0:
        im = ax.imshow(data, cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(np.zeros_like(real_img), cmap='gray')
        ax.text(0.5, 0.5, "empty", ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=10)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')

if old_real.key_pts_image is not None:
    axes2[0, 0].imshow(old_real.key_pts_image)
axes2[0, 0].set_title("Key points — BEFORE", fontsize=9, fontweight='bold')
axes2[0, 0].axis('off')
show_map(axes2[0, 1], old_real.length_map_all,  "Length map — BEFORE")
show_map(axes2[0, 2], old_real.curve_map_all,   "Curvature map — BEFORE")

if new_real.key_pts_image is not None:
    axes2[1, 0].imshow(new_real.key_pts_image)
axes2[1, 0].set_title("Key points — AFTER", fontsize=9, fontweight='bold')
axes2[1, 0].axis('off')
show_map(axes2[1, 1], new_real.length_map_all,  "Length map — AFTER")
show_map(axes2[1, 2], new_real.curve_map_all,   "Curvature map — AFTER")

metrics_summary = (
    f"{'Metric':<26} {'BEFORE':>10} {'AFTER':>10} {'Δ':>10}\n"
    f"{'-'*58}\n"
    f"{'Total length (px)':<26} {old_real.total_length:>10.1f} {new_real.total_length:>10.1f} "
    f"{new_real.total_length - old_real.total_length:>+10.1f}\n"
    f"{'Branches':<26} {old_real.num_branches:>10d} {new_real.num_branches:>10d} "
    f"{new_real.num_branches - old_real.num_branches:>+10d}\n"
    f"{'Tips':<26} {old_real.num_tips:>10d} {new_real.num_tips:>10d} "
    f"{new_real.num_tips - old_real.num_tips:>+10d}\n"
    f"{'Growth unit (px)':<26} {old_real.growth_unit:>10.2f} {new_real.growth_unit:>10.2f} "
    f"{new_real.growth_unit - old_real.growth_unit:>+10.2f}\n"
    f"{'Fractal dim':<26} {old_real.frac_dim:>10.4f} {new_real.frac_dim:>10.4f} "
    f"{new_real.frac_dim - old_real.frac_dim:>+10.4f}\n"
    f"{'Avg curvature (all)':<26} {old_real.avg_curve_all:>10.3f} {new_real.avg_curve_all:>10.3f} "
    f"{new_real.avg_curve_all - old_real.avg_curve_all:>+10.3f}\n"
)
fig2.text(0.01, 0.005, metrics_summary, fontsize=8, family='monospace',
          verticalalignment='bottom',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig2.tight_layout(rect=[0, 0.20, 1, 0.97])
out2 = VIZ_DIR / "fig2_real_image_before_after.png"
fig2.savefig(out2, dpi=120, bbox_inches='tight')
print(f"Saved {out2}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Fix #5: Jensen's inequality illustration
# ═══════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)
gap_areas     = rng.lognormal(mean=3.0, sigma=1.2, size=200)
radii_true    = np.sqrt(gap_areas / np.pi)

mean_area     = np.mean(gap_areas)
std_area      = np.std(gap_areas)
mean_r_wrong  = np.sqrt(mean_area / np.pi)
std_r_wrong   = np.sqrt(std_area  / np.pi)
mean_r_correct = np.mean(radii_true)
std_r_correct  = np.std(radii_true)

fig3, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("Fix #5: Radius statistics — Jensen's inequality\n"
              "sqrt(mean(area)/π) ≠ mean(sqrt(area/π))",
              fontsize=12, fontweight='bold')

ax_a.hist(gap_areas, bins=40, color='steelblue', alpha=0.7, edgecolor='k', linewidth=0.3)
ax_a.axvline(mean_area, color='navy', lw=2, label=f"mean area = {mean_area:.1f} µm²")
ax_a.set_xlabel("Gap area (µm²)", fontsize=10)
ax_a.set_ylabel("Count", fontsize=10)
ax_a.set_title("Simulated gap area distribution (log-normal)", fontsize=10)
ax_a.legend(fontsize=9)

ax_r.hist(radii_true, bins=40, color='darkorange', alpha=0.7, edgecolor='k', linewidth=0.3)
ax_r.axvline(mean_r_wrong,   color='red',   lw=2.5, linestyle='--',
             label=f"OLD: sqrt(mean_area/π) = {mean_r_wrong:.2f} µm")
ax_r.axvline(mean_r_correct, color='green', lw=2.5, linestyle='-',
             label=f"NEW: mean(sqrt(area/π)) = {mean_r_correct:.2f} µm")
ax_r.set_xlabel("Gap radius (µm)", fontsize=10)
ax_r.set_ylabel("Count", fontsize=10)
ax_r.set_title("Radius distribution and mean estimates", fontsize=10)
ax_r.legend(fontsize=9)

rel_err = abs(mean_r_wrong - mean_r_correct) / mean_r_correct * 100
fig3.text(0.5, 0.01,
          f"Old method overestimates mean radius by {rel_err:.1f}%\n"
          f"Old std (meaningless): sqrt(std_area/π) = {std_r_wrong:.2f} µm   "
          f"vs   correct std(radii) = {std_r_correct:.2f} µm",
          ha='center', fontsize=9, style='italic',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig3.tight_layout(rect=[0, 0.12, 1, 1])
out3 = VIZ_DIR / "fig3_fix5_radius_stats.png"
fig3.savefig(out3, dpi=120, bbox_inches='tight')
print(f"Saved {out3}")


print("\nAll figures saved to viz_results/")
print(f"  {out1.name}")
print(f"  {out2.name}")
print(f"  {out3.name}")
