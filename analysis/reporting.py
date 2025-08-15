from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import numpy as np
import math
import os
import re

# ---------------------------
# Global style
# ---------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.titlesize": 12,
})

# ---------------------------
# Filename helpers (Windows-safe)
# ---------------------------
_INVALID_WIN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

def _sanitize_filename(name: str, replacement: str = "_") -> str:
    if not isinstance(name, str):
        name = str(name)
    cleaned = _INVALID_WIN_CHARS.sub(replacement, name).strip(" .")
    return cleaned or "output"

def _prepare_output_path(output_pdf_path: str) -> str:
    outdir, fname = os.path.split(output_pdf_path or "")
    if not fname:
        fname = "report.pdf"
    if not fname.lower().endswith(".pdf"):
        fname = f"{fname}.pdf"
    fname = _sanitize_filename(fname)
    outdir = outdir or "."
    os.makedirs(outdir, exist_ok=True)
    return os.path.abspath(os.path.join(outdir, fname))

# ---------------------------
# Utilities
# ---------------------------
def _safe_val(x, fmt="{:.3f}"):
    if x is None:
        return "—"
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return "—"
        return fmt.format(xf)
    except Exception:
        return "—"

def _make_table(ax, title, rows):
    """
    Compact table with a reserved top band for the title to prevent overlap.
    rows: list[(label, value_string)]
    """
    ax.clear()
    ax.set_axis_off()

    # More breathing room between title and table
    table = ax.table(
        cellText=[[lab, val] for (lab, val) in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.85],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.10)

    ax.text(
        0.0, 0.97, title,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="semibold",
        va="top", ha="left"
    )
    return table

def _make_wide_table(ax, title, header, rows, font_size=8, bbox=(0.0, 0.0, 1.0, 0.88)):
    """
    Wide table helper used on the Mean Summary page.
    header: list[str]
    rows:   list[list[str]]
    """
    ax.clear()
    ax.set_axis_off()

    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="center",
        cellLoc="left",
        colLoc="left",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.08)

    ax.text(
        0.0, 0.98, title,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="semibold",
        va="top", ha="left"
    )
    return table

# ---------------------------
# Plotters (draw onto Axes)
# ---------------------------
def _draw_cont_bands_with_labels(ax, t_vals, categories_cont):
    """
    Draw shaded horizontal bands with centered labels (like plot_comfort_timeseries).
    """
    if not categories_cont:
        return
    lo_min = min(lo for (lo, hi), _ in categories_cont)
    hi_max = max(hi for (lo, hi), _ in categories_cont)
    for (lo, hi), label in categories_cont:
        ax.axhspan(lo, hi, color='gray', alpha=0.10)
        y_mid = 0.5 * (lo + hi)
        x_mid = t_vals[len(t_vals) // 2] if len(t_vals) > 0 else 0.5
        ax.text(x_mid, y_mid, label, ha='center', va='center',
                fontsize=8, color='black', alpha=0.85)
    ax.set_ylim(lo_min, hi_max)

def _draw_continuous_cc_on_ax(ax, Cx, Cy, Cz, ride_speed_mph: float, categories_cont=None):
    """
    Continuous comfort plot (Cx/Cy/Cz) with shaded bands + labels and elapsed seconds (5s step).
    Legend is pinned upper-left to avoid band labels.
    """
    n = min(len(Cx), len(Cy), len(Cz))
    if n <= 0:
        ax.text(0.5, 0.5, "No continuous (C) data", ha="center", va="center")
        return

    t_5s_secs = [i * 5 for i in range(n)]

    # Bands + labels
    _draw_cont_bands_with_labels(ax, t_5s_secs, categories_cont)

    # Lines (explicit colors to stabilize legend)
    h1, = ax.plot(t_5s_secs, Cx[:n], marker='o', linestyle='-', label='Cx (X)', color='#1f77b4')
    h2, = ax.plot(t_5s_secs, Cy[:n], marker='s', linestyle='-', label='Cy (Y)', color='#ff7f0e')
    h3, = ax.plot(t_5s_secs, Cz[:n], marker='^', linestyle='-', label='Cz (Z)', color='#2ca02c')

    # Formatting
    ax.set_ylabel('5s Weighted RMS Accel (m/s²)')
    ax.set_xlabel('Time (seconds elapsed)')
    ax.set_title(f'Continuous Comfort — 5s Interval @ {ride_speed_mph} mph', fontsize=10)

    # Limit xticks for readability
    if n > 15:
        step = max(1, n // 10)
        ax.set_xticks(t_5s_secs[::step])

    ax.legend([h1, h2, h3], ['Cx (X)', 'Cy (Y)', 'Cz (Z)'],
              loc='upper left', fontsize=8, framealpha=0.85)
    ax.grid(True, linestyle='--', alpha=0.35)

def _draw_iso_on_ax(ax, ax_s, ay_s, az_s, av_s, ride_speed_mph: float):
    """
    ISO panel (ax/ay/az/a_v) — elapsed seconds at 1s step.
    """
    n = min(len(ax_s), len(ay_s), len(az_s), len(av_s))
    if n <= 0:
        ax.text(0.5, 0.5, "No ISO (a_x/a_y/a_z/a_v) data", ha="center", va="center")
        return
    t_secs = np.arange(n, dtype=float)

    h1, = ax.plot(t_secs, ax_s[:n], label='a_x', color='#1f77b4')
    h2, = ax.plot(t_secs, ay_s[:n], label='a_y', color='#ff7f0e')
    h3, = ax.plot(t_secs, az_s[:n], label='a_z', color='#2ca02c')
    h4, = ax.plot(t_secs, av_s[:n], linestyle=':', color='black', label='a_v')

    ax.set_ylabel('Accel / Composite (m/s²)')
    ax.set_xlabel('Time (seconds elapsed)')
    ax.set_title(f'ISO 2631 — 1s Interval @ {ride_speed_mph} mph', fontsize=10)

    if n > 15:
        step = max(1, n // 10)
        ax.set_xticks(t_secs[::step])

    ax.legend([h1, h2, h3, h4], ['a_x', 'a_y', 'a_z', 'a_v'],
              loc='upper right', fontsize=8, framealpha=0.85)
    ax.grid(True, linestyle='--', alpha=0.35)

def _draw_mean_nmv_on_ax(ax, nmvs, categories_mean, floor_triaxials, ride_speed_mph: float):
    """
    EXACTLY like NMV plot:
      - floors at x=0.50 (marker 'x')
      - seats  at x=1.25 (marker 'o')
      - comfort bands as horizontal lines with labels at x=1.5
    """
    ax.set_xlim(0, 2)
    y_lo = min(lo for (lo, hi), _ in categories_mean)
    y_hi = max(hi for (lo, hi), _ in categories_mean)
    ax.set_ylim(y_lo, y_hi)

    # Floors
    for triax, nmv in nmvs:
        if triax in floor_triaxials:
            ax.scatter(0.50, nmv, label=f'Floor Triax: {triax} N_M_V', marker='x')
    # Seats
    for triax, nmv in nmvs:
        if triax not in floor_triaxials:
            ax.scatter(1.25, nmv, label=f'Seat Triax: {triax} N_M_V', marker='o')

    # Comfort lines + labels
    for (lo, hi), label in categories_mean:
        ax.axhline(lo, linestyle='-', color='gray', linewidth=0.5)
        ax.text(1.5, lo + 0.1, label, fontsize=8, va='bottom')

    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f'Mean Comfort (Standard) — NMV @ {ride_speed_mph} mph', fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("Comfort Index")

def _draw_mean_nvd_nva_on_ax(ax, nvds, nvaz, categories_mean, pair_dict, ride_speed_mph: float):
    """
    EXACTLY like NVD/NVA plot:
      - NVD (floors) at x=0.50 marker 'x'
      - NVA (seats)  at x=1.25 marker 'o' with orientation in label
      - comfort bands as horizontal lines with labels at x=1.5
    """
    ax.set_xlim(0, 2)
    y_lo = min(lo for (lo, hi), _ in categories_mean)
    y_hi = max(hi for (lo, hi), _ in categories_mean)
    ax.set_ylim(y_lo, y_hi)

    # NVD (floors)
    for triax, nvd in nvds:
        if nvd is None:
            continue
        ax.scatter(0.50, nvd, label=f'triax: {triax} Standing (NVD)', marker='x')

    # NVA (seats)
    for triax, nva in nvaz:
        if nva is None:
            continue
        orientation = None
        for (floor, seat), label in pair_dict or []:
            if seat == triax:
                orientation = label
                break
        suffix = f", {orientation}" if orientation else ""
        ax.scatter(1.25, nva, label=f'triax: {triax} Seated (NVA){suffix}', marker='o')

    # Comfort lines + labels
    for (lo, hi), label in categories_mean:
        ax.axhline(lo, linestyle='-', color='gray', linewidth=0.5)
        ax.text(1.5, lo + 0.1, label, fontsize=8, va='bottom')

    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f'Mean Comfort (Complete) — NVD / NVA @ {ride_speed_mph} mph', fontsize=10)
    ax.set_xticks([])
    ax.set_ylabel("Comfort Index")

# ---------------------------
# Row renderer (per triax)
# ---------------------------
def _render_triax_row(fig, gs_row, triax_id, md, categories_cont, ride_speed_mph):
    """
    One visual row per triax with TWO columns:
      Col 1: Continuous Comfort — table(Max Cc*) + plot(Ccx/Ccy/Ccz @5s)  [uses categories_cont with labels]
      Col 2: ISO 2631 — table(Max a*) + plot(ax/ay/az/a_v @1s)
    """
    row_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_row, wspace=0.15)

    # ---------- Column 1: Continuous Comfort ----------
    col1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=row_gs[0, 0], hspace=0.05)
    ax_tbl1 = fig.add_subplot(col1[0, 0])
    rows1 = [
        ("Max Ccx", _safe_val(md.get("max_Cx"))),
        ("Max Ccy", _safe_val(md.get("max_Cy"))),
        ("Max Ccz", _safe_val(md.get("max_Cz"))),
    ]
    _make_table(ax_tbl1, f"Triax {triax_id} — Continuous Comfort (metrics)", rows1)

    ax_plot1 = fig.add_subplot(col1[1, 0])
    _draw_continuous_cc_on_ax(ax_plot1, md.get("Cx"), md.get("Cy"), md.get("Cz"),
                              ride_speed_mph, categories_cont=categories_cont)

    # ---------- Column 2: ISO 2631 ----------
    col2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=row_gs[0, 1], hspace=0.05)
    ax_tbl2 = fig.add_subplot(col2[0, 0])
    rows2 = [
        ("Max ax", _safe_val(md.get("max_ax"))),
        ("Max ay", _safe_val(md.get("max_ay"))),
        ("Max az", _safe_val(md.get("max_az"))),
        ("Max av", _safe_val(md.get("max_av"))),
    ]
    _make_table(ax_tbl2, f"Triax {triax_id} — ISO 2631 (metrics)", rows2)

    ax_plot2 = fig.add_subplot(col2[1, 0])
    _draw_iso_on_ax(ax_plot2, md.get("ax"), md.get("ay"), md.get("az"), md.get("av"), ride_speed_mph)

# ---------------------------
# Mean Summary Page (table + two plots on one page)
# ---------------------------
def _render_mean_summary_page(
    pdf,
    run_title,
    nmvs,
    nvds,
    nvaz,
    categories_mean,
    floor_triaxials,
    seat_pairs,
    ride_speed_mph
):

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Mean Comfort Summary — " + run_title, fontsize=12, fontweight="semibold")

    # 2 rows x 2 cols; top row gets more height for large graphs
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[1, 1],
        height_ratios=[1.8, 1.0],   # top ≈ 64% / bottom ≈ 36%
        wspace=0.20, hspace=0.28
    )

    # ---- Top-left: NMV (Standard) large plot ----
    ax_nmv = fig.add_subplot(gs[0, 0])
    _draw_mean_nmv_on_ax(
        ax=ax_nmv,
        nmvs=nmvs,
        categories_mean=categories_mean,
        floor_triaxials=floor_triaxials,
        ride_speed_mph=ride_speed_mph
    )

    # ---- Top-right: NVD/NVA (Complete) large plot ----
    ax_nvdnva = fig.add_subplot(gs[0, 1])
    _draw_mean_nvd_nva_on_ax(
        ax=ax_nvdnva,
        nvds=nvds,
        nvaz=nvaz,
        categories_mean=categories_mean,
        pair_dict=seat_pairs or [],
        ride_speed_mph=ride_speed_mph
    )

    # ---- Bottom: Wide compact table spanning both columns ----
    ax_table = fig.add_subplot(gs[1, :])

    # Build table rows: Triax | NMV | NVD | NVA
    tr_set = set([t for t, _ in nmvs]) | set([t for t, _ in nvds]) | set([t for t, _ in nvaz])
    tr_list = sorted(tr_set)

    rows = []
    for t in tr_list:
        nmv = next((v for (tt, v) in nmvs if tt == t), None)
        nvd = next((v for (tt, v) in nvds if tt == t), None)
        nva = next((v for (tt, v) in nvaz if tt == t), None)
        rows.append([f"T{t}", _safe_val(nmv), _safe_val(nvd), _safe_val(nva)])

    # Slightly smaller bbox height so title doesn't crowd table
    _make_wide_table(
        ax_table,
        title="Per-Triax Mean Comfort Metrics",
        header=["Triax", "NMV", "NVD", "NVA"],
        rows=rows,
        font_size=8,
        bbox=(0.0, 0.02, 1.0, 0.82)   # lower + shorter table -> more whitespace, less overlap
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Pagination helper
# ---------------------------
def _chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ---------------------------
# Public API
# ---------------------------
def export_run_report(
    output_pdf_path,
    run_title,
    metrics_dict,
    categories_cont,          # [((lo, hi), label), ...] for Continuous (C) plots
    categories_mean,          # [((lo, hi), label), ...] for NMV/NVD/NVA mean plots
    floor_triaxials,          # e.g., [1,3,5]
    seat_pairs=None,          # [((floor, seat), orientation_name), ...]
    rows_per_page=2,
    ride_speed_mph=0,         # for plot titles
):
    """
    Build a multi-page PDF.

    Inputs:
      output_pdf_path: path to save PDF (extension auto-added if missing)
      run_title: title string (e.g., "Clause 6 Summary — <ride_id>")
      metrics_dict: dict keyed by triax "1".."6" containing:
         Cx, Cy, Cz, max_Cx, max_Cy, max_Cz,
         ax, ay, az, av, max_ax, max_ay, max_az, max_av,
         N_MV, N_VD, N_VA
      categories_cont: continuous comfort bands for C plots
      categories_mean: mean comfort bands for NMV/NVD/NVA
      floor_triaxials: list of floor triax IDs (for NMV x-positioning)
      seat_pairs: [((floor, seat), orientation_name), ...] (for NVA labels)
      rows_per_page: number of triax rows per page
      ride_speed_mph: numeric, used in titles

    Layout:
      • Pages 1..N: rows of triaxes
      • Final page: Mean Summary — table + two stacked plots (NMV and NVD/NVA)
    """
    safe_pdf_path = _prepare_output_path(output_pdf_path)

    triaxes = [k for k in sorted(metrics_dict.keys(), key=lambda s: int(s))]

    # Build nmvs / nvds / nvaz lists from metrics_dict for the mean summary
    nmvs = []
    nvds = []
    nvaz = []
    for k in triaxes:
        tr = int(k)
        md = metrics_dict[k]
        if md.get("N_MV") is not None:
            nmvs.append((tr, md.get("N_MV")))
        nvds.append((tr, md.get("N_VD")))  # may be None for seats
        nvaz.append((tr, md.get("N_VA")))  # may be None for floors

    with PdfPages(safe_pdf_path) as pdf:
        for page_trs in _chunk(triaxes, rows_per_page):
            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)  # landscape
            fig.suptitle(run_title, fontsize=12, fontweight="semibold")
            page_gs = gridspec.GridSpec(len(page_trs), 1, figure=fig, hspace=0.30)

            for i, tr in enumerate(page_trs):
                md = metrics_dict[tr]
                _render_triax_row(
                    fig, page_gs[i, 0], int(tr), md,
                    categories_cont=categories_cont,
                    ride_speed_mph=ride_speed_mph
                )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ---- Mean Summary page (table + two plots) ----
        _render_mean_summary_page(
            pdf=pdf,
            run_title=run_title,
            nmvs=nmvs,
            nvds=nvds,
            nvaz=nvaz,
            categories_mean=categories_mean,
            floor_triaxials=floor_triaxials,
            seat_pairs=seat_pairs or [],
            ride_speed_mph=ride_speed_mph
        )
