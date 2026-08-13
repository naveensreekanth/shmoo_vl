"""
plot_builder.py
---------------
100% Local, Offline Server-Side Shmoo Plot Rendering using Matplotlib.
Zero external CDN or internet dependencies.
Generates crisp, high-resolution dark-theme plot images for the Web UI and
light-theme plot images for the PDF report.
"""

import io
import base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

COLOURS = {
    'PASS':        '#2ecc71',  # Bright Green
    'FREQ_MARGIN': '#e74c3c',  # Bright Red
    'TIMING':      '#f39c12',  # Orange
    'NA':          '#e74c3c',
}


def build_shmoo_plot(
    df: pd.DataFrame,
    results,
    save_path: str = None,
    as_json: bool  = False,
    as_base64: bool = False,
):
    """
    Builds the Shmoo plot locally using Matplotlib.
    
    Parameters
    ----------
    df : pd.DataFrame
    results : ShmooResults
    save_path : str, optional
        If provided, saves light-theme PNG to disk (for PDF report).
    as_base64 : bool, optional
        If True, returns base64 data URI string `data:image/png;base64,...` (for Web UI).
    """
    is_web = as_json or as_base64

    # ── Normalize DataFrame Strings for Robust Plotting ────────────────────────
    df = df.copy()
    df['Test_Result']  = df['Test_Result'].astype(str).str.strip().str.upper()
    df['Failure_Code'] = df['Failure_Code'].fillna('NA').astype(str).str.strip().str.upper()
    df['Failure_Code'] = df['Failure_Code'].replace({'NAN': 'NA', 'NONE': 'NA', '': 'NA', 'NULL': 'NA', 'N/A': 'NA'})

    # ── Theme Colors ──────────────────────────────────────────────────────────
    bg_color    = '#0f172a' if is_web else '#ffffff'
    card_bg     = '#1e293b' if is_web else '#ffffff'
    text_color  = '#f8fafc' if is_web else '#1b2a4a'
    muted_color = '#94a3b8' if is_web else '#555555'
    grid_color  = '#334155' if is_web else '#ecf0f1'
    line_color  = '#38bdf8' if is_web else '#3498db'
    star_color  = '#a855f7' if is_web else '#9b59b6'

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140, facecolor=bg_color)
    ax.set_facecolor(card_bg)
    ax.grid(True, color=grid_color, linestyle='--', linewidth=0.8, alpha=0.7)

    # ── Scatter: PASS ─────────────────────────────────────────────────────────
    pass_mask = df['Test_Result'].isin(['PASS', 'PASSED', '1', 'TRUE', 'P'])
    if pass_mask.any():
        pass_df = df[pass_mask]
        ax.scatter(
            pass_df['VDD_V'], pass_df['Frequency_GHz'],
            c=COLOURS['PASS'], s=28, marker='s', alpha=0.85,
            label='PASS', zorder=3
        )

    # ── Scatter: FAIL codes ───────────────────────────────────────────────────
    fail_mask = ~pass_mask
    if fail_mask.any():
        fail_df = df[fail_mask]
        for code in fail_df['Failure_Code'].unique():
            sub = fail_df[fail_df['Failure_Code'] == code]
            color = COLOURS.get(code, '#e74c3c')
            ax.scatter(
                sub['VDD_V'], sub['Frequency_GHz'],
                c=color, s=28, marker='s', alpha=0.85,
                label=f'FAIL ({code})', zorder=3
            )

    # ── Predicted Boundary Line ───────────────────────────────────────────────
    vdd_min, vdd_max = float(df['VDD_V'].min()), float(df['VDD_V'].max())
    freq_min, freq_max = float(df['Frequency_GHz'].min()), float(df['Frequency_GHz'].max())

    vdd_lin = np.linspace(vdd_min, vdd_max, 200)
    if results.ransac is not None:
        bnd_freq = results.ransac.predict(vdd_lin.reshape(-1, 1)).flatten()
    else:
        bnd_freq = results.boundary_slope * vdd_lin + results.boundary_intercept

    ax.plot(
        vdd_lin, bnd_freq,
        color=line_color, linestyle='--', linewidth=2.5,
        label=f'Predicted Boundary (R²={results.boundary_r2:.3f})', zorder=4
    )

    # ── Recommended Operating Point ───────────────────────────────────────────
    ax.scatter(
        [results.recommended_vdd], [results.recommended_freq],
        c=star_color, s=180, marker='*', edgecolors='white', linewidth=1.2,
        label=f'Rec. OP ({results.recommended_vdd:.2f}V, {results.recommended_freq:.2f}GHz)', zorder=5
    )

    # Dotted guardband lines
    ax.axhline(results.recommended_freq, color=star_color, linestyle=':', linewidth=1.2, alpha=0.8)
    ax.axvline(results.recommended_vdd, color=star_color, linestyle=':', linewidth=1.2, alpha=0.8)

    # ── Labels & Formatting ───────────────────────────────────────────────────
    ax.set_title('SHMOO Characterization Plot — VDD vs Frequency', color=text_color, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('VDD (V)', color=muted_color, fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency (GHz)', color=muted_color, fontsize=10, fontweight='bold')

    x_pad = (vdd_max - vdd_min) * 0.05
    y_pad = (freq_max - freq_min) * 0.08
    ax.set_xlim(vdd_min - x_pad, vdd_max + x_pad)
    ax.set_ylim(freq_min - y_pad, freq_max + y_pad)

    ax.tick_params(colors=muted_color, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    legend = ax.legend(
        loc='upper left', frameon=True,
        facecolor=card_bg, edgecolor=grid_color, fontsize=9
    )
    for text in legend.get_texts():
        text.set_color(text_color)

    plt.tight_layout()

    # Save PNG to disk for PDF generator
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return save_path

    # Return base64 URI string for Web UI
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    data_uri = f"data:image/png;base64,{b64_str}"

    if as_json or as_base64:
        return data_uri

    return fig
