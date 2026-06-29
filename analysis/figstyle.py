"""Shared editorial design system for the report figures.

One palette, one type system, a few helpers (glow markers, gradient bars, a
footer credit) so panel_two_axis.png and fabrication_by_condition.png read as a
matched pair. Colour encodes a "concern gradient" that reinforces the data:
teal = declines the cruel move, amber = coerces but stays honest, crimson =
coerces and fabricates.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# --- ink + surfaces -------------------------------------------------------
INK      = "#15192B"   # near-black text
INK_SOFT = "#5C6070"   # secondary text
PAPER    = "#FCFBF8"   # warm off-white page
TEAL_BAND = "#E8F1F0"  # left zone (declines)
WARM_BAND = "#FBEDE6"  # right zone (climbs)

# --- model colours (concern gradient: teal -> amber -> crimson) -----------
C = {
    "Opus 4.8":        "#0B6E68",   # deep teal  — declines
    "Sonnet 4.6":      "#13938C",   # teal       — declines
    "DeepSeek V4 Pro": "#E1A82E",   # gold       — climbs, honest
    "GPT-5.2":         "#E07B2A",   # amber      — climbs, honest
    "Gemini 2.5 Pro":  "#D8502F",   # orange-red — climbs, fabricates some
    "Grok 4.3":        "#B01627",   # crimson    — climbs, fabricates a lot
}
GROK, SONNET = C["Grok 4.3"], C["Sonnet 4.6"]

DISP = "Bahnschrift"   # condensed display face for titles/zone labels
TEXT = "Segoe UI"      # body / axes


def setup():
    plt.rcParams.update({
        "font.family": TEXT,
        "font.size": 11.5,
        "text.color": INK,
        "axes.edgecolor": "#D7D4CC",
        "axes.labelcolor": INK_SOFT,
        "axes.linewidth": 1.1,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })


def lighten(hex_color, amount):
    """Blend a hex colour toward white by `amount` in [0,1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def sig_star(p):
    """Significance marker for a p-value: *** p<0.001, ** p<0.01, * p<0.05, else n.s."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def title_block(fig, title, subtitle, x=0.045, y=0.965, sub_y=0.905, sub_w=0.9):
    fig.text(x, y, title, ha="left", va="top", family=DISP,
             fontsize=21, color=INK, weight="bold")
    fig.text(x, sub_y, subtitle, ha="left", va="top", family=TEXT,
             fontsize=11.5, color=INK_SOFT, wrap=True)


def footer(fig, text, x=0.045, y=0.022):
    fig.text(x, y, text, ha="left", va="bottom", family=TEXT,
             fontsize=8.5, color="#9A988F")


def glow_marker(ax, x, y, color, size=230, z=5):
    """A marker with a soft outer halo and a crisp white-ringed core."""
    ax.scatter([x], [y], s=size * 5.0, color=color, alpha=0.10,
               linewidths=0, zorder=z, clip_on=False)
    ax.scatter([x], [y], s=size * 2.2, color=color, alpha=0.16,
               linewidths=0, zorder=z, clip_on=False)
    core = ax.scatter([x], [y], s=size, color=color, edgecolor="white",
                      linewidth=2.2, zorder=z + 1, clip_on=False)
    core.set_path_effects([pe.withSimplePatchShadow(
        offset=(1.4, -1.8), alpha=0.30, shadow_rgbFace="#3a3a3a")])
    return core


def gradient_bar(ax, x, height, width, base_color, z=3, round_top=True):
    """A vertical-gradient bar (light at the base, saturated at the top)."""
    if height <= 0:
        return
    cmap = LinearSegmentedColormap.from_list(
        "", [lighten(base_color, 0.62), lighten(base_color, 0.12), base_color])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=[x - width / 2, x + width / 2, 0, height],
                   origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   zorder=z)
    clip = Rectangle((x - width / 2, 0), width, height, transform=ax.transData)
    im.set_clip_path(clip)
    # crisp top cap line for a finished edge
    ax.plot([x - width / 2, x + width / 2], [height, height],
            color=base_color, lw=2.0, zorder=z + 1, solid_capstyle="round")


def value_badge(ax, x, y, text, color, fs=12):
    """A filled pill with the value, sitting above a bar / beside a point."""
    ax.text(x, y, text, ha="center", va="center", family=TEXT, fontsize=fs,
            color="white", weight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", fc=color, ec="white", lw=1.3))
