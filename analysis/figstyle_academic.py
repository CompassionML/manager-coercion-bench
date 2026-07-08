"""Academic figure style (serif, dense, colour-graded, caption-driven).

Matches the conventional look of the TAC paper figures: serif type that sits inside
the paper body, rich colour, information density, and NO baked-in headline / subtitle /
branding (the interpretation lives in the LaTeX \\caption). Import setup() and C.
"""
import matplotlib.pyplot as plt

# model colours (concern gradient: teal declines -> crimson climbs), for labels + points
C = {
    "Opus 4.8":        "#0B6E68",
    "Sonnet 4.6":      "#13938C",
    "DeepSeek V4 Pro": "#D79A21",
    "GPT-5.2":         "#D9741F",
    "Gemini 2.5 Pro":  "#C8442B",
    "Grok 4.3":        "#A5152A",
}
HEAT = "RdBu_r"   # diverging heatmap colormap: high value = red = deeper coercion


# Paper body text is 10pt; figure text targets 9pt (1pt below) AFTER the figure
# is scaled to \linewidth: scripts pass base = 9 * figwidth / TEXTWIDTH_IN.
TEXTWIDTH_IN = 6.53   # letter paper, 2.5cm margins (caml.cls geometry)


def setup(base=11.5):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })


def colour_labels(ticklabels, names):
    """Colour a set of axis tick labels by model (bold), TAC-style."""
    for t, name in zip(ticklabels, names):
        t.set_color(C.get(name, "#111111"))
        t.set_fontweight("bold")


# shared academic-skin constants (match the TAC paper figures)
GRID = "#CFCFCF"      # dotted gridlines
EDGE = "#1A1A1A"      # dark edge on bars / markers
INK = "#111111"       # primary text
INK_SOFT = "#555555"  # secondary text ("ns", count labels)
BAND = "#FBE9E3"      # light red threat band (as in the headline waves figure)
BANDTXT = "#C0523A"


def style_axes(ax, grid="y"):
    """White panel, left+bottom spines only, dotted gridlines behind the data."""
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#333333")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(color="#333333", length=3, width=0.8)
    if grid:
        ax.grid(axis=grid, ls=(0, (1, 3)), lw=0.8, color=GRID, zorder=0)
        ax.set_axisbelow(True)


def sig_star(p):
    """*** p<0.001, ** p<0.01, * p<0.05, else ns (grey by convention)."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def lighten(hex_color, amount):
    """Blend a hex colour toward white by `amount` in [0,1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * amount) for v in (r, g, b))
