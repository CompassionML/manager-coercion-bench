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
