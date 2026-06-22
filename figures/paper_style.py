"""Shared paper-figure style for every script-generated graph.

Applies SciencePlots' science+nature look (without LaTeX, so a custom font can be
used), sets Inter as the type family, and keeps only the left and bottom spines.
Import and call :func:`apply_style` before creating any figure, then pass each
Axes through :func:`style_axes` after plotting.

``BBOX_COLOR`` is the single ground-truth box color used by the report gallery.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_INTER_DIR = Path("/usr/share/fonts/opentype/inter")
BBOX_COLOR = "#2FFFF2"


def _style_files():
    """Locate the SciencePlots science/nature/no-latex .mplstyle files by path.

    The installed scienceplots (2.2.1) fails to import under matplotlib 3.11
    (it calls the removed ``plt.style.core``), so the style sheets are loaded
    directly from the package directory instead of registering by name.
    """
    spec = importlib.util.find_spec("scienceplots")
    styles = Path(spec.origin).parent / "styles"
    return [str(styles / "science.mplstyle"),
            str(styles / "journals" / "nature.mplstyle"),
            str(styles / "misc" / "no-latex.mplstyle")]


def apply_style():
    """Activate science+nature (no LaTeX) with Inter as the type family."""
    for fname in ("Inter-Regular.otf", "Inter-SemiBold.otf", "Inter-Bold.otf"):
        fpath = _INTER_DIR / fname
        if fpath.exists():
            fm.fontManager.addfont(str(fpath))
    plt.style.use(_style_files())
    plt.rcParams.update({
        "font.family": "Inter",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Inter",
        "mathtext.it": "Inter:italic",
        "mathtext.bf": "Inter:bold",
        "svg.fonttype": "none",
    })


def style_axes(*axes):
    """Keep only the left and bottom spines (drop top and right) on each Axes."""
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.tick_params(top=False, right=False, which="both")
