"""One visual system for every figure in the repository.

The palette is the validated default from the data visualisation reference: the
same eight hues stepped separately for a light and a dark surface, rather than
one palette flipped. Every figure is rendered twice so that the README can hand
the reader whichever one matches their GitHub theme.

Rules the figures keep to, and the reason for each:

* never two y scales on one plot, so panels stack instead
* categorical hues are assigned in a fixed order and never cycled
* sequential data gets one hue running light to dark, signed data gets the
  blue/red diverging pair with a grey midpoint, and neither gets a rainbow
* every series is labelled at its own end as well as in the legend, so identity
  never rests on colour alone
"""

from __future__ import annotations

from dataclasses import dataclass, field

import textwrap

import matplotlib as mpl
import numpy as np
from matplotlib.transforms import offset_copy
from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    panel: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    sequential: tuple[str, ...]
    diverging: tuple[str, ...]
    band: str
    annotation: str = field(default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "annotation", self.annotation or self.text_secondary)

    @property
    def cmap_sequential(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            f"{self.name}_seq", list(self.sequential)
        )

    @property
    def cmap_diverging(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            f"{self.name}_div", list(self.diverging)
        )

    @property
    def cmap_cyclic(self) -> LinearSegmentedColormap:
        """For colouring tokens by their residue, where 0 and p - 1 are adjacent."""
        stops = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#2a78d6"]
        return LinearSegmentedColormap.from_list(f"{self.name}_cyc", stops)


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    panel="#ffffff",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    text_muted="#8a8880",
    grid="#e7e6e1",
    axis="#c9c7c0",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e34948"),
    sequential=("#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"),
    diverging=("#104281", "#3987e5", "#cde2fb", "#f0efec", "#f6c9c8", "#e34948", "#8e2020"),
    band="#f0efec",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    panel="#111110",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    text_muted="#8a8880",
    grid="#2e2e2b",
    axis="#4a4a46",
    series=("#3987e5", "#d95926", "#199e70", "#9085e9", "#c98500", "#e66767"),
    sequential=("#0d1f33", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"),
    diverging=("#cde2fb", "#86b6ef", "#3987e5", "#383835", "#e66767", "#e34948", "#a32e2d"),
    band="#262623",
)

THEMES = {"light": LIGHT, "dark": DARK}


def apply(theme: Theme) -> None:
    """Install a theme as the active matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "figure.edgecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "savefig.edgecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "axes.edgecolor": theme.axis,
            "axes.labelcolor": theme.text_secondary,
            "axes.titlecolor": theme.text_primary,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "grid.color": theme.grid,
            "grid.linewidth": 0.7,
            "xtick.color": theme.text_muted,
            "ytick.color": theme.text_muted,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "text.color": theme.text_primary,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": theme.text_secondary,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.markersize": 4.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
            "font.size": 10,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
        }
    )


HEADLINE_SIZE = 12.5
SUBTITLE_SIZE = 9.5


def header(fig, headline: str, subtitle: str | None, theme: Theme) -> None:
    """A heavy headline with a quiet sentence under it, above the plot area.

    The block measures itself: the subtitle is wrapped to the figure's width and
    the axes are then pushed down by exactly as many points as the text needs.
    Hand-tuned margins go wrong the moment a caption gets one word longer.
    """
    wrap_at = max(int(fig.get_figwidth() * 13), 40)
    lines = textwrap.wrap(subtitle, wrap_at) if subtitle else []

    height_points = 21 + SUBTITLE_SIZE * 1.35 * len(lines) + 14
    top = 1.0 - height_points / (fig.get_figheight() * 72.0)
    fig.subplots_adjust(top=min(max(top, 0.55), 0.93))

    fig.text(
        0.0,
        1.0,
        headline,
        ha="left",
        va="top",
        fontsize=HEADLINE_SIZE,
        fontweight="semibold",
        color=theme.text_primary,
    )
    if lines:
        shifted = offset_copy(fig.transFigure, fig=fig, x=0, y=-21, units="points")
        fig.text(
            0.0,
            1.0,
            "\n".join(lines),
            transform=shifted,
            ha="left",
            va="top",
            fontsize=SUBTITLE_SIZE,
            linespacing=1.35,
            color=theme.text_secondary,
        )


def direct_label(
    ax, x, y, text: str, color: str, dx: float = 6.0, dy: float = 0.0, **kwargs
) -> None:
    """Put a series name next to the series, not only in the legend."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="semibold",
        va="center",
        **kwargs,
    )


def label_series_ends(ax, entries: list[tuple[float, str, str]], min_gap: float = 0.075):
    """Name each line at its own right hand end, nudged apart when they collide.

    Two series that both finish at 100% would otherwise print their labels on
    top of each other, which is the usual way a direct label stops being useful.
    """
    ordered = sorted(entries, key=lambda item: item[0], reverse=True)
    inverse = ax.transAxes.inverted()
    x = ax.get_xlim()[1]
    placed: list[float] = []
    for value, text, colour in ordered:
        # Go through the data transform so that a log axis places labels where
        # the line actually is rather than where linear arithmetic says.
        position = float(inverse.transform(ax.transData.transform((x, value)))[1])
        for taken in placed:
            if abs(position - taken) < min_gap:
                position = taken - min_gap
        position = min(max(position, 0.0), 1.0)
        placed.append(position)
        ax.annotate(
            text,
            xy=(1.0, position),
            xycoords="axes fraction",
            xytext=(8, 0),
            textcoords="offset points",
            color=colour,
            fontsize=9,
            fontweight="semibold",
            va="center",
            annotation_clip=False,
        )


def phase_band(ax, start: float, end: float, label: str, theme: Theme, y: float = 0.97):
    """Shade a stretch of training and name it."""
    ax.axvspan(start, end, color=theme.band, zorder=0, linewidth=0)
    centre = np.sqrt(start * end) if start > 0 and ax.get_xscale() == "log" else (start + end) / 2
    ax.text(
        centre,
        y,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.5,
        color=theme.text_muted,
    )


def log_steps(ax, max_step: int) -> None:
    """Training steps belong on a log axis, starting at 1 rather than 0."""
    ax.set_xscale("log")
    ax.set_xlim(1, max_step)
    ax.set_xlabel("optimiser step")


def clean_percent(ax) -> None:
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
