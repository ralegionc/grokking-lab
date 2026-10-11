"""The figure suite.

Every figure is built from one finished run and rendered once per theme. The
choice of form follows the job the data is doing: change over time gets a line,
magnitude over a grid gets a single-hue heatmap, signed quantities get the
diverging pair, and comparisons of a handful of conditions get bars.

Run it as a module:

    python -m grokking.figures --run runs/headline --out figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from grokking import style
from grokking.analysis import (
    Run,
    frequency_ablations,
    load_run,
    neuron_grid,
    phase_table,
    spectrum_over_time,
)
from grokking.fourier import (
    analyse_in_both_bases,
    circle_coordinates,
    frequency_pair_power,
    primitive_root,
)
from grokking.style import Theme


# ---------------------------------------------------------------------------
# helpers


def _footer(fig: Figure, run: Run, theme: Theme) -> None:
    cfg = run.config
    text = (
        f"{cfg.op} mod {cfg.p}  ·  {cfg.arch}, d_model {cfg.d_model}  ·  "
        f"train fraction {cfg.train_frac:g}  ·  weight decay {cfg.weight_decay:g}  ·  "
        f"seed {cfg.seed}"
    )
    fig.text(0.0, -0.015, text, fontsize=8, color=theme.text_muted, ha="left", va="top")


def _steps(run: Run) -> np.ndarray:
    """Training steps, with step 0 nudged to 1 so it survives a log axis."""
    step = run.history["step"].copy()
    step[step == 0] = 1
    return step


def _mark_event(ax, x: float, label: str, theme: Theme) -> None:
    """A dashed rule with its name written above the plot area."""
    ax.axvline(x, color=theme.text_muted, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        label,
        xy=(x, 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=theme.text_secondary,
        annotation_clip=False,
    )


def _model_phrase(run: Run) -> str:
    cfg = run.config
    if cfg.arch == "transformer":
        layers = "one layer" if cfg.n_layers == 1 else f"{cfg.n_layers} layers"
        return f"A {layers} transformer"
    return "A two layer network"


# ---------------------------------------------------------------------------
# 1. the curve everyone comes for


def fig_curves(run: Run, theme: Theme) -> Figure:
    history, step = run.history, _steps(run)
    phases = phase_table(run)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.8, 6.6), sharex=True, height_ratios=[1.2, 1]
    )

    for ax, keys, logy in (
        (top, ("train_acc", "val_acc"), False),
        (bottom, ("train_loss", "val_loss"), True),
    ):
        ends = []
        for i, (key, name) in enumerate(zip(keys, ("train", "held out"))):
            ax.plot(step, history[key], color=theme.series[i], label=name, zorder=3)
            ends.append((history[key][-1], name, theme.series[i]))
        if logy:
            ax.set_yscale("log")
            # Once both losses are around 1e-6 the model is exactly right and
            # the remaining movement is AdamW rattling around the minimum. Left
            # unbounded it is eight decades of noise that hides the part worth
            # seeing, so the axis stops at 1e-5 and the label says so.
            floor = 1e-5
            if min(history[key].min() for key in keys) < floor:
                ax.set_ylim(bottom=floor)
                ax.set_ylabel("cross entropy\n(axis stops at 1e-5)")
        else:
            style.clean_percent(ax)
        style.label_series_ends(ax, ends)

    top.set_ylabel("accuracy")
    style.log_steps(bottom, int(step[-1]))
    top.set_xscale("log")
    top.legend(loc="upper left", bbox_to_anchor=(0.015, 0.97))

    memorised, generalised = phases["memorised_at"], phases["generalised_at"]
    if memorised:
        _mark_event(top, max(memorised, 1), f"memorised, step {memorised:,}", theme)
    if generalised:
        _mark_event(top, generalised, f"generalised, step {generalised:,}", theme)
    if memorised and generalised:
        y = 0.44
        top.annotate(
            "",
            xy=(max(memorised, 1), y),
            xytext=(generalised, y),
            arrowprops=dict(arrowstyle="<->", color=theme.text_muted, linewidth=1.0),
        )
        top.text(
            np.sqrt(max(memorised, 1) * generalised),
            y + 0.03,
            f"{generalised - memorised:,} steps of apparently nothing",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=theme.text_secondary,
        )

    style.header(
        fig,
        "Memorisation is quick. Understanding takes its time.",
        f"{_model_phrase(run)} on {run.config.op} mod {run.p}, shown {run.summary['n_train']:,} of the "
        f"{run.summary['n_train'] + run.summary['n_val']:,} pairs and tested on the rest",
        theme,
    )
    fig.align_ylabels([top, bottom])
    _footer(fig, run, theme)
    return fig


# ---------------------------------------------------------------------------
# 2. the point of the repository


def fig_progress(run: Run, theme: Theme) -> Figure:
    history, step = run.history, _steps(run)
    phases = phase_table(run)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))

    memorised = max(phases["memorised_at"] or 1, 1)
    generalised = phases["generalised_at"] or int(step[-1])
    ax.axvspan(memorised, generalised, color=theme.band, zorder=0, linewidth=0)
    ax.text(
        np.sqrt(memorised * generalised),
        0.94,
        "the plateau",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color=theme.text_muted,
    )

    ends = []
    for key, name, colour in (
        ("val_acc", "held out accuracy", theme.series[1]),
        ("freq_concentration", "Fourier concentration", theme.series[2]),
    ):
        ax.plot(step, history[key], color=colour, label=name, zorder=3)
        ends.append((history[key][-1], name, colour))

    # The honest baseline is this run's own initialisation, not a flat spectrum:
    # the top five of forty eight noisy frequencies already sum to more than
    # five forty eighths, and quoting the ideal would flatter the measure.
    baseline = float(history["freq_concentration"][0])
    ax.axhline(baseline, color=theme.axis, linewidth=0.9, linestyle=":", zorder=1)
    ax.annotate(
        "where this run's random initialisation started",
        xy=(1.0, baseline),
        xytext=(4, -6),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color=theme.text_muted,
    )

    concentration = history["freq_concentration"]
    take_off = None
    for i, value in enumerate(concentration):
        if value > baseline + 0.12 and step[i] < generalised:
            take_off = i
            break
    if take_off is not None:
        x, y = step[take_off], concentration[take_off]
        ax.annotate(
            f"the circuit is already forming here.\nStep {int(x):,}, and held out accuracy\n"
            f"is still {history['val_acc'][take_off]:.0%}.",
            xy=(x, y),
            xytext=(max(x / 14, 1.4), min(y + 0.30, 0.86)),
            fontsize=9,
            color=theme.text_secondary,
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color=theme.text_muted,
                linewidth=0.9,
                shrinkB=5,
                connectionstyle="arc3,rad=-0.12",
            ),
        )

    style.clean_percent(ax)
    style.log_steps(ax, int(step[-1]))
    style.label_series_ends(ax, ends)
    ax.set_ylabel("fraction")
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.99))
    style.header(
        fig,
        "The plateau is not a plateau",
        "Share of the embedding's energy held by its five strongest frequencies, against held out accuracy",
        theme,
    )
    _footer(fig, run, theme)
    return fig


# ---------------------------------------------------------------------------
# 3. the spectrum over training


def fig_spectrum(run: Run, theme: Theme) -> Figure:
    power = spectrum_over_time(run)[:, 1:]  # drop the constant term
    steps = run.snapshot_steps
    fig, ax = plt.subplots(figsize=(8.8, 4.9))

    image = ax.imshow(
        power.T,
        aspect="auto",
        origin="lower",
        cmap=theme.cmap_sequential,
        interpolation="nearest",
        extent=(-0.5, len(steps) - 0.5, 0.5, power.shape[1] + 0.5),
        vmin=0.0,
        vmax=float(np.percentile(power, 99.7)),
    )

    ticks = np.unique(np.linspace(0, len(steps) - 1, 7).astype(int))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{steps[i]:,}" for i in ticks])
    ax.set_xlabel("optimiser step (snapshots are spaced logarithmically)")
    ax.set_ylabel("frequency")
    ax.grid(False)

    for freq in run.summary.get("key_frequencies", []):
        ax.annotate(
            str(freq),
            xy=(len(steps) - 0.5, freq),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=theme.text_secondary,
            va="center",
            annotation_clip=False,
        )

    bar = fig.colorbar(image, ax=ax, pad=0.055, fraction=0.035)
    bar.set_label("share of embedding power", fontsize=9, color=theme.text_secondary)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, color=theme.axis, labelcolor=theme.text_muted)

    style.header(
        fig,
        "A few frequencies win",
        "The embedding begins as noise spread over every frequency and ends as a handful of waves",
        theme,
    )
    _footer(fig, run, theme)
    return fig


# ---------------------------------------------------------------------------
# 4. the circle


def fig_circle(run: Run, theme: Theme) -> Figure:
    key_freqs = run.summary.get("key_frequencies") or [1]
    freq = int(key_freqs[0])
    steps, embeds = run.snapshot_steps, run.snapshot_embeds

    phases = phase_table(run)
    memorised = max(phases["memorised_at"] or 1, 1)
    generalised = phases["generalised_at"] or int(steps[-1])
    # Sample the moments that differ from each other: the start, the end of
    # memorisation, the middle of the plateau, the moment of generalisation, and
    # the end. Spacing these geometrically instead lands every panel before the
    # circuit exists and the figure shows four pictures of noise.
    wanted = [0, memorised, generalised // 2, generalised, int(steps[-1])]
    picks = sorted(dict.fromkeys(int(np.argmin(np.abs(steps - w))) for w in wanted))

    fig, axes = plt.subplots(1, len(picks), figsize=(2.15 * len(picks) + 0.4, 3.0))
    colours = theme.cmap_cyclic(np.linspace(0, 1, run.p, endpoint=False))

    for ax, index in zip(np.atleast_1d(axes), picks):
        coords = circle_coordinates(torch.from_numpy(embeds[index]), freq, run.p)
        radius = (np.abs(coords).max() or 1.0) * 1.2
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colours,
            s=22,
            linewidths=0.7,
            edgecolors=theme.surface,
            zorder=3,
        )
        for token in (0, run.p // 3, 2 * run.p // 3):
            ax.annotate(
                str(token),
                xy=coords[token],
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=7.5,
                color=theme.text_secondary,
            )
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(theme.grid)
        ax.set_title(f"step {steps[index]:,}", loc="left", fontsize=9.5)

    style.header(
        fig,
        f"The numbers 0 to {run.p - 1} arrange themselves on a circle",
        f"Each token's embedding projected onto the cosine and sine directions of frequency {freq}. "
        f"Neighbours on the circle differ by {freq} rather than by 1, because the angle is 2π·{freq}x/{run.p}.",
        theme,
    )
    _footer(fig, run, theme)
    return fig


# ---------------------------------------------------------------------------
# 5. the weight norm and the three phases


def fig_phases(run: Run, theme: Theme) -> Figure:
    history, step = run.history, _steps(run)
    phases = phase_table(run)
    memorised = max(phases["memorised_at"] or 1, 1)
    generalised = phases["generalised_at"] or int(step[-1])

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(step, history["weight_norm"], color=theme.series[3], zorder=3)

    spans = [
        (1, memorised, "memorising"),
        (memorised, generalised, "plateau"),
        (generalised, float(step[-1]), "cleaning up"),
    ]
    for i, (start, end, label) in enumerate(spans):
        if i % 2 == 1:
            ax.axvspan(start, end, color=theme.band, zorder=0, linewidth=0)
        if end > start:
            ax.text(
                np.sqrt(max(start, 1) * end),
                0.045,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=9,
                color=theme.text_secondary,
            )

    peak = int(np.argmax(history["weight_norm"]))
    ax.annotate(
        f"peak {history['weight_norm'][peak]:.0f} at step {int(history['step'][peak]):,}",
        xy=(step[peak], history["weight_norm"][peak]),
        xytext=(14, 12),
        textcoords="offset points",
        fontsize=9,
        color=theme.text_secondary,
        arrowprops=dict(arrowstyle="-", color=theme.text_muted, linewidth=0.9),
    )

    style.log_steps(ax, int(step[-1]))
    ax.set_ylabel("L2 norm of all parameters")
    style.header(
        fig,
        "Weight decay is the clock",
        "The memorised solution is large. The general one is small, and the optimiser is paid to prefer it.",
        theme,
    )
    _footer(fig, run, theme)
    return fig


# ---------------------------------------------------------------------------
# 6. the causal test


def fig_ablation(run: Run, theme: Theme) -> Figure:
    results = frequency_ablations(run)
    freqs = results[0].freqs
    named = ", ".join(str(f) for f in freqs[:-1]) + f" and {freqs[-1]}" if len(freqs) > 1 else str(freqs[0])
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    fig.subplots_adjust(left=0.27)

    y = np.arange(len(results))
    height = 0.33
    for i, (name, values) in enumerate(
        (
            ("train", [r.train_acc for r in results]),
            ("held out", [r.val_acc for r in results]),
        )
    ):
        offset = (-0.5 + i) * (height + 0.04)
        bars = ax.barh(
            y + offset,
            values,
            height=height,
            color=theme.series[i],
            edgecolor=theme.surface,
            linewidth=1.5,
            label=name,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                value + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1%}",
                va="center",
                fontsize=8.5,
                color=theme.text_secondary,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([r.label for r in results], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("accuracy after filtering the embedding")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")

    style.header(
        fig,
        "The waves are the mechanism, not a by-product",
        f"Keep only the key frequencies and nothing breaks. Remove only them and everything does. "
        f"Here they are {named}, out of {run.p // 2} available.",
        theme,
    )
    _footer(fig, run, theme)
    return fig


# ---------------------------------------------------------------------------
# 7. the neurons


def tuned_frequency(spectrum: np.ndarray) -> tuple[int, float]:
    """Which single frequency a neuron listens to, and how much of it that is.

    A neuron in one of these networks is not a clean product of two waves. Its
    pre-activation is a *sum* of a wave in a and a wave in b, and the
    nonlinearity then adds the cross term, so the same frequency f shows up in
    three cells: (f, 0) for the wave in a, (0, f) for the wave in b, and (f, f)
    for their product. Scoring those three together is what identifies the
    frequency the neuron is actually tuned to.
    """
    total = spectrum.sum() or 1.0
    scores = [
        (spectrum[f, 0] + spectrum[0, f] + spectrum[f, f]) / total
        for f in range(spectrum.shape[0])
    ]
    scores[0] = 0.0  # the constant is a bias, not a tuning
    freq = int(np.argmax(scores))
    return freq, float(scores[freq])


def fig_neurons(run: Run, theme: Theme, n_show: int = 4) -> Figure:
    grid = neuron_grid(run)  # (p, p, n_neurons)
    variance = grid.reshape(-1, grid.shape[-1]).var(axis=0)

    # Show one neuron per frequency rather than the four loudest, which in a
    # grokked network would all be tuned to the same wave and look identical.
    order = np.argsort(variance)[::-1]
    tuning_of = {}
    for neuron in order:
        values = grid[:, :, neuron]
        tuning_of[int(neuron)] = tuned_frequency(
            frequency_pair_power(values - values.mean())
        )

    picks: list[int] = []
    tunings: list[tuple[int, float]] = []
    seen: set[int] = set()
    for neuron in (int(n) for n in order):
        freq, share = tuning_of[neuron]
        if freq not in seen:
            seen.add(freq)
            picks.append(neuron)
            tunings.append((freq, share))
        if len(picks) == n_show:
            break
    for neuron in (int(n) for n in order):  # a memorising model has no structure
        if len(picks) >= n_show:
            break
        if neuron not in picks:
            picks.append(neuron)
            tunings.append(tuning_of[neuron])

    fig, axes = plt.subplots(2, n_show, figsize=(2.1 * n_show + 0.6, 5.4))
    for column, neuron in enumerate(picks):
        values = grid[:, :, neuron]
        centred = values - values.mean()
        limit = np.abs(centred).max() or 1.0
        axes[0, column].imshow(
            centred,
            cmap=theme.cmap_diverging,
            vmin=-limit,
            vmax=limit,
            origin="lower",
            interpolation="nearest",
        )
        axes[0, column].set_title(f"neuron {neuron}", loc="left", fontsize=9.5)

        spectrum = frequency_pair_power(centred)
        spectrum = spectrum / (spectrum.max() or 1.0)
        n = spectrum.shape[0]
        axes[1, column].imshow(
            spectrum,
            cmap=theme.cmap_sequential,
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            extent=(-0.5, n - 0.5, -0.5, n - 0.5),
        )
        freq, share = tunings[column]
        axes[1, column].set_xlabel(
            f"tuned to frequency {freq}\n{share:.0%} of the neuron's power",
            fontsize=8.5,
            color=theme.text_secondary,
        )
        axes[1, column].set_xticks([0, n - 1])
        axes[1, column].set_yticks([0, n - 1])
        axes[1, column].tick_params(labelsize=7.5)

    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat:
        ax.grid(False)
    axes[0, 0].set_ylabel("over the (a, b) table", fontsize=8.5)
    axes[1, 0].set_ylabel("frequency in a", fontsize=8.5)

    style.header(
        fig,
        "Each neuron listens to one frequency",
        "Top: the neuron's response over the whole table, which at these frequencies looks like interference. "
        "Bottom: the same response binned by frequency pair. The three bright cells are one wave in a, "
        "the same wave in b, and their product, which is what a nonlinearity applied to a sum of two waves gives.",
        theme,
    )
    _footer(fig, run, theme)
    return fig


# ---------------------------------------------------------------------------
# 8. what the answer looks like


def fig_logits(run: Run, theme: Theme, n_show: int = 3) -> Figure:
    """The score the model gives one answer, over every pair of inputs."""
    model, p = run.model(), run.p
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    tokens = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    with torch.no_grad():
        logits = model(tokens).reshape(p, p, p).numpy()

    targets = [0, p // 3, 2 * p // 3][:n_show]
    fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show + 0.6, 3.7))
    fig.subplots_adjust(bottom=0.24)
    for ax, c in zip(np.atleast_1d(axes), targets):
        surface = logits[:, :, c] - logits[:, :, c].mean()
        limit = np.abs(surface).max() or 1.0
        ax.imshow(
            surface,
            cmap=theme.cmap_diverging,
            vmin=-limit,
            vmax=limit,
            origin="lower",
            interpolation="nearest",
        )
        # State the separation rather than leaving it to the eye.
        on_line = np.zeros((p, p), dtype=bool)
        for row in range(p):
            on_line[row, (c - row) % p] = True
        ax.set_title(f"score for the answer {c}", loc="left", fontsize=9.5)
        ax.set_xticks([0, p - 1])
        ax.set_yticks([0, p - 1])
        ax.set_xticklabels(["0", str(p - 1)], fontsize=8)
        ax.set_yticklabels(["0", str(p - 1)], fontsize=8)
        on = int(round(float(surface[on_line].mean())))
        off = int(round(float(surface[~on_line].mean())))
        ax.set_xlabel(f"b\n{on:+d} on the line, {off:+d} off", fontsize=8.5)
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("a", fontsize=9)

    style.header(
        fig,
        "What having learned the rule looks like",
        f"Each panel is the model's score for one answer over all {p}×{p} input pairs, and the deep red "
        f"diagonal is exactly the set of pairs with a + b ≡ c.",
        theme,
    )
    _footer(fig, run, theme)
    return fig


# ---------------------------------------------------------------------------
# 9. the same measure, two bases, two tasks


def fig_two_bases(add_run: Run, div_run: Run, theme: Theme) -> Figure:
    """Why a flat spectrum is not evidence that nothing was learned."""
    p = add_run.p
    rows = [("a + b mod {}".format(p), add_run), ("a / b mod {}".format(p), div_run)]
    columns = [
        ("additive", "frequency over 0, 1, 2, ... , p-1"),
        ("multiplicative", "frequency over 1, g, g², ... , g^(p-2)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.4), sharey=True)
    for r, (task, run) in enumerate(rows):
        spectra = analyse_in_both_bases(run.model().embed.detach(), run.p)
        for c, (basis, _) in enumerate(columns):
            ax = axes[r, c]
            spectrum = spectra[basis]
            share = spectrum.power[1:] / max(spectrum.power[1:].sum(), 1e-12)
            keys = set(int(f) for f in spectrum.key_freqs)
            colours = [
                theme.series[0] if (i + 1) in keys else theme.grid
                for i in range(len(share))
            ]
            ax.bar(np.arange(1, len(share) + 1), share, color=colours, width=0.78, zorder=3)
            ax.set_ylim(0, max(0.45, float(share.max()) * 1.18))
            ax.text(
                0.97,
                0.92,
                f"concentration {spectrum.concentration:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                fontweight="semibold",
                color=theme.text_primary if spectrum.n_key else theme.text_muted,
            )
            if r == 0:
                ax.set_title(f"{basis} basis", loc="left", fontsize=10)
            if r == 1:
                ax.set_xlabel(columns[c][1], fontsize=8.5)
            if c == 0:
                ax.set_ylabel(f"{task}\nshare of power", fontsize=9)
            ax.grid(axis="x", visible=False)

    style.header(
        fig,
        "A flat spectrum means you are looking in the wrong basis",
        "The same measurement on two networks that both reached 100% on held out pairs. Addition puts its "
        "waves over the residues in their usual order; division puts them over the residues in the order "
        "1, g, g², ... , because a / b depends on b through its inverse. Neither network is less "
        "structured than the other.",
        theme,
    )
    cfg = div_run.config
    fig.text(
        0.0,
        -0.015,
        f"add and div mod {p}  ·  {cfg.arch}, d_model {cfg.d_model}  ·  "
        f"train fraction {cfg.train_frac:g}  ·  weight decay {cfg.weight_decay:g}  ·  "
        f"seed {cfg.seed}  ·  primitive root g = {primitive_root(p)}",
        fontsize=8,
        color=theme.text_muted,
        ha="left",
        va="top",
    )
    return fig


def fig_seed_spread(runs: list[Run], theme: Theme) -> Figure:
    """Every seed, overlaid, so the single run above is not taken on faith."""
    fig, ax = plt.subplots(figsize=(8.8, 4.9))

    series = [
        ("val_acc", "held out accuracy", theme.series[1]),
        ("freq_concentration", "Fourier concentration", theme.series[2]),
    ]
    grok_steps = []
    for run in runs:
        step = _steps(run)
        for key, name, colour in series:
            ax.plot(step, run.history[key], color=colour, linewidth=1.2, alpha=0.75, zorder=3)
        if run.summary.get("grok_step_95"):
            grok_steps.append(int(run.summary["grok_step_95"]))

    # One legend entry per quantity rather than one per run: the seeds are
    # replicates, so their identity carries no information worth a colour.
    handles = [
        plt.Line2D([], [], color=colour, linewidth=2, label=name)
        for _, name, colour in series
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.99))

    if grok_steps:
        low, high = min(grok_steps), max(grok_steps)
        ax.axvspan(low, high, color=theme.band, zorder=0, linewidth=0)
        ax.text(
            0.04,
            0.68,
            f"all {len(grok_steps)} seeds generalised between\n"
            f"step {low:,} and step {high:,}, the shaded band",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color=theme.text_secondary,
        )

    style.clean_percent(ax)
    style.log_steps(ax, int(max(_steps(run)[-1] for run in runs)))
    ax.set_ylabel("fraction")
    style.header(
        fig,
        f"The same thing happens for every seed",
        f"{len(runs)} runs that differ only in the random seed, which changes both the initial weights and "
        f"which half of the table is held out. The progress measure leads the accuracy in all of them.",
        theme,
    )
    cfg = runs[0].config
    fig.text(
        0.0,
        -0.015,
        f"{cfg.op} mod {cfg.p}  ·  {cfg.arch}, d_model {cfg.d_model}  ·  "
        f"train fraction {cfg.train_frac:g}  ·  weight decay {cfg.weight_decay:g}  ·  "
        f"seeds {', '.join(str(r.config.seed) for r in runs)}",
        fontsize=8,
        color=theme.text_muted,
        ha="left",
        va="top",
    )
    return fig


def render_seed_spread(sweep_dir: Path, out_dir: Path) -> list[Path]:
    directories = sorted(
        (d for d in Path(sweep_dir).iterdir() if d.is_dir() and d.name.startswith("seed_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not directories:
        raise SystemExit(f"no seed_* runs under {sweep_dir}")
    runs = [load_run(d) for d in directories]

    written = []
    for theme_name, theme in style.THEMES.items():
        target = out_dir if theme_name == "light" else out_dir / "dark"
        target.mkdir(parents=True, exist_ok=True)
        style.apply(theme)
        figure = fig_seed_spread(runs, theme)
        path = target / "10_seeds.png"
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
        print(f"wrote {path}")
    return written


def render_two_bases(add_dir: Path, div_dir: Path, out_dir: Path) -> list[Path]:
    add_run, div_run = load_run(add_dir), load_run(div_dir)
    written = []
    for theme_name, theme in style.THEMES.items():
        target = out_dir if theme_name == "light" else out_dir / "dark"
        target.mkdir(parents=True, exist_ok=True)
        style.apply(theme)
        figure = fig_two_bases(add_run, div_run, theme)
        path = target / "09_two_bases.png"
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
        print(f"wrote {path}")
    return written


# ---------------------------------------------------------------------------
# sweep figures, built from a list of run summaries


def fig_sweep(
    summaries: list[dict],
    theme: Theme,
    x_key: str,
    x_label: str,
    headline: str,
    subtitle: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    rows = sorted(summaries, key=lambda s: s[x_key])
    xs = [row[x_key] for row in rows]
    grok = [row["grok_step_95"] for row in rows]
    ceiling = max(row["steps"] for row in rows)

    solved = [(x, g) for x, g in zip(xs, grok) if g is not None]
    failed = [x for x, g in zip(xs, grok) if g is None]

    if solved:
        ax.plot(
            *zip(*solved), color=theme.series[0], marker="o", zorder=3, label="generalised"
        )
        for x, g in solved:
            ax.annotate(
                f"{g:,}",
                xy=(x, g),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=theme.text_secondary,
            )
    if failed:
        ax.scatter(
            failed,
            [ceiling] * len(failed),
            color=theme.series[1],
            marker="x",
            s=55,
            linewidths=2,
            zorder=3,
            label=f"never, within {ceiling:,} steps",
        )

    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("step at which held out accuracy passed 95%")
    ax.legend(loc="upper right")
    style.header(fig, headline, subtitle, theme)
    return fig


def render_sweep(path: Path, out_dir: Path) -> list[Path]:
    """Render one sweep JSON written by grokking.sweep."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    field = blob["field"]
    labels = {
        "train_frac": (
            "fraction of the table shown during training",
            "How little data is too little",
            "Below a threshold the model memorises and stops there, however long it is trained",
        ),
        "weight_decay": (
            "weight decay",
            "Without the pressure to be small, it never generalises",
            "Weight decay is what makes the general solution worth finding",
        ),
        "seed": ("seed", "Run to run variation", "The same settings, different initial weights"),
    }
    x_label, headline, subtitle = labels.get(
        field, (field, f"Sweeping {field}", "Step at which the model generalised")
    )

    written = []
    for theme_name, theme in style.THEMES.items():
        target = out_dir if theme_name == "light" else out_dir / "dark"
        target.mkdir(parents=True, exist_ok=True)
        style.apply(theme)
        figure = fig_sweep(blob["runs"], theme, field, x_label, headline, subtitle)
        out = target / f"sweep_{field}.png"
        figure.savefig(out)
        plt.close(figure)
        written.append(out)
        print(f"wrote {out}")
    return written


# ---------------------------------------------------------------------------

FIGURES = {
    "01_curves": fig_curves,
    "02_progress": fig_progress,
    "03_spectrum": fig_spectrum,
    "04_circle": fig_circle,
    "05_phases": fig_phases,
    "06_ablation": fig_ablation,
    "07_neurons": fig_neurons,
    "08_logits": fig_logits,
}


def render(run_dir: Path, out_dir: Path, only: list[str] | None = None) -> list[Path]:
    run = load_run(run_dir)
    written: list[Path] = []
    for theme_name, theme in style.THEMES.items():
        target = out_dir if theme_name == "light" else out_dir / "dark"
        target.mkdir(parents=True, exist_ok=True)
        for name, builder in FIGURES.items():
            if only and name not in only:
                continue
            style.apply(theme)
            figure = builder(run, theme)
            path = target / f"{name}.png"
            figure.savefig(path)
            plt.close(figure)
            written.append(path)
            print(f"wrote {path}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m grokking.figures", description="Render the figure suite."
    )
    parser.add_argument("--run", type=Path, default=Path("runs/headline"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--only", nargs="*", help="figure names to rebuild")
    parser.add_argument("--sweep", type=Path, help="render a sweep JSON instead")
    parser.add_argument(
        "--two-bases",
        nargs=2,
        type=Path,
        metavar=("ADD_RUN", "DIV_RUN"),
        dest="two_bases",
        help="render the two basis comparison from an addition and a division run",
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        metavar="SWEEP_DIR",
        help="render the seed spread from the seed_* runs in a sweep directory",
    )
    args = parser.parse_args(argv)

    if args.sweep:
        render_sweep(args.sweep, args.out)
    elif args.two_bases:
        render_two_bases(args.two_bases[0], args.two_bases[1], args.out)
    elif args.seeds:
        render_seed_spread(args.seeds, args.out)
    else:
        render(args.run, args.out, args.only)


if __name__ == "__main__":  # pragma: no cover
    main()
