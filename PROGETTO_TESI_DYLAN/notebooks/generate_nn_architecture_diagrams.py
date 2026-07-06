"""
Genera i diagrammi architetturali di Linear AE, Autoencoder Profondo (Deep AE) e VAE
descritti nel Capitolo 6 (Architetture dei Modelli e Setup Sperimentale) della tesi.

Le tre figure vengono salvate in ../tesi/Pictures/capitolo6/ ed embeddate nelle
Sezioni 6.2 (Linear AE), 6.3 (Deep AE) e 6.4 (VAE) di capitolo6.tex.

Esegui con l'interprete del venv di progetto:
    ../venv/Scripts/python.exe generate_nn_architecture_diagrams.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================================
# Palette categoriale (colorblind-safe, light mode) — stessa palette per le
# tre figure: ogni colore mantiene sempre lo stesso significato.
# ============================================================================
COLOR_IO = "#e1e0d9"       # layer di dati (input / output)
COLOR_IO_EDGE = "#898781"
COLOR_DENSE = "#2a78d6"    # layer denso con attivazione LeakyReLU
COLOR_LATENT = "#1baf7a"   # codice/bottleneck latente (AE) o mu (VAE)
COLOR_LOGVAR = "#eda100"   # log-varianza (solo VAE)
COLOR_Z = "#4a3aa7"        # campione latente z (solo VAE)

TEXT_DARK = "#0b0b0b"
TEXT_LIGHT = "#ffffff"
ARROW_COLOR = "#4a4a48"

FIGURES_FOLDER = Path("../tesi/Pictures/capitolo6")
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

D = 65703  # dimensione canonica dell'input/output (dominio Cholesky, Sezione 5.4)


# ============================================================================
# Helper di disegno
# ============================================================================
def layer_height(dim, h_min=0.55, h_max=3.3, dim_min=20, dim_max=D):
    """Altezza del box (scala log) proporzionale alla dimensione del layer."""
    log_d = np.log10(np.clip(dim, dim_min, dim_max))
    log_min, log_max = np.log10(dim_min), np.log10(dim_max)
    frac = (log_d - log_min) / (log_max - log_min)
    return h_min + frac * (h_max - h_min)


def draw_box(ax, x, y, w, h, kind_label, dim_label, color, text_color=TEXT_DARK):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.07",
        linewidth=1.3, edgecolor=COLOR_IO_EDGE if color == COLOR_IO else color,
        facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, kind_label, ha="center", va="center", fontsize=9,
             fontweight="bold", color=text_color, zorder=4)
    if dim_label:
        ax.text(x, y - h / 2 - 0.30, dim_label, ha="center", va="top",
                 fontsize=8.3, color="#2a2a28", zorder=4)


def draw_arrow(ax, xy0, xy1, label=None, label_offset=0.20, curve=0.0):
    connectionstyle = f"arc3,rad={curve}" if curve else "arc3,rad=0"
    arrow = FancyArrowPatch(
        xy0, xy1, arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
        color=ARROW_COLOR, shrinkA=3, shrinkB=3, zorder=2,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    if label:
        xm, ym = (xy0[0] + xy1[0]) / 2, (xy0[1] + xy1[1]) / 2
        ax.text(xm, ym + label_offset, label, ha="center",
                 va="bottom" if label_offset >= 0 else "top",
                 fontsize=7.3, color=ARROW_COLOR, style="italic", zorder=4)


def legend_patch(color, label):
    return mpatches.Patch(facecolor=color, edgecolor=COLOR_IO_EDGE if color == COLOR_IO else color, label=label)


def finalize(fig, ax, save_path, title, legend_handles, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=16)
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.16),
               ncol=len(legend_handles), frameon=False, fontsize=8.6, handlelength=1.2, handleheight=1.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Salvato: {save_path.resolve()}")
    plt.close(fig)


# ============================================================================
# 1. Linear AE — singolo layer denso lineare, senza bias, per lato
# ============================================================================
def plot_linear_ae():
    fig, ax = plt.subplots(figsize=(9.6, 3.4))

    xs = [0.0, 3.2, 6.4]
    dims = [D, 20, D]  # 20 = valore rappresentativo per la sola scala visiva del bottleneck K
    kinds = ["Input", "$K$", "Output"]
    dim_labels = [f"$D={D}$", "Spazio Latente", f"$D={D}$"]
    colors = [COLOR_IO, COLOR_LATENT, COLOR_IO]
    heights = [layer_height(d) for d in dims]

    for x, h, kind, dl, c in zip(xs, heights, kinds, dim_labels, colors):
        draw_box(ax, x, 0, 1.35, h, kind, dl, c)

    draw_arrow(ax, (xs[0] + 0.72, 0), (xs[1] - 0.72, 0), label="Encoder\n(Dense lineare, no bias)", label_offset=0.55)
    draw_arrow(ax, (xs[1] + 0.72, 0), (xs[2] - 0.72, 0), label="Decoder\n(Dense lineare, no bias)", label_offset=0.55)

    legend_handles = [
        legend_patch(COLOR_IO, "Dati (input/output)"),
        legend_patch(COLOR_LATENT, "Spazio latente ($K \\in \\{3,10,20,100\\}$)"),
    ]
    finalize(fig, ax, FIGURES_FOLDER / "fig_architettura_linear_ae.png",
             "Autoencoder Lineare (Linear AE)", legend_handles,
             xlim=(-1.1, 7.5), ylim=(-2.6, 2.2))


# ============================================================================
# 2. Autoencoder Profondo (Deep AE) — imbuto simmetrico con LeakyReLU
# ============================================================================
def plot_deep_ae():
    fig, ax = plt.subplots(figsize=(15.5, 4.3))

    dx = 1.65
    xs = [i * dx for i in range(9)]
    dims = [D, 2048, 1024, 512, 20, 512, 1024, 2048, D]
    kinds = ["Input", "Dense", "Dense", "Dense", "$K$",
             "Dense", "Dense", "Dense", "Output"]
    dim_labels = [f"$D={D}$", "2048", "1024", "512", "Spazio Latente",
                  "512", "1024", "2048", f"$D={D}$"]
    colors = [COLOR_IO, COLOR_DENSE, COLOR_DENSE, COLOR_DENSE, COLOR_LATENT,
              COLOR_DENSE, COLOR_DENSE, COLOR_DENSE, COLOR_IO]
    heights = [layer_height(d) for d in dims]

    for x, h, kind, dl, c in zip(xs, heights, kinds, dim_labels, colors):
        text_color = TEXT_LIGHT if c == COLOR_DENSE else TEXT_DARK
        draw_box(ax, x, 0, 1.15, h, kind, dl, c, text_color=text_color)

    arrow_labels = ["LeakyReLU"] * 7 + ["Lineare\n(no attivazione)"]
    # Offset sopra/sotto per ciascuna freccia: alternati per leggibilità, ma le due
    # frecce adiacenti al box "K" (indici 2 e 3) restano sopra per non sovrapporsi
    # alla didascalia "Spazio Latente" sotto quel box.
    offsets = [0.20, -0.55, 0.20, 0.20, 0.20, -0.55, 0.20, -0.55]
    for i in range(8):
        x0 = xs[i] + 0.62
        x1 = xs[i + 1] - 0.62
        draw_arrow(ax, (x0, 0), (x1, 0), label=arrow_labels[i], label_offset=offsets[i])

    legend_handles = [
        legend_patch(COLOR_IO, "Dati (input/output)"),
        legend_patch(COLOR_DENSE, "Layer denso (LeakyReLU, $\\alpha=0.01$)"),
        legend_patch(COLOR_LATENT, "Spazio latente ($K$)"),
    ]
    finalize(fig, ax, FIGURES_FOLDER / "fig_architettura_deep_ae.png",
             "Autoencoder Profondo (Deep AE)", legend_handles,
             xlim=(-1.1, xs[-1] + 1.3), ylim=(-2.6, 2.4))


# ============================================================================
# 3. VAE — encoder a imbuto condiviso col Deep AE, biforcazione mu/log-sigma^2
#    e reparameterization trick
# ============================================================================
def plot_vae():
    fig, ax = plt.subplots(figsize=(16.5, 5.6))

    dx = 1.65
    # encoder condiviso: Input, 2048, 1024, 512
    xs_enc = [i * dx for i in range(4)]
    dims_enc = [D, 2048, 1024, 512]
    kinds_enc = ["Input", "Dense", "Dense", "Dense"]
    dim_labels_enc = [f"$D={D}$", "2048", "1024", "512"]

    x_branch = xs_enc[-1] + dx        # colonna di mu e log(sigma^2)
    x_merge = x_branch + dx            # colonna di z (reparameterization)
    xs_dec = [x_merge + dx * (i + 1) for i in range(4)]  # 512,1024,2048,Output
    dims_dec = [512, 1024, 2048, D]
    kinds_dec = ["Dense", "Dense", "Dense", "Output"]
    dim_labels_dec = ["512", "1024", "2048", f"$D={D}$"]

    y_branch = 1.15  # offset verticale di mu (sopra) e log-sigma^2 (sotto)

    # ---- encoder condiviso ----
    for x, dim, kind, dl in zip(xs_enc, dims_enc, kinds_enc, dim_labels_enc):
        c = COLOR_IO if kind == "Input" else COLOR_DENSE
        tc = TEXT_LIGHT if c == COLOR_DENSE else TEXT_DARK
        draw_box(ax, x, 0, 1.15, layer_height(dim), kind, dl, c, text_color=tc)
    for i in range(3):
        draw_arrow(ax, (xs_enc[i] + 0.62, 0), (xs_enc[i + 1] - 0.62, 0),
                    label="LeakyReLU", label_offset=0.20 if i % 2 == 0 else -0.55)

    # ---- biforcazione: mu (sopra) e log(sigma^2) (sotto) ----
    h_branch = layer_height(20)
    draw_box(ax, x_branch, y_branch, 1.05, h_branch, "$\\mu$", "$K=20$", COLOR_LATENT)
    draw_box(ax, x_branch, -y_branch, 1.05, h_branch, "$\\log \\sigma^2$", "$K=20$", COLOR_LOGVAR)
    draw_arrow(ax, (xs_enc[-1] + 0.62, 0.10), (x_branch - 0.58, y_branch - 0.05))
    draw_arrow(ax, (xs_enc[-1] + 0.62, -0.10), (x_branch - 0.58, -y_branch + 0.05))

    # ---- reparameterization trick: z = mu + sigma * epsilon ----
    draw_box(ax, x_merge, 0, 1.15, layer_height(20), "$z$", "$K=20$", COLOR_Z, text_color=TEXT_LIGHT)
    draw_arrow(ax, (x_branch + 0.58, y_branch - 0.05), (x_merge - 0.62, 0.10))
    draw_arrow(ax, (x_branch + 0.58, -y_branch + 0.05), (x_merge - 0.62, -0.10))
    ax.text(x_merge, y_branch + 0.55, r"$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$"
            "\n" r"$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$",
            ha="center", va="bottom", fontsize=8.6, color="#2a2a28")

    # ---- decoder ----
    prev_x, prev_dim = x_merge, 20
    dec_x_all = [x_merge] + xs_dec
    dec_dim_all = [20] + dims_dec
    for i in range(4):
        x, dim, kind, dl = xs_dec[i], dims_dec[i], kinds_dec[i], dim_labels_dec[i]
        c = COLOR_IO if kind == "Output" else COLOR_DENSE
        tc = TEXT_LIGHT if c == COLOR_DENSE else TEXT_DARK
        draw_box(ax, x, 0, 1.15, layer_height(dim), kind, dl, c, text_color=tc)
        label = "LeakyReLU" if i < 3 else "Lineare\n(no attivazione)"
        off = 0.20 if i % 2 == 0 else -0.55
        draw_arrow(ax, (dec_x_all[i] + (0.58 if i == 0 else 0.62), 0),
                   (x - 0.62, 0), label=label, label_offset=off)

    legend_handles = [
        legend_patch(COLOR_IO, "Dati (input/output)"),
        legend_patch(COLOR_DENSE, "Layer denso (LeakyReLU)"),
        legend_patch(COLOR_LATENT, "$\\mu$ (media)"),
        legend_patch(COLOR_LOGVAR, "$\\log \\sigma^2$ (log-varianza)"),
        legend_patch(COLOR_Z, "$z$ campionato (reparameterization trick)"),
    ]
    finalize(fig, ax, FIGURES_FOLDER / "fig_architettura_vae.png",
             "Variational Autoencoder (VAE)", legend_handles,
             xlim=(-1.1, xs_dec[-1] + 1.3), ylim=(-2.9, 2.9))


if __name__ == "__main__":
    plot_linear_ae()
    plot_deep_ae()
    plot_vae()
    print("\nTutti i diagrammi architetturali sono stati generati con successo.")
