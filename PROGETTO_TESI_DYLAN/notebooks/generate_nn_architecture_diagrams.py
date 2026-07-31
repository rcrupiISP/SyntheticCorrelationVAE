"""
Genera i diagrammi architetturali di Linear AE, Autoencoder Profondo (Deep AE) e VAE
descritti nel Capitolo 6 (Architetture dei Modelli e Setup Sperimentale) della tesi.

Le reti sono disegnate a neuroni (pallini) e sinapsi (linee) con la libreria `nnv`
(https://pypi.org/project/nnv/), sopra la quale vengono aggiunte le annotazioni
necessarie alla tesi: dimensioni reali dei layer, funzioni di attivazione, legenda.

Le tre figure vengono salvate in ../tesi/Pictures/capitolo6/ ed embeddate nelle
Sezioni 6.2 (Linear AE), 6.3 (Deep AE) e 6.4 (VAE) di capitolo6.tex.

Dipendenza: `pip install nnv` (testato con nnv 0.0.5).

Esegui con l'interprete del venv di progetto:
    ../venv/Scripts/python.exe generate_nn_architecture_diagrams.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # generazione di file, nessuna finestra interattiva
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nnv import NNV
# `NNV` sa disegnare solo catene lineari di layer: la biforcazione mu / log(sigma^2)
# del VAE viene costruita posizionando a mano le colonne con la primitiva `Layer`
# della stessa libreria, cosi' il rendering resta identico a quello delle altre figure.
from nnv.nnv import Layer

# ============================================================================
# Palette categoriale (colorblind-safe, light mode) — stessa palette per le
# tre figure: ogni colore mantiene sempre lo stesso significato.
# ============================================================================
COLOR_IO = "#b9b7ae"       # neuroni di dati (input / output)
COLOR_DENSE = "#2a78d6"    # layer denso con attivazione LeakyReLU
COLOR_LATENT = "#1baf7a"   # codice/bottleneck latente (AE) o mu (VAE)
COLOR_LOGVAR = "#eda100"   # log-varianza (solo VAE)
COLOR_Z = "#4a3aa7"        # campione latente z (solo VAE)
COLOR_DOTS = "#9a988f"     # tre puntini di troncamento (layer troppo grandi da disegnare)

EDGE_COLOR = "#cfcec9"     # sinapsi
EDGE_WIDTH = 0.6
TEXT = "#2a2a28"
ACT_COLOR = "#4a4a48"

FIGURES_FOLDER = Path(__file__).resolve().parent / ".." / "tesi" / "Pictures" / "capitolo6"
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

D = 65703  # dimensione canonica dell'input/output (dominio Cholesky, Sezione 5.4)

# Le dimensioni dei layer sono etichettate come spazi vettoriali (R^n), coerentemente
# con la notazione usata nel testo del Capitolo 6.
LABEL_IO = f"$\\mathbb{{R}}^{{D}}$\n($D={D}$)"

# ============================================================================
# Geometria del disegno (unita' arbitrarie, l'asse e' in aspect ratio 1:1).
# Il numero di pallini per colonna e' simbolico: la dimensione reale del layer
# e' scritta sotto ogni colonna. I tre puntini compaiono solo su Input/Output,
# dove D = 65703 rende impossibile qualunque rappresentazione fedele.
# ============================================================================
NODE_RADIUS = 18
SPACING_NODES = 10          # spazio verticale fra i neuroni di una colonna
SPACING_LAYER = 155         # spazio orizzontale fra colonne consecutive
MAX_NODES_VISIBLE = 12      # oltre questa soglia nnv inserisce i tre puntini
# Unita' di disegno per pollice. Non cambia le proporzioni della figura (e quindi
# l'ingombro in pagina), ma solo la sua larghezza in pollici: alzandola, i font
# — misurati in punti — risultano piu' grandi rispetto al disegno una volta che
# LaTeX riscala la figura a \textwidth.
SCALE = 155.0


# ============================================================================
# Helper di disegno
# ============================================================================
def fix_dots(layer, delta=None, color=COLOR_DOTS):
    """Riallinea e colora i tre puntini di troncamento.

    Workaround per un bug di nnv 0.0.5: `Layer.set_top()` trasla i neuroni ma
    dimentica `self.three_dots`, che resterebbe alla quota originale.
    """
    if layer.three_dots is None:
        return
    if delta:
        layer.three_dots.set_top(layer.three_dots.top + delta)
    for node in layer.three_dots.nodes:
        node.color = color


def set_top_fixed(layer, top):
    """`Layer.set_top()` con i tre puntini trascinati insieme ai neuroni."""
    delta = top - layer.top
    for node in layer.nodes:
        node.move_y(delta)
    fix_dots(layer, delta)
    layer.top += delta
    layer.bottom += delta


def bounds(layers):
    """Estremi (sinistra, destra, alto, basso) dell'insieme di colonne."""
    return (min(l.get_left() for l in layers), max(l.get_right() for l in layers),
            max(l.top for l in layers), min(l.get_bottom() for l in layers))


def legend_dot(color, label):
    return Line2D([], [], marker="o", linestyle="none", markersize=9,
                  markerfacecolor=color, markeredgecolor=color, label=label)


def annotate_columns(ax, layers, names, activations, dim_labels,
                     y_name, y_activation, y_dim):
    """Etichetta una colonna: nome del layer e attivazione sopra, dimensione sotto.

    L'attivazione e' scritta insieme al layer che la applica (e non a meta' fra
    due colonne): cosi' e' immediato capire che, ad esempio, il layer da 2048
    unita' e' un layer denso seguito da LeakyReLU.
    """
    for layer, name, activation, dim in zip(layers, names, activations, dim_labels):
        x = layer.get_center_x()
        if name:
            ax.text(x, y_name, name, ha="center", va="bottom", fontsize=12.5,
                    fontweight="bold", color=TEXT)
        if activation:
            ax.text(x, y_activation, activation, ha="center", va="bottom",
                    fontsize=10, color=ACT_COLOR)
        if dim:
            ax.text(x, y_dim, dim, ha="center", va="top", fontsize=10.5, color=TEXT)


def finalize(fig, ax, save_path, title, legend_handles, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    # aspect="equal" + figsize proporzionale alla finestra dati: evita che
    # bbox_inches="tight" ritagli lasciando fasce bianche laterali.
    fig.set_size_inches((xlim[1] - xlim[0]) / SCALE, (ylim[1] - ylim[0]) / SCALE)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.09),
              ncol=len(legend_handles), frameon=False, fontsize=9.5)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Salvato: {save_path.resolve()}")
    plt.close(fig)


def build_chain(units, colors, spacing_layer=SPACING_LAYER):
    """Rete a catena lineare costruita con `NNV`, pronta per le annotazioni."""
    layers_list = [{"title": None, "units": u, "color": c,
                    "edges_color": EDGE_COLOR, "edges_width": EDGE_WIDTH}
                   for u, c in zip(units, colors)]
    net = NNV(layers_list, spacing_layer=spacing_layer, spacing_nodes=SPACING_NODES,
              max_num_nodes_visible=MAX_NODES_VISIBLE, node_radius=NODE_RADIUS)
    for layer in net.layers:   # NNV ha gia' centrato le colonne: recupera i puntini
        fix_dots(layer)
    fig, ax = net.render(do_not_show=True)
    return net, fig, ax


def plot_chain(save_name, units, names, activations, dim_labels, colors,
               title, legend_handles, spacing_layer=SPACING_LAYER):
    net, fig, ax = build_chain(units, colors, spacing_layer)

    left, right, top, bottom = bounds(net.layers)
    y_activation, y_name, y_dim = top + 45, top + 115, bottom - 45
    annotate_columns(ax, net.layers, names, activations, dim_labels,
                     y_name, y_activation, y_dim)

    finalize(fig, ax, FIGURES_FOLDER / save_name, title, legend_handles,
             xlim=(left - 70, right + 70), ylim=(y_dim - 100, y_name + 65))


# ============================================================================
# 1. Linear AE — singolo layer denso lineare, senza bias, per lato
# ============================================================================
def plot_linear_ae():
    # spacing_layer maggiorato: con sole tre colonne servono per non ottenere
    # una figura quasi quadrata, sproporzionata rispetto alle altre due.
    plot_chain(
        "fig_architettura_linear_ae.png",
        units=[200, 3, 200],          # 200 = "molti", troncato ai tre puntini
        names=["Input", "Spazio latente $K$", "Output"],
        activations=[None, "Encoder: Dense lineare, no bias",
                     "Decoder: Dense lineare, no bias"],
        dim_labels=[LABEL_IO,
                    "$\\mathbb{R}^{K}$\n($K \\in \\{3,10,20,100\\}$)",
                    LABEL_IO],
        colors=[COLOR_IO, COLOR_LATENT, COLOR_IO],
        title="Autoencoder Lineare (Linear AE)",
        legend_handles=[
            legend_dot(COLOR_IO, "Dati (input/output)"),
            legend_dot(COLOR_LATENT, "Spazio latente ($K$)"),
        ],
        spacing_layer=620,
    )


# ============================================================================
# 2. Autoencoder Profondo (Deep AE) — imbuto simmetrico con LeakyReLU
# ============================================================================
def plot_deep_ae():
    plot_chain(
        "fig_architettura_deep_ae.png",
        units=[200, 9, 7, 5, 3, 5, 7, 9, 200],
        names=["Input", "Dense", "Dense", "Dense", "$K$",
               "Dense", "Dense", "Dense", "Output"],
        # L'attivazione e' quella applicata dal layer stesso sulle proprie unita'.
        # Nota: l'ultimo Linear dell'Encoder (512 -> K) e quello del Decoder
        # (2048 -> D) non hanno attivazione (cfr. 06_AE_cholesky.ipynb, il ciclo
        # aggiunge LeakyReLU solo per i < len(dimensions) - 2).
        activations=[None] + ["LeakyReLU"] * 3 + ["Dense\nlineare"] +
                    ["LeakyReLU"] * 3 + ["Dense\nlineare"],
        dim_labels=[LABEL_IO, "$\\mathbb{R}^{2048}$", "$\\mathbb{R}^{1024}$",
                    "$\\mathbb{R}^{512}$", "$\\mathbb{R}^{K}$",
                    "$\\mathbb{R}^{512}$", "$\\mathbb{R}^{1024}$",
                    "$\\mathbb{R}^{2048}$", LABEL_IO],
        colors=[COLOR_IO] + [COLOR_DENSE] * 3 + [COLOR_LATENT] +
               [COLOR_DENSE] * 3 + [COLOR_IO],
        title="Autoencoder Profondo (Deep AE)",
        legend_handles=[
            legend_dot(COLOR_IO, "Dati (input/output)"),
            legend_dot(COLOR_DENSE, "Layer denso (LeakyReLU, $\\alpha=0.01$)"),
            legend_dot(COLOR_LATENT, "Spazio latente ($K$)"),
        ],
    )


# ============================================================================
# 3. VAE — encoder a imbuto condiviso col Deep AE, biforcazione mu/log-sigma^2
#    e reparameterization trick
# ============================================================================
def plot_vae():
    fig, ax = plt.subplots()
    pitch = 2 * NODE_RADIUS + SPACING_LAYER
    y_branch = 155  # offset verticale di mu (sopra) e log-sigma^2 (sotto)

    def column(index, units, color, max_visible=None, y_offset=0.0):
        layer = Layer(left=index * pitch, num_nodes=units, node_radius=NODE_RADIUS,
                      title=None, spacing_nodes=SPACING_NODES,
                      max_num_nodes_visible=max_visible, color=color,
                      edges_color=EDGE_COLOR, edges_width=EDGE_WIDTH)
        set_top_fixed(layer, layer.get_height() / 2 + y_offset)
        return layer

    # ---- encoder condiviso: Input, 2048, 1024, 512 ----
    encoder = [column(0, 200, COLOR_IO, max_visible=MAX_NODES_VISIBLE),
               column(1, 9, COLOR_DENSE),
               column(2, 7, COLOR_DENSE),
               column(3, 5, COLOR_DENSE)]
    # ---- biforcazione: mu (sopra) e log(sigma^2) (sotto) ----
    mu = column(4, 3, COLOR_LATENT, y_offset=+y_branch)
    logvar = column(4, 3, COLOR_LOGVAR, y_offset=-y_branch)
    # ---- reparameterization trick: z = mu + sigma * epsilon ----
    z = column(5, 3, COLOR_Z)
    # ---- decoder speculare: 512, 1024, 2048, Output ----
    decoder = [column(6, 5, COLOR_DENSE),
               column(7, 7, COLOR_DENSE),
               column(8, 9, COLOR_DENSE),
               column(9, 200, COLOR_IO, max_visible=MAX_NODES_VISIBLE)]

    all_layers = encoder + [mu, logvar, z] + decoder
    for layer in all_layers:
        layer.render(ax)

    for left_layer, right_layer in zip(encoder, encoder[1:]):
        left_layer.fully_connect(right_layer, ax)
    encoder[-1].fully_connect(mu, ax)
    encoder[-1].fully_connect(logvar, ax)
    mu.fully_connect(z, ax)
    logvar.fully_connect(z, ax)
    z.fully_connect(decoder[0], ax)
    for left_layer, right_layer in zip(decoder, decoder[1:]):
        left_layer.fully_connect(right_layer, ax)

    left, right, top, bottom = bounds(all_layers)
    y_activation, y_name, y_dim = top + 45, top + 115, bottom - 45

    # colonne allineate sull'asse centrale: mu e log-sigma^2 sono etichettati a parte
    axis_layers = encoder + [z] + decoder
    annotate_columns(
        ax, axis_layers,
        ["Input", "Dense", "Dense", "Dense", "$\\mathbf{z}$",
         "Dense", "Dense", "Dense", "Output"],
        [None, "LeakyReLU", "LeakyReLU", "LeakyReLU", "campionamento\nstocastico",
         "LeakyReLU", "LeakyReLU", "LeakyReLU", "Dense\nlineare"],
        [LABEL_IO, "$\\mathbb{R}^{2048}$", "$\\mathbb{R}^{1024}$",
         "$\\mathbb{R}^{512}$", "$\\mathbb{R}^{K}$\n($K=20$)",
         "$\\mathbb{R}^{512}$", "$\\mathbb{R}^{1024}$",
         "$\\mathbb{R}^{2048}$", LABEL_IO],
        y_name, y_activation, y_dim,
    )

    # teste gemelle dell'encoder: due layer densi lineari paralleli
    ax.text(mu.get_center_x(), mu.top + 78, "$\\boldsymbol{\\mu}$",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=TEXT)
    ax.text(mu.get_center_x(), mu.top + 22, "Dense lineare, $\\mathbb{R}^{20}$",
            ha="center", va="bottom", fontsize=10, color=ACT_COLOR)
    ax.text(logvar.get_center_x(), logvar.get_bottom() - 22,
            "$\\log \\boldsymbol{\\sigma}^2$",
            ha="center", va="top", fontsize=12.5, fontweight="bold", color=TEXT)
    ax.text(logvar.get_center_x(), logvar.get_bottom() - 78,
            "Dense lineare, $\\mathbb{R}^{20}$",
            ha="center", va="top", fontsize=10, color=ACT_COLOR)

    ax.text(z.get_center_x(), y_name + 62,
            r"$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$,"
            r"  $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$",
            ha="center", va="bottom", fontsize=11, color=TEXT)

    finalize(fig, ax, FIGURES_FOLDER / "fig_architettura_vae.png",
             "Variational Autoencoder (VAE)",
             [legend_dot(COLOR_IO, "Dati (input/output)"),
              legend_dot(COLOR_DENSE, "Layer denso (LeakyReLU)"),
              legend_dot(COLOR_LATENT, "$\\mu$ (media)"),
              legend_dot(COLOR_LOGVAR, "$\\log \\sigma^2$ (log-varianza)"),
              legend_dot(COLOR_Z, "$z$ campionato")],
             xlim=(left - 70, right + 70), ylim=(y_dim - 100, y_name + 120))


if __name__ == "__main__":
    plot_linear_ae()
    plot_deep_ae()
    plot_vae()
    print("\nTutti i diagrammi architetturali sono stati generati con successo.")
