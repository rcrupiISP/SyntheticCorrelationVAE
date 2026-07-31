"""
=============================================================================
 VERIFICA AGGREGATA DEI CINQUE FATTI STILIZZATI SU TUTTI I 1000 SCENARI
 (Tesi, Capitolo 8, Sezione 8.2.1 -- Tabella 8.1)
=============================================================================

Estende all'intero insieme degli scenari generati la verifica dei cinque fatti
stilizzati che il notebook 14 conduce su un singolo scenario per volta.
Lo script NON produce grafici: il suo risultato e' la tabella riassuntiva di
conformita' (stampata a video, salvata in CSV e in un frammento LaTeX pronto
per essere incollato nella tesi).

Per ciascuna delle 1000 matrici sintetiche calcola un indicatore scalare per
fatto stilizzato, con le STESSE definizioni operative delle celle 5/7/9/11/13
del notebook `14_VAE_generation_analysis.ipynb`:

    FS1  media dei coefficienti pairwise fuori diagonale          (cella 5)
    FS2  lambda_1 e n. di autovalori oltre il bulk di M-P         (cella 7)
    FS3  componente minima del market mode normalizzato           (cella 9)
    FS4  rapporto correlazione media intra-settore / inter-settore
         (indicatore scalare della struttura a blocchi che la cella 11
          rende visivamente tramite il dendrogramma colorato per GICS)
    FS5  esponente gamma della legge di potenza P(k) ~ k^-gamma   (cella 13)

Gli stessi indicatori vengono calcolati anche su:
  - la matrice di correlazione odierna reale (riferimento del Capitolo 8);
  - le 412 matrici di correlazione reali del bacino epurato (indici 49..460
    del tensore grezzo, si veda il Capitolo 5, Sezione 5.3), che forniscono
    l'intervallo storicamente osservato usato come termine di paragone.

CRITERI DI CONFORMITA'. Sono quelli strutturali del Capitolo 3, non intervalli
numerici rigidi. In particolare per FS1 il fatto stilizzato richiede uno
spostamento marcato della distribuzione verso valori positivi: l'intervallo
[0.2, 0.5] citato in letteratura descrive la magnitudo abituale nei regimi
ordinari, non un limite algebrico, e una correlazione media superiore a 0.5
(scenario di forte sincronizzazione del listino) non costituisce violazione.

VALIDAZIONE DELLE DEFINIZIONI. Eseguito su `matrice_oggi` questo script
riproduce esattamente i valori gia' riportati nel Capitolo 8: media 0.4512,
mediana 0.4593, lambda_1 = 169.1, 9 autovalori oltre lambda_+, pendenza MST
-2.119, grado massimo 15. Il controllo e' automatico.

OUTPUT (in <payload_dir>/aggregate_stylized_facts/)
  metrics_synth.csv     indicatori dei 1000 scenari sintetici
  metrics_real.csv      indicatori delle 412 matrici reali del bacino epurato
  metrics_oggi.csv      indicatori della matrice odierna reale
  summary_table.csv     tabella riassuntiva di conformita'
  summary_table.tex     stessa tabella come frammento LaTeX (Tabella 8.1)

USO (con working directory = cartella `notebooks/`, come per i notebook):
    python 15_aggregate_stylized_facts.py
    python 15_aggregate_stylized_facts.py --table-only   # riusa i CSV salvati

Tempo di esecuzione: ~75 s (1412 matrici 362x362).
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import adjusted_rand_score

# =============================================================================
# 0. PARAMETRI (allineati alla cella 0 del notebook 14)
# =============================================================================
NUM_ASSETS = 362
WINDOW_LENGTH = 724
LATENT_DIM = 20
FILE_NAME = "data_00_20"
STRIDE_GEN = 1
STRIDE_TRAIN = 10
DATASET_NAME = f"{FILE_NAME}_w{WINDOW_LENGTH}_s{STRIDE_GEN}"
DATASET_MODEL_NAME = f"{FILE_NAME}_w{WINDOW_LENGTH}_s{STRIDE_TRAIN}"
RUN_NAME = "VAE_20dim_cholesky_06_loss"      # run definitiva usata nella tesi
N_SCENARIOS = 1000

# Indice della prima finestra valida dopo il filtro di Perron-Frobenius
# (Capitolo 5, Sezione 5.3: le finestre 0..48 violano la proprieta').
FIRST_VALID_WINDOW = 49

BASE_DIR = Path("..")
PAYLOAD_PATH = (BASE_DIR / "generated_matrices" / DATASET_MODEL_NAME / RUN_NAME /
                DATASET_NAME / f"forecast_scenarios_{N_SCENARIOS}" /
                f"forecast_scenarios_{N_SCENARIOS}.pt")
REAL_MATRICES_PATH = (BASE_DIR / "data/processed" / FILE_NAME / "correlation_matrices" /
                      f"{DATASET_MODEL_NAME}.pt")
TICKERS_PATH = BASE_DIR / "data/processed" / FILE_NAME / f"{FILE_NAME}_tickers.json"
GICS_PATH = BASE_DIR / "data/processed/gics_by_ticker.csv"

OUT_DIR = PAYLOAD_PATH.parent / "aggregate_stylized_facts"

# Limiti teorici di Marchenko-Pastur, q = N/T (identici alla cella 7 di nb14)
Q = NUM_ASSETS / WINDOW_LENGTH
MP_UPPER = (1.0 + np.sqrt(Q)) ** 2           # 2.9142
TRIU = np.triu_indices(NUM_ASSETS, k=1)

# Soglie: FS1_MIN e' l'estremo inferiore dell'intervallo tipico [0.2, 0.5]
# indicato dal Capitolo 3; non esiste invece un estremo superiore vincolante.
FS1_MIN = 0.2
FS5_RANGE = (2.0, 3.0)        # Capitolo 3, Sezione 3.6


# =============================================================================
# 1. METRICHE DEI CINQUE FATTI STILIZZATI
# =============================================================================
def sanitize_correlation_matrix(corr_matrix):
    """Garantisce i limiti [-1,1], la simmetria perfetta e la diagonale
    unitaria (anti-floating-point error). Identica a nb14, cella 11."""
    C = np.clip(np.asarray(corr_matrix, dtype=np.float64), -1.0, 1.0)
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C


def build_distance_matrix(corr_matrix):
    """Trasforma la correlazione in distanza metrica: d = sqrt(2*(1-rho)).
    Identica a nb14, cella 11."""
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr_matrix)))
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def degree_powerlaw(deg):
    """Fit log-log della distribuzione di grado dell'MST.

    Replica esattamente `process_degree_data` + `np.polyfit` di nb14 cella 13:
    si considerano i soli gradi k >= 1 effettivamente osservati e la frequenza
    espressa in percentuale. Ritorna (pendenza, gamma = -pendenza, R2, k_max).
    """
    max_degree = int(deg.max())
    counts = np.bincount(deg, minlength=max_degree + 1)
    freq = counts / counts.sum()
    degrees_arr = np.arange(max_degree + 1)
    mask = (freq > 0) & (degrees_arr > 0)
    x, y = degrees_arr[mask], freq[mask] * 100.0
    if len(x) < 2:
        return np.nan, np.nan, np.nan, max_degree
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    ss_res = np.sum((ly - (intercept + slope * lx)) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, -slope, r2, max_degree


def sector_stats(C, sector_ids, known_mask):
    """Correlazione media intra-settore e inter-settore (fuori diagonale) e
    loro rapporto. Gli asset senza settore GICS assegnato sono esclusi."""
    valid = known_mask[:, None] & known_mask[None, :]
    same = (sector_ids[:, None] == sector_ids[None, :]) & valid
    vals = C[TRIU]
    valid_u, same_u = valid[TRIU], same[TRIU]
    intra = vals[same_u].mean()
    inter = vals[valid_u & ~same_u].mean()
    return intra, inter, intra / inter


def compute_stylized_facts(corr_matrix, sector_ids, n_sectors, known_mask):
    """Tutti gli indicatori dei cinque fatti stilizzati per una matrice."""
    C = sanitize_correlation_matrix(corr_matrix)
    out = {}

    # --- FS1: distribuzione pairwise (nb14 cella 5) ---------------------
    vals = C[TRIU]
    out['mean_corr'] = float(vals.mean())
    out['median_corr'] = float(np.median(vals))
    out['frac_neg'] = float((vals < 0).mean())

    # --- FS2 / FS3: spettro (nb14 celle 7 e 9) --------------------------
    evals, evecs = np.linalg.eigh(C)
    evals, evecs = evals[::-1], evecs[:, ::-1]
    out['lambda1'] = float(evals[0])
    out['lambda2'] = float(evals[1])
    out['lambda_min'] = float(evals[-1])
    out['n_above_mp'] = int(np.sum(evals > MP_UPPER))
    out['var_expl_1'] = float(evals[0] / NUM_ASSETS)

    v1 = evecs[:, 0].copy()
    if v1.sum() < 0:                      # normalizzazione di segno (nb14)
        v1 = -v1
    out['v1_min'] = float(v1.min())
    out['pf_ok'] = bool(np.all(v1 >= -1e-4))     # stessa tolleranza di nb14
    out['pf_strict'] = bool(np.all(v1 > 0))
    out['eig_gap'] = float(evals[0] - evals[1])  # molteplicita' di lambda_1

    # --- FS4: struttura gerarchica (nb14 cella 11) ----------------------
    D = build_distance_matrix(C)
    Z = linkage(squareform(D, checks=False), method='average')
    labels = fcluster(Z, t=n_sectors, criterion='maxclust')
    # ARI calcolato per completezza: con average linkage su 362 asset il
    # taglio a k cluster degenera in un cluster gigante piu' singoletti
    # (ARI ~ 0.01 anche sulle matrici reali), per cui NON viene usato come
    # indicatore nella tabella; si usa il rapporto intra/inter.
    out['ari_gics'] = float(adjusted_rand_score(sector_ids, labels))
    intra, inter, ratio = sector_stats(C, sector_ids, known_mask)
    out['corr_intra'] = float(intra)
    out['corr_inter'] = float(inter)
    out['intra_inter_ratio'] = float(ratio)
    out['merge_max'] = float(Z[:, 2].max())

    # --- FS5: MST scale-free (nb14 cella 13) ----------------------------
    mst = minimum_spanning_tree(D)
    adj = (mst.toarray() > 0).astype(int)
    adj = adj + adj.T
    deg = adj.sum(axis=1).astype(int)
    slope, gamma, r2, kmax = degree_powerlaw(deg)
    out['mst_slope'] = float(slope)
    out['mst_gamma'] = float(gamma)
    out['mst_r2'] = float(r2)
    out['mst_kmax'] = int(kmax)
    out['mst_leaves_pct'] = float(100.0 * np.sum(deg == 1) / NUM_ASSETS)
    return out


def load_gics_sectors(tickers):
    """Mappa ticker -> settore GICS (stesso file usato da nb14 cella 11)."""
    df = pd.read_csv(GICS_PATH)
    df['ticker'] = df['ticker'].astype(str).str.upper()
    gmap = dict(zip(df['ticker'], df['gics_category']))
    sectors = [gmap.get(str(t).upper(), 'Unknown') for t in tickers]
    uniq = sorted(set(sectors))
    idx = {s: i for i, s in enumerate(uniq)}
    ids = np.array([idx[s] for s in sectors])
    known = np.array([s != 'Unknown' for s in sectors])
    return ids, uniq, known


# =============================================================================
# 2. CALCOLO SU 1000 SCENARI + 412 MATRICI REALI + MATRICE ODIERNA
# =============================================================================
def run_metrics():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = json.load(open(TICKERS_PATH, encoding='utf-8'))
    sector_ids, uniq_sectors, known = load_gics_sectors(tickers)
    n_sectors = len(uniq_sectors)
    print(f"Ticker: {len(tickers)} | settori GICS: {n_sectors} "
          f"({int((~known).sum())} asset senza settore, esclusi da FS4)")
    print(f"Marchenko-Pastur: q = {Q:.4f}, lambda_+ = {MP_UPPER:.4f}\n")

    # ---------------- scenari sintetici ----------------
    print(f"Caricamento payload: {PAYLOAD_PATH}")
    payload = torch.load(PAYLOAD_PATH, map_location='cpu', weights_only=False)
    C_oggi = np.asarray(payload['matrice_oggi'])
    scenarios = np.asarray(payload['forecast_matrices'])
    deltas = np.asarray(payload['sampled_cumulative_deltas'])
    print(f"  forecast_matrices: {scenarios.shape}\n")

    rows, t0 = [], time.time()
    for i in range(scenarios.shape[0]):
        row = compute_stylized_facts(scenarios[i], sector_ids, n_sectors, known)
        row['idx'] = i
        row['delta_norm'] = float(np.linalg.norm(deltas[i]))
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  scenari elaborati: {i+1}/{scenarios.shape[0]}  ({time.time()-t0:.0f}s)")
    df_synth = pd.DataFrame(rows)
    df_synth.to_csv(OUT_DIR / "metrics_synth.csv", index=False)
    del scenarios, payload

    # ---------------- matrice odierna ----------------
    m_oggi = compute_stylized_facts(C_oggi, sector_ids, n_sectors, known)
    pd.DataFrame([m_oggi]).to_csv(OUT_DIR / "metrics_oggi.csv", index=False)

    # ---------------- 412 matrici reali (bacino epurato) ----------------
    print(f"\nCaricamento matrici reali: {REAL_MATRICES_PATH}")
    real = np.asarray(torch.load(REAL_MATRICES_PATH, map_location='cpu',
                                 weights_only=False)['corr_tensor'])
    print(f"  tensore grezzo {real.shape} -> uso [{FIRST_VALID_WINDOW}:] "
          f"= {real.shape[0]-FIRST_VALID_WINDOW} matrici valide")
    rows = []
    for i in range(FIRST_VALID_WINDOW, real.shape[0]):
        row = compute_stylized_facts(real[i], sector_ids, n_sectors, known)
        row['idx'] = i
        rows.append(row)
    df_real = pd.DataFrame(rows)
    df_real.to_csv(OUT_DIR / "metrics_real.csv", index=False)

    print(f"\nCompletato in {time.time()-t0:.0f}s -> {OUT_DIR}")
    return df_synth, df_real, pd.Series(m_oggi)


# =============================================================================
# 3. CONTROLLO: RIPRODUZIONE DEI NUMERI GIA' RIPORTATI NEL CAPITOLO 8
# =============================================================================
def check_against_thesis(m_oggi, df_synth):
    print("\n" + "=" * 78)
    print(" CONTROLLO DI COERENZA CON I VALORI RIPORTATI NEL CAPITOLO 8")
    print("=" * 78)
    s63 = df_synth[df_synth.idx == 63].iloc[0]
    checks = [
        ("odierna  media pairwise", m_oggi['mean_corr'], 0.4512),
        ("odierna  mediana pairwise", m_oggi['median_corr'], 0.4593),
        ("odierna  lambda_1", m_oggi['lambda1'], 169.1),
        ("odierna  n autovalori > MP+", m_oggi['n_above_mp'], 9),
        ("odierna  pendenza MST", m_oggi['mst_slope'], -2.119),
        ("odierna  grado massimo MST", m_oggi['mst_kmax'], 15),
        ("scen.63  media pairwise", s63['mean_corr'], 0.4635),
        ("scen.63  mediana pairwise", s63['median_corr'], 0.4711),
        ("scen.63  lambda_1", s63['lambda1'], 173.2),
        ("scen.63  n autovalori > MP+", s63['n_above_mp'], 9),
        ("scen.63  pendenza MST", s63['mst_slope'], -2.227),
        ("scen.63  grado massimo MST", s63['mst_kmax'], 14),
    ]
    ok_all = True
    for name, got, expected in checks:
        tol = 0.05 if abs(expected) > 1 else 0.002
        ok = abs(got - expected) < tol
        ok_all &= ok
        print(f"  {name:30s} calcolato = {got:>12.4f}   tesi = {expected:>8}   "
              f"{'OK' if ok else '*** DIFFORME ***'}")
    print("=" * 78)
    print(" Tutti i valori coincidono." if ok_all else
          " ATTENZIONE: alcuni valori non coincidono con il testo della tesi.")
    return ok_all


# =============================================================================
# 4. TABELLA RIASSUNTIVA DI CONFORMITA' (Tabella 8.1 della tesi)
# =============================================================================
def build_summary_table(df_synth, df_real, m_oggi):
    s, r, o = df_synth, df_real, m_oggi

    # Maschere di conformita', una per fatto stilizzato.
    masks = {
        'FS1': s.mean_corr >= FS1_MIN,                             # spostamento positivo marcato
        'FS2': (s.lambda1 > MP_UPPER) & (s.n_above_mp >= 2),       # mercato + almeno un settore
        'FS3': s.pf_ok & (s.lambda1 > s.lambda2),                  # segno uniforme, lambda_1 semplice
        'FS4': s.intra_inter_ratio > 1.0,                          # blocchi settoriali presenti
        'FS5': s.mst_gamma.between(*FS5_RANGE, inclusive='neither'),
    }
    tutti = masks['FS1'] & masks['FS2'] & masks['FS3'] & masks['FS4'] & masks['FS5']

    def rng(df, col, fmt='.4f'):
        return f"[{df[col].min():{fmt}}, {df[col].max():{fmt}}]"

    rows = [
        dict(fatto='FS1 - Distribuzione pairwise', indicatore='media fuori diagonale',
             criterio='spostamento positivo marcato',
             sintetici=rng(s, 'mean_corr'), odierna=f"{o['mean_corr']:.4f}",
             reali=rng(r, 'mean_corr'), conformi=int(masks['FS1'].sum())),
        dict(fatto='FS2 - Spettro e Marchenko-Pastur',
             indicatore='lambda_1; n. modi > lambda_+',
             criterio='lambda_1 >> lambda_+; >= 2 modi',
             sintetici=f"{rng(s,'lambda1','.1f')}; {rng(s,'n_above_mp','d')}",
             odierna=f"{o['lambda1']:.1f}; {int(o['n_above_mp'])}",
             reali=f"{rng(r,'lambda1','.1f')}; {rng(r,'n_above_mp','d')}",
             conformi=int(masks['FS2'].sum())),
        dict(fatto='FS3 - Perron-Frobenius', indicatore='min_i v_1i', criterio='> 0',
             sintetici=rng(s, 'v1_min'), odierna=f"{o['v1_min']:.4f}",
             reali=rng(r, 'v1_min'), conformi=int(masks['FS3'].sum())),
        dict(fatto='FS4 - Struttura gerarchica', indicatore='C_intra / C_inter',
             criterio='> 1', sintetici=rng(s, 'intra_inter_ratio', '.3f'),
             odierna=f"{o['intra_inter_ratio']:.3f}",
             reali=rng(r, 'intra_inter_ratio', '.3f'), conformi=int(masks['FS4'].sum())),
        dict(fatto='FS5 - MST scale-free', indicatore='esponente gamma',
             criterio='2 < gamma < 3', sintetici=rng(s, 'mst_gamma', '.3f'),
             odierna=f"{o['mst_gamma']:.3f}", reali=rng(r, 'mst_gamma', '.3f'),
             conformi=int(masks['FS5'].sum())),
        dict(fatto='TUTTI E CINQUE SIMULTANEAMENTE', indicatore='', criterio='',
             sintetici='', odierna='', reali='', conformi=int(tutti.sum())),
    ]
    table = pd.DataFrame(rows)
    table['su'] = len(s)
    table['perc'] = (100.0 * table.conformi / len(s)).round(2)
    return table, masks, tutti


def print_summary(table, masks, tutti, df_synth, df_real, m_oggi):
    s, r = df_synth, df_real
    print("\n" + "=" * 100)
    print(f" TABELLA RIASSUNTIVA -- CONFORMITA' AI CINQUE FATTI STILIZZATI "
          f"({len(s)} SCENARI GENERATI)")
    print("=" * 100)
    hdr = f"  {'Fatto stilizzato':34s} {'Criterio':30s} {'Scenari sintetici':24s} {'Conformi':>16s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, row in table.iterrows():
        conf = f"{row.conformi}/{row.su} ({row.perc:.1f}%)"
        print(f"  {row.fatto:34s} {row.criterio:30s} {row.sintetici:24s} {conf:>16s}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  matrici definite positive (lambda_min > 0): "
          f"{int((s.lambda_min > 0).sum())}/{len(s)}"
          f"   lambda_min in [{s.lambda_min.min():.4f}, {s.lambda_min.max():.4f}]")
    if (~tutti).any():
        print(f"  scenari NON conformi: {list(s.idx[~tutti])}")

    # Nota su FS1: la soglia 0.5 dell'intervallo tipico non e' vincolante.
    above = s.mean_corr > 0.5
    if above.any():
        print(f"\n  Nota FS1: {int(above.sum())} scenario/i supera(no) 0.5 di correlazione media "
              f"(max {s.mean_corr.max():.4f}, scenario n.{int(s.idx[s.mean_corr.idxmax()])}).")
        print(f"  Non e' una violazione: il fatto stilizzato richiede uno spostamento positivo "
              f"marcato, che un valore piu' alto soddisfa a maggior ragione.")
        print(f"  Anche il mercato reale arriva a {r.mean_corr.max():.4f} nelle finestre piu' "
              f"sincronizzate del bacino storico.")

    print("\n  Riferimento: stessi criteri sulle 412 matrici REALI del bacino epurato")
    print(f"    FS1 media >= {FS1_MIN}      : {100*(r.mean_corr >= FS1_MIN).mean():6.2f}%")
    print(f"    FS3 Perron-Frobenius   : {100*r.pf_ok.mean():6.2f}%"
          f"   (il bacino e' epurato per costruzione, Cap. 5 Sez. 5.3)")
    print(f"    FS4 intra/inter > 1    : {100*(r.intra_inter_ratio > 1).mean():6.2f}%")
    print(f"    FS5 gamma in (2,3)     : "
          f"{100*r.mst_gamma.between(*FS5_RANGE, inclusive='neither').mean():6.2f}%")

    # Scarto sistematico rispetto alla matrice odierna: stima dell'effetto di
    # ricostruzione del decoder (commento riportato nell'osservazione di 8.2.1).
    if 'delta_norm' in s:
        i_min = int(s.delta_norm.idxmin())
        print(f"\n  Scenario con salto latente minimo: idx {int(s.idx[i_min])}, "
              f"||dz|| = {s.delta_norm[i_min]:.3f} (mediana {s.delta_norm.median():.3f})"
              f" -> media = {s.mean_corr[i_min]:.4f}  vs  odierna reale {m_oggi['mean_corr']:.4f}")
        print("  (lo scarto e' quindi attribuibile alla ricostruzione del decoder, "
              "non all'evoluzione a un mese)")
    print("=" * 100)


def write_latex_table(table, path):
    """Frammento LaTeX corrispondente alla Tabella 8.1 della tesi."""
    tex = {
        'FS1': (r'FS1 -- Distribuzione \emph{pairwise} \newline (media fuori diagonale)',
                'spostamento positivo marcato'),
        'FS2': (r'FS2 -- Spettro e Marchenko-Pastur \newline ($\lambda_1$; n. modi $> \lambda_+$)',
                r'$\lambda_1 \gg \lambda_+$; almeno 2 modi'),
        'FS3': (r'FS3 -- Perron-Frobenius \newline ($\min_i v_{1,i}$)', r'$> 0$'),
        'FS4': (r'FS4 -- Struttura gerarchica \newline (rapporto intra/inter)', r'$> 1$'),
        'FS5': (r'FS5 -- MST \emph{scale-free} \newline (esponente $\gamma$)', r'$2 < \gamma < 3$'),
    }
    math = lambda v: '$' + v.replace('; ', '$; \\newline $') + '$' if v else ''
    lines = [r'\begin{table}[h!]', r'\centering', r'\footnotesize',
             r'\begin{tabular}{@{}p{2.6cm} p{2.0cm} p{2.4cm} p{1.4cm} p{2.4cm} p{1.7cm}@{}}',
             r'\hline',
             r'\textbf{Fatto stilizzato \newline (indicatore)} & \textbf{Criterio} & '
             r'\textbf{1000 scenari sintetici} & \textbf{Matrice odierna} & '
             r'\textbf{412 matrici reali} & \textbf{Scenari conformi} \\',
             r'\hline']
    for key, (label, criterio) in tex.items():
        row = table[table.fatto.str.startswith(key)].iloc[0]
        lines.append(f"{label} & {criterio} & {math(row.sintetici)} & {math(row.odierna)} & "
                     f"{math(row.reali)} & ${row.conformi}/{row.su}$ \\newline "
                     f"$({row.perc:g}\\%)$ \\\\[4pt]")
    last = table.iloc[-1]
    lines += [r'\hline',
              r'\multicolumn{5}{@{}l}{\textbf{Tutti e cinque i fatti stilizzati simultaneamente}} & '
              rf'$\mathbf{{{last.conformi}/{last.su}}}$ \newline $\mathbf{{({last.perc:g}\%)}}$ \\',
              r'\hline', r'\end{tabular}',
              r'\caption{Esito della verifica aggregata dei cinque fatti stilizzati sui 1000 '
              r'scenari generati dal modello \texttt{VAE\_20dim\_cholesky\_06\_loss}.}',
              r'\label{tab:sf_aggregate}', r'\end{table}']
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Frammento LaTeX salvato in: {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Verifica aggregata dei 5 fatti stilizzati sui 1000 scenari generati.")
    ap.add_argument('--table-only', action='store_true',
                    help='riusa i CSV gia salvati, ricalcola solo la tabella')
    args = ap.parse_args()

    if args.table_only:
        df_synth = pd.read_csv(OUT_DIR / "metrics_synth.csv")
        df_real = pd.read_csv(OUT_DIR / "metrics_real.csv")
        m_oggi = pd.read_csv(OUT_DIR / "metrics_oggi.csv").iloc[0]
    else:
        if not PAYLOAD_PATH.exists():
            sys.exit(f"Payload non trovato: {PAYLOAD_PATH.resolve()}\n"
                     "Eseguire questo script con working directory = cartella notebooks/.")
        df_synth, df_real, m_oggi = run_metrics()
        check_against_thesis(m_oggi, df_synth)

    table, masks, tutti = build_summary_table(df_synth, df_real, m_oggi)
    print_summary(table, masks, tutti, df_synth, df_real, m_oggi)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "summary_table.csv", index=False)
    print(f"\nTabella riassuntiva salvata in: {OUT_DIR / 'summary_table.csv'}")
    write_latex_table(table, OUT_DIR / "summary_table.tex")


if __name__ == '__main__':
    main()
