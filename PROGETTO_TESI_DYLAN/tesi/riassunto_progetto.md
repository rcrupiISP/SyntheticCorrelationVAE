# Riassunto del progetto — base di verifica per la stesura della tesi

> **Come usare questo documento.** È diviso in due parti nettamente separate:
> - **Parte A — Ground truth**: cosa è stato effettivamente costruito, eseguito e misurato, ricavato leggendo direttamente codice, output stampati (stdout) e file di risultato (`run_results.json`, CSV, `dataset_info.json`) del progetto. Non contiene alcuna affermazione presa dai capitoli già scritti della tesi: è pensata come riferimento "pulito" con cui confrontare il testo.
> - **Parte B — Registro delle discrepanze**: punti specifici in cui il testo già scritto in `tesi/Parte_I`, `Parte_II`, `Parte_III` diverge da quanto risulta dal codice/dai dati, oppure in cui la tesi descrive lavoro non ancora svolto. Ogni voce indica cosa dice il draft, cosa mostra il codice, e cosa andrebbe deciso/corretto.
>
> Tutti i numeri di Parte A sono citati con il file sorgente (notebook, cella di codice, o path del file JSON/CSV) da cui provengono, così da poter essere ri-verificati.

> **Decisioni prese con l'autore (2026-07-06)**, che sovrascrivono le raccomandazioni originarie della Parte B dove in conflitto:
> 1. Il Capitolo 9 (VaR/Expected Shortfall) **non fa più parte del perimetro della tesi** — non è un gap da colmare, è stato tolto per scelta. `scaletta_tesi.md` è stato aggiornato di conseguenza (Cap. 9 rimosso, VaR spostato tra gli "sviluppi futuri" in Conclusioni).
> 2. La tesi usa **esclusivamente** la pipeline finale a **362 asset, finestra 724 giorni** (`data_00_20_w724_s10`). La pipeline preliminare a 100 asset/252 giorni (`data_00_20_w252_s5`) è superata: ogni riferimento a N=100 o D=4950 nei capitoli già scritti (Cap. 3, Cap. 7) è un refuso da correggere a 362/65.341 (dominio raw) o 65.703 (dominio Cholesky), non un secondo esperimento da preservare.
> 3. **Non è vero, in generale, che il VAE ricostruisce peggio di PCA/LinearAE**: guardando le tabelle di confronto (dominio Cholesky, quello definitivo), il VAE batte sia PCA sia LinearAE a K=20 (resta comunque dietro l'AE). Il Cap. 7 va aggiornato per riflettere questo, non per confermare la narrazione opposta.
> 4. Il limite di rigore sulla validazione dei 5 stylized facts su singolo scenario (B.4) è confermato ma **non prioritario**: lasciarlo così per ora.
> 5. Confermato: la generazione usa il **block-bootstrap storico nello spazio latente** (non il semplice campionamento dal prior).

---

# PARTE A — Ground truth: cosa è stato fatto

## A.1 Panoramica

Il progetto (repo `PROGETTO_TESI_DYLAN`, tesi Politecnico di Torino, Laurea Magistrale Ingegneria Informatica, indirizzo AI and Data Analytics, candidato Dylan Magliano, relatori proff. Flavio Giobergia e Sergio Caprioli, titolo: *"Dai pixel ai portafogli finanziari: generare scenari di mercato realistici tramite variational autoencoder"*) implementa una pipeline completa che:

1. Costruisce un dataset di matrici di correlazione empiriche a finestra mobile su un paniere di 362 titoli S&P 500 (2000–2021).
2. Confronta quattro tecniche di compressione/ricostruzione di queste matrici: **PCA**, **Autoencoder Lineare**, **Autoencoder profondo (non lineare)**, **Variational Autoencoder**. Il dominio di lavoro **non è stato scelto a priori come confronto parallelo**, ma è il risultato di un'evoluzione metodologica in due fasi: prima si è testata la ricostruzione lavorando direttamente sul **triangolo inferiore della matrice di correlazione grezza**; verificato che le matrici ricostruite in questo dominio **non rispettavano sistematicamente la proprietà di semidefinita positività (PSD)**, si è passati a ricostruire il **triangolo inferiore della decomposizione di Cholesky**, che garantisce la PSD per costruzione ($L\hat L^T$). Il dominio raw resta quindi nel repository come fase superata/diagnostica, non come un'alternativa equivalente tenuta apposta per il confronto (vedi A.6).
3. Usa il VAE addestrato (dominio Cholesky) per **generare scenari sintetici futuri a 1 mese** tramite una tecnica di *block-bootstrap* nello spazio latente (non semplice campionamento dal prior).
4. Verifica, su singoli scenari generati, la conformità ai **5 stylized facts** delle matrici di correlazione finanziarie (distribuzione pairwise, spettro di Marchenko–Pastur, Perron-Frobenius, struttura gerarchica, MST scale-free).
5. **Non** implementa (in nessun notebook) l'applicazione finale a Value at Risk / Expected Shortfall / simulazione Monte Carlo di portafoglio. Questo era originariamente annunciato nell'abstract e nella scaletta (Cap. 9), ma **per decisione dell'autore (2026-07-06) è stato tolto dal perimetro della tesi** (vedi nota in cima al documento e B.1) — non è quindi (più) un gap da colmare.

Tutto il codice è in notebook Jupyter (cartella `notebooks/`), non esiste un package Python separato (`src/`); tutti i modelli sono in PyTorch.

## A.2 Struttura del repository

```
PROGETTO_TESI_DYLAN/
├── data/raw/                      # CSV.gz grezzi (prezzi S&P500; data_10_24/data_13_17 non pertinenti al lavoro di tesi)
├── data/processed/data_00_20/     # prezzi puliti, log-return, tensori di correlazione, dataset train/val/test
├── notebooks/                     # 01…14, GICS_extractor, mapping_date
├── models/data_00_20_w252_s5/     # run preliminari (N=100, dataset "vecchio")
├── models/data_00_20_w724_s10/    # run finali (N=362, dataset definitivo) — PCA/linearAE/AE/VAE, dominio raw e cholesky
├── results/, results_cholesky/    # analisi aggregate di ricostruzione e confronto fra modelli
├── generated_matrices/            # scenari sintetici generati dal VAE + analisi dei 5 stylized facts
├── loss_VAE.md                    # derivazione matematica della loss di ricostruzione "Divergenza di Jeffreys"
└── tesi/                          # il documento LaTeX (7 capitoli scritti su 9 previsti dalla scaletta)
```

## A.3 Evoluzione del progetto: due generazioni di dataset

Il codice mostra chiaramente **due fasi sperimentali distinte**, sovrapposte nel repository. È importante distinguerle perché numeri delle due fasi non sono confrontabili tra loro.

> Nota: il notebook `AE_linAE_PCA.ipynb` (dataset `data_10_24.csv.gz`, N=151, TensorFlow/Keras) **non è considerato parte del lavoro di tesi** — è un notebook di esempio fornito dai relatori, non un prototipo scritto da Dylan. Non compare quindi nella tabella sottostante né in nessun'altra parte di questo riassunto.

| Fase | Notebook/cartella | Dataset sorgente | N asset | Finestra / stride | Split train/val/test | Framework |
|---|---|---|---|---|---|---|
| 1 — Pipeline preliminare | `models/data_00_20_w252_s5/{21,252}_days_gap/` | `data_00_20` | 100 | W=252, stride=5 | split **temporale con "purging gap"** (21gg o 252gg) | PyTorch |
| 2 — Pipeline finale | `models/data_00_20_w724_s10/` | `data_00_20` | 362 | W=724, stride=10 | split **random-shuffle** 70/20/10 (seed 42) | PyTorch |

- La Fase 1 (`data_00_20_w252_s5`) è quella richiamata nel testo del Cap. 5 come "test preliminari con partizionamento temporale sequenziale" che hanno mostrato *distribution shift* estremo — è infatti lì che si osserva il fallimento di generalizzazione (vedi A.8).
- La Fase 2 (`data_00_20_w724_s10`) è la pipeline che **tutti i capitoli 1, 2, 5, 6 descrivono** ed è quella su cui sono stati addestrati e confrontati tutti i modelli "finali" citati nel notebook 12.

## A.4 Pipeline dati della versione finale (`data_00_20_w724_s10`)

1. **Dati grezzi** (`01_data_cleaning.ipynb`): `data_00_20.csv.gz`, prezzi di chiusura rettificati, 5334 giorni (1999-11-01 → 2021-01-12), 502 ticker iniziali.
2. **Pulizia**: soglia `MIN_QUOTES_PER_DAY=2` (di fatto **non rimuove nessun giorno**, è un no-op su questo dataset); filtro ticker a **zero missing values** su tutto lo storico → **362 ticker sopravvissuti**. Un ulteriore step di "campionamento stratificato per settore GICS" (`N_TICKERS_TO_KEEP=362`) è anch'esso un no-op, perché il target coincide esattamente col numero di ticker disponibili.
3. **Classificazione settoriale GICS** (`GICS_extractor.ipynb`): i settori GICS **non provengono da un provider dati ufficiale**, ma da classificazione automatica tramite LLM (**Google Gemini, modello `gemini-2.5-flash`**, prompt few-shot con 5 esempi, retry/backoff per gli errori 503 del servizio). Risultato: 362/362 ticker classificati, solo 2 rimasti "Unknown" (`ABC`, `CERN`). Distribuzione settoriale finale: Industrials 56 (15.5%), Financials 50 (13.8%), Health Care 46 (12.7%), Information Technology 44 (12.2%), Consumer Discretionary 33 (9.1%), Consumer Staples 29 (8.0%), Real Estate 28 (7.7%), Utilities 24 (6.6%), Materials 20 (5.5%), Energy 19 (5.2%), Communication Services 11 (3.0%), Unknown 2 (0.6%). Nessuna validazione incrociata con fonti terze.
4. **Log-return**: $r_t=\ln(P_t/P_{t-1})$, shape finale `(5333, 362)`.
5. **Matrici di correlazione a finestra mobile** (`02_corr_matrix_creation.ipynb`): correlazione di Pearson standard (`.corr()`), finestra `WINDOW_LENGTH=724` giorni (~3 anni). **Nota**: il file `data_00_20_w724_s10.pt` (stride=10, 461 finestre totali) usato da tutti i notebook successivi **non risulta generato da nessuna cella catturata** in questo notebook (che produce solo la versione stride=1, 4610 finestre) — è stato prodotto rieseguendo lo stesso notebook con `STRIDE=10` modificato a mano, run non documentata nel dump disponibile.
6. **Analisi stylized facts sui dati reali** (`03_base_corr_matrix_analysis.ipynb`) — su selezione **finestre 49:460 (412 matrici)**: vedi tabella in A.5.
7. **Costruzione dataset per training** (`04_dataset_creation.ipynb`): decomposizione di **Cholesky** con jitter `1e-6` sulla diagonale (`torch.linalg.cholesky`); split **puramente casuale** (permutazione `np.random.default_rng(42)`, **nessuna procedura di purging temporale**) → per la versione finale usata nel training: **train 288 / val 82 / test 42** (frazioni 70/20/10 su 412 matrici totali — confermato in `dataset_info.json`).
8. Il file con **stride=1** (`data_00_20_w724_s1`, 4610 finestre giornaliere quasi completamente sovrapposte) **non viene usato per addestrare modelli**: serve esclusivamente al notebook 13 per ricostruire, tramite encoder già allenato, una **traiettoria latente storica continua giorno-per-giorno**, usata poi per il block-bootstrap (vedi A.9).

**Nota metodologica non banale**: lo split random-shuffle (punto 7) elimina il problema del *distribution shift* di regime macroeconomico (obiettivo dichiarato nel Cap. 5), ma **non elimina il rischio di leakage da sovrapposizione di finestre**: con stride=10 su una finestra di 724 giorni, due matrici con indice consecutivo condividono 714/724 ≈ 98.6% delle osservazioni sottostanti. Poiché lo split è una permutazione casuale senza vincoli di distanza minima tra indici, matrici quasi identiche possono finire l'una in train e l'altra in val/test, gonfiando artificialmente le metriche di generalizzazione. Il codice non applica alcun "purging"/embargo per mitigare questo effetto — è un limite metodologico reale, distinto dal *distribution shift* temporale che il random-shuffle risolve.

## A.5 Stylized facts verificati sui dati reali (412 matrici, `03_base_corr_matrix_analysis.ipynb`)

| Proprietà | Risultato empirico |
|---|---|
| Distribuzione pairwise | Media 0.360, mediana 0.353, std 0.151; positivi 99.64%, negativi 0.36% (su 26.9M coefficienti) |
| Spettro / Marchenko–Pastur | $q=N/T=362/724=0.5$; limiti teorici $[a,b]=[0.086,\,2.914]$; in media **8.99** autovalori sopra $b$ (std 1.34, range 7–12) per matrice; $\lambda_1$ medio spiega il **36.16%** della varianza (std 9.6%), primi 4 autovalori insieme il **45.55%** |
| PSD | 412/412 matrici (100%) semidefinite positive entro tolleranza $10^{-8}$ |
| Perron-Frobenius | 412/412 (100%) soddisfano la proprietà (componenti del primo autovettore tutte $\ge 0$, molteplicità 1) |
| MST scale-free | Esponente della power-law stimato (fit log-log su tutte le 412 matrici): **γ ≈ −3.37** |

Tutte e cinque le proprietà risultano quindi **verificate al 100%** sui dati reali usati per il training — buona base empirica per il Cap. 3.

## A.6 Modelli implementati

Tutti condividono lo stesso dataset (`data_00_20_w724_s10`, 288/82/42), stesso seed (42), GPU NVIDIA RTX 3060. Il codice espone **due varianti di dominio** per ciascun modello, ma **non sono due alternative provate in parallelo per scelta**: sono due stadi successivi della stessa ricerca metodologica.

**Perché esistono entrambe (motivazione, non semplice confronto)**: il primo tentativo (dominio raw) ricostruiva direttamente il triangolo inferiore della matrice di correlazione. Verificando le matrici ricostruite, queste **non rispettavano in modo affidabile la proprietà di semidefinita positività (PSD)** — evidenza concreta nel codice della pipeline: nel notebook `10_single_reconstruction_analysis.ipynb`, la ricostruzione di un AE nel dominio raw produce un autovalore negativo (−0.000183) mentre la matrice originale è perfettamente PSD. Per **risolvere strutturalmente** questo problema, la pipeline è passata a ricostruire il **fattore di Cholesky** della matrice invece della correlazione grezza: essendo $C=L\hat L^T$ per costruzione, la matrice ricostruita è **matematicamente garantita PSD**, indipendentemente da quanto bene la rete abbia imparato a comprimerla. Il dominio Cholesky è quindi l'esito dell'evoluzione, non un ramo sperimentale parallelo — coerente con la scelta del Cap. 1 di adottare Cholesky come parametrizzazione definitiva. Il dominio raw resta nel repository perché più run (anche nella pipeline finale) sono stati comunque eseguiti/salvati in quella fase precedente, prima di consolidare la parametrizzazione Cholesky.

- **Raw** (fase superata): input = triangolo inferiore della correlazione, **esclusa** diagonale (`k=-1`) → **65.341** feature. Nessuna garanzia di PSD sulla ricostruzione (solo clip euristico `[-1,1]` e diagonale forzata a 1).
- **Cholesky** (parametrizzazione adottata in via definitiva): input = triangolo inferiore del fattore di Cholesky, **inclusa** diagonale (`k=0`, la diagonale di L è informativa) → **65.703** feature. Ricostruzione garantita PSD per costruzione: $L\hat L^T$ → normalizzazione a correlazione ($D^{-1/2} \cdot C \cdot D^{-1/2}$) → clip finale. Verificato esplicitamente nel notebook `06_AE_cholesky.ipynb` (funzione `verify_psd_batch`): 42/42 matrici del test set risultate PSD (100%).

### PCA
`sklearn.decomposition.PCA`, fit solo su train, nessuno scaling. K testati: 3, 10, 20, (25 solo dominio cholesky), 100, 288. A K=288 (= n_train) ricostruzione train quasi perfetta (rango pieno).

### Autoencoder Lineare e Autoencoder profondo
Un'**unica classe** `AutoEncoder(input_dim, latent_dim, hidden_dims=None, dropout_prob)`: se `hidden_dims=None` → Linear AE puro (nessun bias, nessuna attivazione); altrimenti Deep AE con `hidden_dims=[2048,1024,512]`, `LeakyReLU(0.01)` + Dropout in encoder, Tanh finale presente nel codice ma **sempre commentata/disattivata** in tutte le run osservate. Optimizer Adam, scheduler opzionale `CosineAnnealingLR`, **checkpointing sul minimo di validation loss** (non early stopping: il training gira comunque per tutte le epoche).

### VAE
Stessa struttura funnel (`hidden_dims=[2048,1024,512]`), encoder che si biforca in `fc_mu`/`fc_logvar`, reparameterization trick standard, in `eval()` si usa `z=mu` (deterministico, niente rumore in inferenza/ricostruzione). **Tre varianti di loss**, tutte con framework $\mathcal L=\mathcal L_{recon}+\beta\mathcal L_{KL}$:
1. **MSE+KL, dominio raw** (`07_VAE.ipynb`): KL scalata dividendo per il numero di feature.
2. **MSE+KL, dominio Cholesky** (`07_VAE_cholesky.ipynb`): stessa struttura, KL scalata idem.
3. **"Divergenza di Jeffreys" + KL, dominio Cholesky** (`07_VAE_cholesky_loss.ipynb`, derivazione in `loss_VAE.md`): la loss di ricostruzione non è un MSE elemento-per-elemento ma una **KL simmetrizzata (Jeffreys) tra $\mathcal N(0,R_{orig})$ e $\mathcal N(0,R_{recon})$**, calcolata sulla matrice di correlazione ricostruita **dentro il grafo computazionale differenziabile** (via `torch.linalg.solve` con ridge $10^{-5}$ per stabilità), non sui coefficienti di Cholesky grezzi. **Qui il termine KL non è scalato per il numero di feature** (a differenza delle varianti 1 e 2): i valori di `beta` **non sono confrontabili tra le tre varianti**. Questa variante mostra instabilità di training (loss iniziale enorme, uno spike anomalo intorno all'epoca 249 mascherato manualmente nei grafici) mitigata con **gradient clipping (max_norm=500)**, assente nelle altre due varianti.

## A.7 Setup sperimentale comune

- Optimizer: Adam. Scheduler: `CosineAnnealingLR` (quando usato). Batch size tipico 64 (talvolta 128). Epoche: 500–2000 a seconda della run.
- **Criterio di selezione pesi**: si salva il modello al minimo di validation loss osservato durante l'intero ciclo di epoche (nessun early stopping vero e proprio).
- Metriche: MSE, MAE, norma di Frobenius **calcolate sull'intera matrice N×N** (non solo sul triangolo superiore) — quindi ogni coefficiente fuori diagonale viene contato **due volte** (simmetria) e la diagonale (errore sempre 0, perché reimposta a 1) diluisce ulteriormente la media. Questo non invalida i confronti *relativi* fra modelli (fatti con la stessa convenzione), ma va tenuto presente se si citano i valori assoluti.
- Metriche topologiche (dal notebook 11): edge overlap % e Jaccard % dell'MST, distanza L1 fra distribuzioni di grado, differenza di *average path length* (pesata e non), overlap dei top-10 nodi per betweenness centrality.

## A.8 Risultati di ricostruzione — tabelle comparative

Fonte: `results_cholesky/data_00_20_w724_s10/comparison_{train,test}/*.csv` (prodotte da `12_compare_models.ipynb`), incrociate con i singoli `run_results.json`. **Dominio Cholesky** (quello scelto come definitivo nel Cap. 1):

**MSE medio per matrice — Test set** (`comparison_test`):

| K | PCA | LinearAE | AE | VAE |
|---|---|---|---|---|
| 3 | 3.56e-03 | 4.14e-03 | 1.70e-04 | — |
| 10 | 5.55e-04 | 6.72e-04 | 8.45e-05 | — |
| 20 | 2.73e-04 | 3.38e-04 | 7.19e-05 | **1.55e-04** (run `_06_loss`) |
| 100 | 7.39e-05 | 8.19e-05 | 7.34e-05 | — |

**MSE medio per matrice — Train set** (`comparison_train`, include anche K=30):

| K | PCA | LinearAE | AE | VAE |
|---|---|---|---|---|
| 3 | 3.38e-03 | 4.11e-03 | 6.89e-05 | — |
| 10 | 4.86e-04 | 6.01e-04 | 2.11e-06 | 6.33e-05 (run `_08_loss`) |
| 20 | 2.14e-04 | 2.78e-04 | 7.23e-07 | 8.69e-05 (run `_06_loss`) |
| 30 | — | — | — | 1.16e-04 (run `_07_loss`) |
| 100 | 1.97e-05 | 2.33e-05 | 8.83e-07 | — |

**Lettura**: **AE domina** su tutti gli altri modelli a ogni K (di 1-2 ordini di grandezza). **PCA ≈ LinearAE** a ogni K (differenza <20%, coerente col teorema di Bourlard–Kamp). **Il VAE (varianti "loss" Jeffreys, dominio Cholesky) batte sia PCA sia LinearAE a K=20**, pur restando nettamente dietro l'AE. Il VAE è l'unico modello il cui errore **non migliora monotonicamente con K** (train MSE: 6.33e-5 a K=10 → 8.69e-5 a K=20 → 1.16e-4 a K=30) — comportamento anomalo attribuibile alla competizione fra ricostruzione e regolarizzazione KL, e/o al fatto che K=10/20/30 sono run diverse con beta/iperparametri diversi, non un ablation controllato sul solo K.

Le metriche topologiche MST (edge overlap %, Jaccard %, ecc.) seguono lo stesso ordinamento: **AE > VAE > PCA ≈ LinearAE** a ogni K comune (es. edge overlap % a K=20, test: AE 91.1%, VAE 88.4%, PCA 83.0%, LinearAE 83.1%).

**Importante — dipendenza dal dominio**: nel **dominio raw** (correlazione grezza, non Cholesky), il quadro è opposto: a K=20 test, VAE MSE = **6.33e-3**, contro PCA 1.78e-4 e LinearAE 1.78e-4 — il VAE **soccombe nettamente** anche ai modelli lineari. È questo il risultato (dominio raw) che corrisponde a quanto narrato nel Cap. 7 (vedi Parte B).

Anche nella Fase 1 (dataset preliminare `data_00_20_w252_s5`, N=100, split temporale con gap): il VAE ha sempre MSE test più alto di PCA/LinearAE a ogni K (es. K=100, gap 252gg: VAE 1.98e-2 vs PCA 1.00e-2 vs LinearAE 1.27e-2), e in quella fase **anche l'AE profondo non batte la PCA a K alti** (overfitting da distribution shift temporale) — coerente con l'affermazione del Cap. 5 sui "test preliminari" falliti.

**Copertura K incompleta per il VAE**: a differenza di PCA/LinearAE/AE (sempre testati a K=3,10,20,100), il **VAE non è mai stato confrontato a K=3 o K=100**, né su train né su test; il confronto su test manca inoltre il K=30. Qualunque tabella di tesi che confronti "tutti i modelli agli stessi K" per il VAE va verificata con attenzione.

**Run non riproducibili al 100%**: i run "finali" scelti per il confronto hanno `run_id` non uniformi tra i K (es. AE: `_01`, `_01`, `_03_`, `_02_` per K=3,10,20,100; VAE: `_08_loss`, `_06_loss`, `_07_loss` per K=10,20,30) — segno di selezione manuale del "run migliore" dopo tuning iterativo di cui non resta traccia sistematica (nessuna tabella con tutti i tentativi, nessun criterio di scelta esplicito nei notebook).

## A.9 Generazione stocastica di scenari (VAE, notebook 13-14)

La procedura **non è un semplice campionamento dal prior** $\mathcal N(0,I)$ come descritto nel Cap. 4, ma un **block-bootstrap storico nello spazio latente**:

1. Si codifica (con l'encoder del VAE Cholesky scelto, es. `VAE_20dim_cholesky_10_loss`) l'**intera traiettoria storica giornaliera** (dataset stride=1, 4124 finestre, da `data_00_20_w724_s1`), ottenendo una sequenza continua $z_{history}$.
2. Si calcolano gli **incrementi giornalieri** $\Delta z_t = z_t - z_{t-1}$.
3. Per ciascuno dei $N$ scenari (100 o 1000), si estrae **casualmente un blocco storico di 21 incrementi giornalieri consecutivi** (≈ 1 mese lavorativo) e lo si somma allo stato attuale $z_{oggi}$, ottenendo $z_{futuro}$.
4. Si decodifica $z_{futuro}$ e si ricostruisce la matrice di correlazione garantendone la PSD (via Cholesky → $LL^T$ → normalizzazione).

Sono stati generati set di **100 e 1000 scenari** a 21 giorni (1 mese) per due run del VAE (`_06_loss` e `_10_loss`).

**Validazione dei 5 stylized facts sugli scenari generati** (notebook 14): per un singolo scenario alla volta (indici esplorati: 0, 1, 10, 63, 99, 312, 500, 705 a seconda della run), si confronta la matrice "oggi reale" con la matrice "futuro sintetico" su tutte e 5 le proprietà. **Nessuna violazione** è stata osservata negli scenari esaminati (PSD sempre verificata, Perron-Frobenius sempre soddisfatta, spettro MP con lo stesso numero di modalità di mercato/settore, dendrogrammi e MST topologicamente simili, es. per lo scenario idx=705: γ_oggi=−2.119 vs γ_futuro=−2.236, 15 vs 14 connessioni massime, 55.0% vs 55.2% nodi foglia).

**Limite di rigore importante**: questa validazione è **sempre su un singolo scenario per volta**, mai in forma aggregata/statistica sull'intero insieme di 100 o 1000 scenari generati. Nel caso della run `_10_loss`/1000 scenari, lo scenario scelto per l'analisi dettagliata (idx=705) è per giunta **lo scenario più estremo in assoluto** secondo la distanza di Mahalanobis dal centroide della nuvola di scenari (distanza 14.14 contro una media di 3.95 e un 95° percentile di 7.42) — cioè proprio l'esempio meno rappresentativo dell'insieme. Non esiste nel codice un loop che calcoli, ad esempio, "su quanti dei 1000 scenari il primo autovettore ha tutte componenti positive" o la distribuzione aggregata dell'esponente γ del power-law. Un'analisi di questo tipo sarebbe l'evidenza naturale da presentare nel Cap. 8, ma **non è stata implementata**.

Analisi aggiuntive presenti: distanza di Mahalanobis della nuvola di scenari dal centroide, PCA sulla nuvola di scenari futuri (la prima componente spiega il 25.3% della varianza nell'esempio osservato), identificazione degli scenari al 5°/95° percentile lungo l'asse principale.

## A.10 Cosa NON è stato implementato (e cosa è ora fuori perimetro per scelta)

- **Nessun notebook contiene calcoli di Value at Risk, Expected Shortfall, simulazione Monte Carlo di portafoglio o backtest di rischio** (verificato per ricerca testuale su tutti i notebook: zero occorrenze di "value at risk", "expected shortfall", "monte carlo", "portfolio"). Gli 1000 scenari generati (A.9) sarebbero stati la materia prima per un motore Monte Carlo, ma il passo finale non è mai stato scritto. **Per decisione dell'autore (2026-07-06) questo non è più un gap da colmare**: l'applicazione VaR/ES è stata tolta dal perimetro della tesi (il Cap. 9 della scaletta è stato rimosso) e relegata a possibile sviluppo futuro nelle Conclusioni.
- **Capitolo 8 della scaletta non ha ancora un file `.tex`** (esiste solo `Parte_III/capitolo7.tex`). Il lavoro corrispondente (validazione stylized facts sul generato) esiste già come notebook/output (A.9) e resta da scrivere in forma testuale — questo, a differenza del Cap. 9, **è ancora nel perimetro della tesi**.
- Nessuna validazione incrociata della classificazione settoriale GICS (ottenuta solo via LLM) con una fonte dati ufficiale.

---

# PARTE B — Registro delle discrepanze e punti da verificare nella tesi

Ogni voce: **[capitolo]** cosa dice il draft → cosa mostra codice/dati → azione suggerita.

### B.1 — RISOLTO (2026-07-06): il Cap. 9 (VaR/Expected Shortfall) è stato tolto dal perimetro della tesi
**Draft (stato precedente)**: Abstract, §2.3 e le Conclusioni (scaletta) motivavano l'intero impianto della tesi con l'applicazione a VaR/Expected Shortfall su portafogli reali. Il Cap. 9 della scaletta era interamente dedicato a questo, ma zero notebook toccano VaR/ES/Monte Carlo di portafoglio (A.10).
**Decisione dell'autore**: il Cap. 9 non fa più parte del lavoro di tesi. `scaletta_tesi.md` è stato aggiornato: rimossa la promessa di VaR dall'abstract, rimosso l'intero Capitolo 9, spostata la menzione di VaR/Monte Carlo tra gli "sviluppi futuri" delle Conclusioni.
**Azione residua**: quando si rivedono i capitoli già scritti in `.tex`, verificare che nessun riferimento a una futura applicazione VaR resti nell'Abstract o nell'Introduzione (Cap. 1) come se fosse ancora un risultato della tesi — vanno riformulati come limite/sviluppo futuro, coerentemente con la nuova scaletta.

### B.2 — RISOLTO (2026-07-06): il Cap. 7 riporta i numeri di una pipeline ormai superata e va corretto a 362/724
**Draft (Cap. 7, §7.1)**: "l'input fornito ai modelli... è costituito dai vettori appiattiti $x\in\mathbb R^{4950}$"; il Cap. 3 parla di "100 asset estratti dall'indice S&P 500".
**Codice**: $4950 = 100\cdot99/2$ — questo combacia **esattamente** con la Fase 1 preliminare (`data_00_20_w252_s5`, N=100, split temporale con gap), **non** con la pipeline finale a 362 titoli (D=65.341/65.703) descritta nei Cap. 1, 2, 5, 6. Confermato incrociando `n_assets`/`n_features` in *tutti* i `run_results.json`: N=100/D=4950 esiste solo sotto `data_00_20_w252_s5`.
**Decisione dell'autore**: la tesi usa esclusivamente la pipeline a **362 asset / finestra 724 giorni**. Ogni riferimento a N=100/D=4950 nei Cap. 3 e 7 è quindi un **refuso da correggere**, non un secondo esperimento da documentare o preservare — sostituire con N=362 e, a seconda del dominio discusso, D=65.341 (raw) o D=65.703 (Cholesky).
**Conseguenza pratica**: la narrazione del Cap. 7 va riscritta usando le tabelle di A.8 (pipeline finale, 362/724, dominio Cholesky) — che raccontano una storia diversa e più favorevole al VAE rispetto a quanto scritto oggi (vedi B.3).
**Azione residua**: quando si riscrive il Cap. 7 in LaTeX, sostituire i numeri con quelli di A.8 e correggere ogni menzione di "100 asset"/"4950" nei Cap. 3 e 7 a 362/65.341 o 65.703.

### B.3 — RISOLTO (2026-07-06): non è vero in generale che il VAE ricostruisce peggio di PCA/LinearAE — dipende dal dominio
**Draft (Cap. 7, §7.2)**: afferma che il VAE ha MSE sistematicamente peggiore di PCA e LinearAE.
**Codice**: vero nel **dominio raw** (K=20 test: VAE 6.33e-3 vs PCA/LinearAE ≈1.78e-4); **invertito nel dominio Cholesky** (K=20 test: VAE 1.55e-4 < PCA 2.73e-4 < LinearAE 3.38e-4; l'AE resta comunque il migliore in assoluto, 7.19e-5).
**Decisione dell'autore, confermata guardando le tabelle di confronto**: il dominio Cholesky (quello definitivo, dichiarato nel Cap. 1) è la narrazione corretta da usare — **il VAE non è peggiore di PCA/LinearAE**, anzi li batte entrambi a K=20, pur restando dietro l'AE. Il Cap. 7 nella sua forma attuale afferma l'opposto e va corretto, non solo "etichettato per dominio".
**Azione residua**: riscrivere §7.2 del Cap. 7 usando le tabelle di A.8 (dominio Cholesky): AE migliore in assoluto a ogni K, PCA≈LinearAE (Bourlard-Kamp confermato), **VAE meglio di PCA/LinearAE ma peggio di AE** a K=20. Il trade-off da raccontare non è più "il VAE ricostruisce peggio di tutto", ma "il VAE rinuncia a un po' di fedeltà rispetto all'AE deterministico (che non genera nulla di utile) in cambio della capacità generativa" — comunque restando competitivo con i benchmark lineari.

### B.4 — CONFERMATO ma NON PRIORITARIO (2026-07-06): la validazione dei 5 stylized facts sul generato è oggi solo su singolo scenario
**Draft**: la scaletta descrive un test "rigoroso" dei 5 stylized facts sulle matrici generate.
**Codice**: come descritto in A.9, il notebook 14 confronta sempre **un solo scenario sintetico per volta** contro la matrice reale odierna, mai una statistica aggregata sui 100/1000 scenari generati; nel caso più citato (idx=705, run `_10_loss`) lo scenario scelto è per giunta il più estremo/anomalo dell'intero insieme secondo Mahalanobis.
**Decisione dell'autore**: limite confermato, ma da lasciare così per ora (non prioritario). Quando si scriverà il Cap. 8, va comunque descritto come un case-study qualitativo su scenari selezionati, **non** come una validazione statistica esaustiva su tutto l'insieme generato — per non promettere un rigore che il codice attuale non fornisce. Se in futuro si vorrà rafforzarlo, l'evidenza aggregata naturale sarebbe: percentuale di scenari su 1000 con Perron-Frobenius soddisfatta, distribuzione dell'esponente γ, istogramma pooled delle correlazioni pairwise su tutti gli scenari vs reale.

### B.5 — CONFERMATO (2026-07-06): procedura generativa reale più sofisticata di quella descritta nel Cap. 4
**Draft (Cap. 4, §4.4 e Cap. 7 §7.3)**: descrive la generazione come campionamento diretto dal prior, $\mathbf z_{sim}\sim\mathcal N(0,I)$.
**Codice**: la generazione usata per produrre gli scenari effettivamente analizzati (A.9) è un **block-bootstrap storico degli incrementi latenti giornalieri** attorno allo stato attuale, non un campionamento indipendente dal prior. È una scelta metodologica più raffinata (ancorata alla dinamica storica recente) che merita di essere descritta esplicitamente, perché cambia il significato di "scenario generato": non è "una matrice plausibile qualsiasi", ma "una proiezione a 1 mese dallo stato di mercato attuale, coerente con la variabilità storica osservata nello spazio latente".
**Azione**: aggiornare Cap. 4 (o il futuro Cap. 8) per descrivere accuratamente questa procedura, distinguendola dal semplice prior sampling.

### B.6 — RISOLTO (fa fede il codice): coefficiente della loss "Divergenza di Jeffreys" è 1/2, non 1/4
**Draft**: `loss_VAE.md` deriva $\mathcal L_{Recon}=\frac14[\mathrm{Tr}(R_{recon}^{-1}R_{orig})+\mathrm{Tr}(R_{orig}^{-1}R_{recon})-2k]$, cioè la **media** delle due KL direzionali ($D_{sym}=\frac12(D_{KL}(P\|Q)+D_{KL}(Q\|P))$).
**Codice** (`07_VAE_cholesky_loss.ipynb`, funzione `vae_loss`, effettivamente eseguito per addestrare tutte le run `*_loss`): implementa `jeffreys_div = 0.5 * (trace1 + trace2 - 2*n_assets)` — coefficiente **1/2**.
**Decisione**: il codice è la fonte di verità (è quello effettivamente usato per addestrare i modelli citati nei risultati). Il coefficiente **1/2 è quello corretto/da documentare**; è `loss_VAE.md` che va corretto, non il codice. Chiarimento matematico per la riscrittura del documento: 1/2 corrisponde alla **somma** delle due divergenze KL direzionali ($D_{KL}(P\|Q)+D_{KL}(Q\|P)$, senza la media $\frac12(\cdot)$ finale) — una convenzione alternativa ma standard in letteratura per la "divergenza di Jeffreys" (talvolta chiamata anche J-divergence), distinta dalla convenzione "media" che il documento deriva. Non è quindi un errore concettuale, ma una discrepanza tra la convenzione descritta a parole in `loss_VAE.md` (media, →1/4) e quella realmente implementata (somma, →1/2).
**Azione residua**: aggiornare `loss_VAE.md` per derivare/dichiarare esplicitamente la convenzione "somma" (coefficiente 1/2), così che il documento descriva esattamente cosa è stato addestrato.

### B.7 — Quale checkpoint VAE è "il" modello finale?
**Codice**: notebook diversi puntano a run diverse senza un criterio dichiarato di scelta: il notebook di confronto finale (`12_compare_models`) usa `VAE_20dim_cholesky_06_loss` (K=20); il notebook di generazione (`13_VAE_generation`) usa `VAE_20dim_cholesky_10_loss`; esistono inoltre varianti **non-"_loss"** (`VAE_20dim_cholesky_01`, `_02`) con MSE test ancora più basso (7.36e-5, quasi al livello dell'AE) rispetto a tutte le varianti con loss di Jeffreys. Nessun notebook motiva esplicitamente la scelta finale.
**Azione**: Dylan deve decidere e dichiarare esplicitamente nella tesi quale run rappresenta "il" VAE del progetto (e perché), invece di lasciare che notebook diversi usino run diverse.

### B.8 — CONFERMATO (2026-07-06): split "senza temporal gap" (Cap. 5) è vero solo a livello di regime, non di leakage a livello di finestra
**Draft (Cap. 5, §5.3)**: presenta il random-shuffle come soluzione che garantisce un test set rappresentativo, senza discutere overlap tra finestre.
**Codice**: vedi A.4 — con stride=10 su finestre di 724 giorni, finestre a indice quasi-adiacente condividono ~98.6% delle osservazioni; lo split casuale non impone alcuna distanza minima tra indici assegnati a split diversi.
**Azione**: aggiungere una nota esplicita sul limite (leakage a livello di singola finestra, distinto dal distribution-shift di regime che il random-shuffle effettivamente risolve), eventualmente in "Limiti del modello attuale" nelle Conclusioni.

### B.9 — RISOLTO (2026-07-06, chiarito dall'autore): N=362, D=65.341 e D=65.703 non sono in conflitto, misurano cose diverse
**Draft (Cap. 5, §5.4)**: riporta $D=N(N-1)/2=65341$ come input canonico di "PCA, modelli deterministici e VAE", senza distinguere esplicitamente dominio raw/Cholesky.
**Chiarimento dell'autore**: **N=362** è il numero di asset usati per calcolare le matrici di correlazione (non cambia mai nella pipeline finale). **D=65.341** è la dimensione del vettore flattenato nel **dominio Raw** (triangolo inferiore della correlazione, diagonale esclusa perché sempre 1). **D=65.703** è la dimensione usata **da quando Cholesky è stata adottata come parametrizzazione definitiva** (triangolo inferiore del fattore L, diagonale inclusa perché informativa) — coerente con l'evoluzione metodologica descritta in A.6 (raw → Cholesky per garantire la PSD).
**Azione residua**: nel Cap. 5, §5.4 riportare $D=65.341$ esplicitamente come cifra del dominio raw (fase preliminare/diagnostica) e $D=65.703$ come dimensione canonica della pipeline definitiva (dominio Cholesky), invece di citare solo 65.341 come se fosse l'unico valore rilevante.

### B.10 — Altri dettagli minori emersi, utili se si vuole essere molto precisi
- Le metriche di errore (MSE/MAE/Frobenius) in tutti i notebook sono calcolate sull'intera matrice N×N, non solo sul triangolo superiore univoco (A.7) — non cambia i confronti relativi, ma va tenuto a mente se si citano valori assoluti confrontandoli con altre fonti/convenzioni.
- Diversi run mostrano "posterior collapse" parziale nel VAE (poche dimensioni latenti con varianza non-trascurabile, la maggioranza collassata vicino a 0) — un fenomeno noto in letteratura VAE che potrebbe meritare una menzione nei limiti del modello.
- La classificazione settoriale GICS è ottenuta interamente via LLM (Gemini) senza validazione incrociata con un provider dati ufficiale — da menzionare come possibile limite se i settori GICS vengono usati per argomentazioni quantitative (es. nella struttura gerarchica, Cap. 3).
- Il notebook 12 confronta i modelli usando `DATASET='train'` nell'ultima esecuzione salvata (esistono comunque anche i CSV per il test set, generati in un'altra esecuzione) — assicurarsi che ogni tabella citata in tesi sia etichettata correttamente come train o test.

---

## Indice dei file sorgente più rilevanti (per ri-verifiche puntuali)

- Dataset finale: `data/processed/data_00_20/dataset/data_00_20_w724_s10/dataset_info.json`
- Metriche per singolo run: `models/data_00_20_w724_s10/{PCA,linearAE,AE,VAE}/*/run_results.json`
- Tabelle di confronto finali (dominio Cholesky): `results_cholesky/data_00_20_w724_s10/comparison_{train,test}/*.csv`
- Tabelle di confronto (dominio raw): `results/data_00_20_w724_s10/comparison_{train,test}/*.csv`
- Scenari generati e analisi stylized facts: `generated_matrices/data_00_20_w724_s10/VAE_*_loss/data_00_20_w724_s1/forecast_scenarios_*/`
- Loss Jeffreys: `loss_VAE.md` vs `notebooks/07_VAE_cholesky_loss.ipynb`
