# Riassunti e citazioni dei paper — Materiale Tesi

Questo documento raccoglie, per ciascun paper PDF presente nella cartella `Materiale Tesi`, una sintesi di cosa tratta e una citazione pronta all'uso (voce BibTeX + citazione testuale in stile APA). I paper sono raggruppati per area tematica, seguendo la numerazione già usata nei file. Le informazioni bibliografiche sono state estratte direttamente dal testo dei PDF; dove un dato (volume, pagine, sede di pubblicazione) non era verificabile con certezza dal documento, è stato omesso o segnalato come tale.

> Nota: il file `messaggio 1.docx` e le sotto-cartelle `NN - riassunto ...` (i tuoi riassunti LaTeX già scritti) non sono stati inclusi in questa rassegna, in quanto non sono paper sorgente.

---

## Indice

1. [PCA e autoencoder](#1-pca-e-autoencoder)
2. [Fatti stilizzati delle correlazioni finanziarie e Random Matrix Theory](#2-fatti-stilizzati-delle-correlazioni-finanziarie-e-random-matrix-theory)
3. [Fatti stilizzati dei rendimenti e modelli fattoriali](#3-fatti-stilizzati-dei-rendimenti-e-modelli-fattoriali)
4. [VAE e GAN: fondamenti metodologici](#4-vae-e-gan-fondamenti-metodologici)
5. [Applicazioni di VAE/GAN a serie storiche e matrici di correlazione finanziarie](#5-applicazioni-di-vaegan-a-serie-storiche-e-matrici-di-correlazione-finanziarie)
6. [Forecasting della volatilità realizzata](#6-forecasting-della-volatilità-realizzata)

---

## 1. PCA e autoencoder

### 1.1 — `01 - PCA.pdf`
**Principal component analysis: a review and recent developments**
Ian T. Jolliffe, Jorge Cadima — 2016 — *Philosophical Transactions of the Royal Society A*, 374, 20150202. DOI: [10.1098/rsta.2015.0202](https://doi.org/10.1098/rsta.2015.0202)

**Di cosa parla.** Review di riferimento sulla Principal Component Analysis, scritta da uno dei massimi esperti del settore. Introduce la PCA come tecnica di riduzione della dimensionalità che costruisce componenti principali non correlate, ottenute come combinazioni lineari delle variabili originali che massimizzano successivamente la varianza spiegata. Il contributo principale è derivare la PCA sia come problema agli autovalori/autovettori della matrice di covarianza (o correlazione), sia in modo equivalente tramite SVD della matrice dei dati centrata, mostrando la connessione fra i due approcci; discute inoltre numerose varianti recenti (PCA robusta, sparsa, ecc.). È il riferimento metodologico fondativo per tutta la parte di tesi su PCA/RMT applicata alle correlazioni finanziarie (autovalori, loadings, varianza spiegata, "market mode").

```bibtex
@article{jolliffe2016pca,
  author  = {Jolliffe, Ian T. and Cadima, Jorge},
  title   = {Principal component analysis: a review and recent developments},
  journal = {Philosophical Transactions of the Royal Society A},
  year    = {2016},
  volume  = {374},
  number  = {2065},
  pages   = {20150202},
  doi     = {10.1098/rsta.2015.0202}
}
```

**Citazione testuale (APA):** Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202. https://doi.org/10.1098/rsta.2015.0202

---

### 1.2 — `1804.10253v3.pdf`
**From Principal Subspaces to Principal Components with Linear Autoencoders**
Elad Plaut — 2018 — arXiv:1804.10253 [stat.ML] (v3)

**Di cosa parla.** Chiarisce il legame formale tra autoencoder lineari e PCA. È noto che un autoencoder con un solo hidden layer, attivazione lineare e costo quadratico apprende pesi che generano lo stesso sottospazio principale della PCA, ma senza produrre loading vector ortonormali e ordinati per varianza. L'autore mostra che applicando la SVD alla matrice dei pesi del layer di decodifica si recuperano esattamente (a meno del segno) i loading vector della PCA classica, e verifica la proprietà su MNIST e CUB-200-2011. Non tratta dati finanziari, ma è utile in tesi per motivare rigorosamente perché lo spazio latente di un VAE (che è, in fondo, un autoencoder) non coincide automaticamente con le componenti principali/RMT usate per interpretare le correlazioni di mercato.

```bibtex
@misc{plaut2018principal,
  title         = {From Principal Subspaces to Principal Components with Linear Autoencoders},
  author        = {Plaut, Elad},
  year          = {2018},
  eprint        = {1804.10253},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  note          = {arXiv:1804.10253v3}
}
```

**Citazione testuale (APA):** Plaut, E. (2018). From principal subspaces to principal components with linear autoencoders. *arXiv preprint* arXiv:1804.10253.

---

## 2. Fatti stilizzati delle correlazioni finanziarie e Random Matrix Theory

### 2.1 — `02 - fatti stilizzati delle corr. finanziarie - Random Matrix Theory per finanza.pdf`
**Financial Applications of Random Matrix Theory: a short review**
Jean-Philippe Bouchaud, Marc Potters — 2009 — arXiv:0910.1205 [q-fin.ST]

**Di cosa parla.** Mini-review didattica delle applicazioni della Random Matrix Theory (RMT) all'analisi delle matrici di correlazione empiriche in finanza. Richiama la distribuzione di Marčenko-Pastur come benchmark per lo spettro "nullo" di autovalori quando N (asset) e T (osservazioni) sono grandi, e mostra come separare segnale (market mode, settori) da rumore statistico confrontando lo spettro empirico delle correlazioni azionarie con tale previsione. Copre anche PCA/eigenportfolios, code pesanti dei rendimenti, effetto leva, memoria lunga della volatilità e tecniche di "pulizia" (cleaning) delle matrici di correlazione per portafogli alla Markowitz. Riferimento fondamentale per validare (in tesi) la struttura di correlazione — reale o generata da modelli VAE/GAN — rispetto al rumore atteso.

```bibtex
@misc{bouchaud2009rmt,
  author        = {Bouchaud, Jean-Philippe and Potters, Marc},
  title         = {Financial Applications of Random Matrix Theory: a short review},
  year          = {2009},
  eprint        = {0910.1205},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.ST},
  url           = {https://arxiv.org/abs/0910.1205}
}
```

**Citazione testuale (APA):** Bouchaud, J.-P., & Potters, M. (2009). Financial applications of Random Matrix Theory: A short review. *arXiv preprint* arXiv:0910.1205.

---

### 2.2 — `03 - fatti stilizzati delle corr. finanziarie - struttura gerarchica delle correlazioni.pdf`
**Hierarchical Structure in Financial Markets**
Rosario N. Mantegna — 1998 — arXiv:cond-mat/9802256

**Di cosa parla.** Contributo fondativo dell'econofisica sulla struttura di correlazione dei mercati. Analizza la correlazione dei rendimenti giornalieri dei titoli DJIA e S&P 500 (1989–1995) e definisce una metrica euclidea a partire dal coefficiente di correlazione, d(i,j) = 1 − ρ²ᵢⱼ, per costruire un Minimal Spanning Tree e il relativo albero ultrametrico. Mostra che i titoli si raggruppano spontaneamente per settore economico, confermando fattori comuni sottostanti. È il precursore diretto delle tecniche di filtraggio delle matrici di correlazione (RMT, PCA) e un riferimento essenziale per validare la struttura gerarchica riprodotta da modelli generativi.

```bibtex
@article{mantegna1998hierarchical,
  author        = {Mantegna, Rosario N.},
  title         = {Hierarchical Structure in Financial Markets},
  year          = {1998},
  eprint        = {cond-mat/9802256},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.stat-mech}
}
```

**Citazione testuale (APA):** Mantegna, R. N. (1998). Hierarchical structure in financial markets. *arXiv preprint* cond-mat/9802256.

---

### 2.3 — `04 - fatti stilizzati delle corr. finanziarie - proprietà matrici corr..pdf`
**Universal and Non-Universal Properties of Cross-Correlations in Financial Time Series**
Vasiliki Plerou, Parameswaran Gopikrishnan, Bernd Rosenow, Luís A. Nunes Amaral, H. Eugene Stanley — 1999 — *Physical Review Letters*, 83(7), 1471–1474. DOI: [10.1103/PhysRevLett.83.1471](https://doi.org/10.1103/PhysRevLett.83.1471)

**Di cosa parla.** Applica la RMT alla matrice di correlazione dei rendimenti a 30 minuti delle 1000 maggiori azioni USA (1994-95). Confronta la distribuzione degli autovalori e le statistiche spettrali (spaziature, varianza del numero, rigidità spettrale) con le previsioni dell'ensemble ortogonale gaussiano (GOE). Mostra che la maggior parte dello spettro è indistinguibile dal rumore casuale, mentre pochi autovalori "fuori dal bulk" contengono informazione genuina; tramite l'inverse participation ratio individua autovettori "estesi" (bulk) e "localizzati" (settori, code), fenomeno analogo alla localizzazione di Anderson. Lavoro fondativo per l'uso di RMT/PCA nel denoising delle correlazioni finanziarie.

```bibtex
@article{plerou1999universal,
  author        = {Plerou, Vasiliki and Gopikrishnan, Parameswaran and Rosenow, Bernd and Amaral, Lu{\'\i}s A. Nunes and Stanley, H. Eugene},
  title         = {Universal and Non-Universal Properties of Cross-Correlations in Financial Time Series},
  journal       = {Physical Review Letters},
  year          = {1999},
  volume        = {83},
  number        = {7},
  pages         = {1471--1474},
  doi           = {10.1103/PhysRevLett.83.1471},
  eprint        = {cond-mat/9902283},
  archiveprefix = {arXiv}
}
```

**Citazione testuale (APA):** Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., & Stanley, H. E. (1999). Universal and non-universal properties of cross-correlations in financial time series. *Physical Review Letters*, 83(7), 1471–1474. https://doi.org/10.1103/PhysRevLett.83.1471

---

### 2.4 — `05 - fatti stilizzati delle corr. finanziarie - RMT e analisi spettrale matr. corr. finanziarie.pdf`
**A Random Matrix Approach to Cross-Correlations in Financial Data**
Vasiliki Plerou, Parameswaran Gopikrishnan, Bernd Rosenow, Luís A. Nunes Amaral, Thomas Guhr, H. Eugene Stanley — 2002 — *Physical Review E*, 65, 066126 (preprint arXiv:cond-mat/0108023, 2001). DOI: [10.1103/PhysRevE.65.066126](https://doi.org/10.1103/PhysRevE.65.066126)

**Di cosa parla.** Estende e approfondisce il lavoro precedente (2.3): applica la RMT alle cross-correlazioni tra ampi panieri di titoli USA, sia a 30 minuti (1994-97) sia giornalieri (1962-96), confrontando lo spettro empirico con matrici di Wishart/Marchenko-Pastur. Conferma che la grande maggioranza degli autovalori è coerente col rumore atteso (GOE), mentre circa il 2% degli autovalori più grandi porta informazione stabile riconducibile a un fattore di mercato comune e a settori industriali. Analizza inoltre stabilità temporale degli autovettori e applicazioni al filtraggio del rumore per portafogli più stabili. È il riferimento metodologico segnale/rumore su cui si basa gran parte della letteratura successiva su denoising delle correlazioni e validazione di matrici sintetiche.

```bibtex
@article{plerou2002randommatrix,
  author        = {Plerou, Vasiliki and Gopikrishnan, Parameswaran and Rosenow, Bernd and Amaral, Lu{\'i}s A. Nunes and Guhr, Thomas and Stanley, H. Eugene},
  title         = {A Random Matrix Approach to Cross-Correlations in Financial Data},
  journal       = {Physical Review E},
  year          = {2002},
  volume        = {65},
  pages         = {066126},
  doi           = {10.1103/PhysRevE.65.066126},
  eprint        = {cond-mat/0108023},
  archivePrefix = {arXiv}
}
```

**Citazione testuale (APA):** Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., Guhr, T., & Stanley, H. E. (2002). A random matrix approach to cross-correlations in financial data. *Physical Review E*, 65, 066126. https://doi.org/10.1103/PhysRevE.65.066126

---

## 3. Fatti stilizzati dei rendimenti e modelli fattoriali

### 3.1 — `06 - fatti stilizzati dei rendimenti finanziari.pdf`
**Empirical properties of asset returns: stylized facts and statistical issues**
Rama Cont — 2001 — *Quantitative Finance*, 1(2), 223–236

**Di cosa parla.** Rassegna sistematica degli undici "fatti stilizzati" dei rendimenti finanziari: assenza di autocorrelazione lineare, code pesanti, asimmetria guadagni/perdite, gaussianità aggregazionale, volatility clustering, code pesanti condizionali anche dopo GARCH, decadimento lento dell'autocorrelazione dei rendimenti assoluti, leverage effect, correlazione volume/volatilità, asimmetria tra scale temporali. Propone questi fatti come vincoli empirici "model-free" che ogni modello stocastico dei prezzi dovrebbe soddisfare. La sezione 6, sulle correlazioni cross-asset, cita esplicitamente lo studio RMT di Laloux-Cizeau-Bouchaud-Potters, collegando direttamente questo paper alla letteratura PCA/RMT. Fonte fondamentale sia per definire gli stylized facts usati per validare modelli generativi (VAE/GAN) sia per il forecasting della volatilità.

```bibtex
@article{cont2001stylizedfacts,
  author    = {Cont, Rama},
  title     = {Empirical properties of asset returns: stylized facts and statistical issues},
  journal   = {Quantitative Finance},
  year      = {2001},
  volume    = {1},
  number    = {2},
  pages     = {223--236},
  publisher = {IOP Publishing}
}
```

**Citazione testuale (APA):** Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223–236.

---

### 3.2 — `1-s2.0-S0304405X19301151-main.pdf`
**Characteristics are covariances: A unified model of risk and return**
Bryan T. Kelly, Seth Pruitt, Yinan Su — 2019 — *Journal of Financial Economics*, 134, 501–524. DOI: [10.1016/j.jfineco.2019.05.001](https://doi.org/10.1016/j.jfineco.2019.05.001)

**Di cosa parla.** Propone l'Instrumented Principal Component Analysis (IPCA), che estende la PCA standard permettendo a loadings dinamici di dipendere da caratteristiche osservabili dei titoli (dimensione, book-to-market, momentum), usate come strumenti per fattori di rischio latenti. Il framework unifica i modelli a fattori pre-specificati (es. Fama-French) e le tecniche di analisi fattoriale/PCA sui rendimenti, offrendo un test per distinguere compensazione del rischio da vere "anomalie". Stimato via minimi quadrati alternati (ALS) su oltre 12.000 titoli USA (1962-2014): pochi fattori latenti e poche caratteristiche spiegano quasi tutta la performance. Rilevante come riferimento sui limiti della PCA standard (loadings statici) applicata ai rendimenti azionari, motivando approcci più flessibili (denoising RMT, modelli generativi VAE/GAN).

```bibtex
@article{kelly2019characteristics,
  author  = {Kelly, Bryan T. and Pruitt, Seth and Su, Yinan},
  title   = {Characteristics are covariances: A unified model of risk and return},
  journal = {Journal of Financial Economics},
  year    = {2019},
  volume  = {134},
  pages   = {501--524},
  doi     = {10.1016/j.jfineco.2019.05.001}
}
```

**Citazione testuale (APA):** Kelly, B. T., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics*, 134, 501–524. https://doi.org/10.1016/j.jfineco.2019.05.001

---

## 4. VAE e GAN: fondamenti metodologici

### 4.1 — `07 - VAE.pdf`
**Tutorial on Variational Autoencoders**
Carl Doersch — 2016 — arXiv:1606.05908 [stat.ML] (v3, 2021)

**Di cosa parla.** Tutorial didattico (non un articolo di ricerca originale) che spiega in modo rigoroso ma accessibile la teoria dei Variational Autoencoder. Parte dai modelli a variabili latenti e dalla massimizzazione della verosimiglianza P(X) = ∫P(X|z;θ)P(z)dz, arriva alla costruzione dell'Evidence Lower BOund (ELBO), introduce la rete di inferenza Q(z|X) e il "reparameterization trick" che rende possibile l'addestramento end-to-end via backpropagation. Esempi empirici su cifre scritte a mano, volti, numeri civici, CIFAR. Riferimento metodologico imprescindibile per citare correttamente qualunque applicazione di VAE alla generazione di dati finanziari sintetici, anche se il paper stesso non tratta temi finanziari.

```bibtex
@misc{doersch2016vae,
  title         = {Tutorial on Variational Autoencoders},
  author        = {Doersch, Carl},
  year          = {2016},
  eprint        = {1606.05908},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  note          = {v3, revised January 3, 2021},
  url           = {https://arxiv.org/abs/1606.05908}
}
```

**Citazione testuale (APA):** Doersch, C. (2016). Tutorial on variational autoencoders. *arXiv preprint* arXiv:1606.05908.

---

### 4.2 — `08 - GAN.pdf`
**NIPS 2016 Tutorial: Generative Adversarial Networks**
Ian Goodfellow — 2016 — arXiv:1701.00160 (v4, 2017)

**Di cosa parla.** Report scritto del tutorial tenuto da Ian Goodfellow (uno degli inventori delle GAN) a NIPS 2016. Spiega perché studiare i modelli generativi, confronta le GAN con altri approcci (inclusi modelli a variabili latenti come i VAE), descrive in dettaglio il gioco minimax generatore/discriminatore, discute le frontiere di ricerca (instabilità dell'addestramento, mode collapse, GAN condizionali) e i modelli allo stato dell'arte per la generazione di immagini. Sistematizzazione autorevole del framework adversariale su cui si basano i modelli finanziari specializzati (CorrGAN, QuantGANs, TimeGAN), pur non trattando esso stesso dati finanziari.

```bibtex
@misc{goodfellow2016gantutorial,
  author        = {Goodfellow, Ian},
  title         = {{NIPS} 2016 Tutorial: Generative Adversarial Networks},
  year          = {2016},
  eprint        = {1701.00160},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1701.00160}
}
```

**Citazione testuale (APA):** Goodfellow, I. (2016). NIPS 2016 tutorial: Generative adversarial networks. *arXiv preprint* arXiv:1701.00160.

---

## 5. Applicazioni di VAE/GAN a serie storiche e matrici di correlazione finanziarie

### 5.1 — `09 - Generazione di serie temporali finanziarie con GAN.pdf`
**Quant GANs: Deep Generation of Financial Time Series**
Magnus Wiese, Robert Knobloch, Ralf Korn, Peter Kretschmer — 2019 — arXiv:1907.06673 [q-fin.MF] (v2)

**Di cosa parla.** Propone i "Quant GANs", modello data-driven basato su GAN per generare la dinamica realistica dei prezzi (serie univariate, applicato all'S&P 500). Usa reti convoluzionali temporali dilatate (Temporal Convolutional Networks, tipo WaveNet) sia per generatore sia per discriminatore, in grado di catturare dipendenze a lungo raggio (volatility clusters) garantendo stazionarietà e parallelizzabilità. Introduce le Stochastic Volatility Neural Networks (SVNN), un generatore strutturato in blocco volatilità/drift + innovazione; dimostra risultati teorici sull'esistenza dei momenti e usa la trasformazione di Lambert W per code pesanti. Validato confrontando istogrammi dei rendimenti, autocorrelazione e leverage effect fra percorsi generati e dati storici, contro un benchmark GARCH(1,1). Non tratta PCA/RMT/correlazioni multi-asset (è univariato), ma è un riferimento metodologico chiave per l'uso di GAN nella generazione sintetica di serie storiche finanziarie e per la verifica quantitativa degli stylized facts.

```bibtex
@misc{wiese2019quantgans,
  title         = {Quant {GANs}: Deep Generation of Financial Time Series},
  author        = {Wiese, Magnus and Knobloch, Robert and Korn, Ralf and Kretschmer, Peter},
  year          = {2019},
  eprint        = {1907.06673},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.MF},
  note          = {arXiv:1907.06673v2}
}
```

**Citazione testuale (APA):** Wiese, M., Knobloch, R., Korn, R., & Kretschmer, P. (2019). Quant GANs: Deep generation of financial time series. *arXiv preprint* arXiv:1907.06673.

---

### 5.2 — `Marti(19)_CorrGAN_sampling realistic financial correlation matrices using Generative Adversarial Networks.pdf`
**CorrGAN: Sampling Realistic Financial Correlation Matrices Using Generative Adversarial Networks**
Gautier Marti — 2019 — arXiv:1910.09504 [q-fin.ST] (v2)

**Di cosa parla.** Propone CorrGAN, primo metodo documentato in letteratura per campionare matrici di correlazione finanziarie realistiche via GAN. Affronta il problema di generare matrici che rispettino gli stylized facts noti: distribuzione delle correlazioni pairwise spostata verso valori positivi, spettro degli autovalori compatibile con Marchenko-Pastur (autovalore di "mercato" dominante + autovalori settoriali), proprietà di Perron-Frobenius, struttura gerarchica e Minimum Spanning Tree scale-free. Addestra un DCGAN (Deep Convolutional GAN) su circa 10.000 matrici di correlazione empiriche S&P 500, dopo aver riordinato gli asset via clustering gerarchico per garantire invarianza per permutazione; proietta poi i campioni generati sulla varietà delle vere matrici di correlazione con alternating projections. I risultati riproducono fedelmente gran parte degli stylized facts, con discrepanze nelle code. Paper chiave: collega esplicitamente RMT, clustering gerarchico e deep generative modeling in uno dei primi esempi concreti di GAN applicata alle correlazioni finanziarie.

```bibtex
@article{marti2019corrgan,
  author  = {Marti, Gautier},
  title   = {CorrGAN: Sampling Realistic Financial Correlation Matrices Using Generative Adversarial Networks},
  journal = {arXiv preprint arXiv:1910.09504},
  year    = {2019}
}
```

**Citazione testuale (APA):** Marti, G. (2019). CorrGAN: Sampling realistic financial correlation matrices using generative adversarial networks. *arXiv preprint* arXiv:1910.09504.

---

### 5.3 — `10 - Simulating realistic correlation matrices for financial applications  correlation matrices with the Perron Frobenius property.pdf`
**Simulating realistic correlation matrices for financial applications: correlation matrices with the Perron–Frobenius property**
Amelie Hüttner, Jan-Frederik Mai — 2019 (online 2018) — *Journal of Statistical Computation and Simulation*, 89(2), 315–336. DOI: [10.1080/00949655.2018.1546861](https://doi.org/10.1080/00949655.2018.1546861)

**Di cosa parla.** Affronta la simulazione (non neurale, ma algoritmico-analitica) di matrici di correlazione realistiche per backtesting e gestione del rischio. Richiama gli stylized facts delle matrici empiriche: primo autovalore dominante (>30% della varianza, coerente con RMT), proprietà di Perron-Frobenius, correlazioni pairwise spostate verso valori positivi, struttura ad albero (MST) scale-free. Propone un algoritmo — estensione del metodo "randcorr" (Bendel-Mickey, rotazioni di Givens sulla decomposizione spettrale) — che genera esattamente matrici con proprietà di Perron-Frobenius e struttura di autovalori arbitraria; dimostra che la proporzione di matrici con tale proprietà è 2^(1−d) in dimensione d. Utile come benchmark/dato di validazione (alternativo o complementare a VAE/GAN) per matrici di correlazione sintetiche con proprietà spettrali realistiche.

```bibtex
@article{huttner2019simulating,
  author  = {H{\"u}ttner, Amelie and Mai, Jan-Frederik},
  title   = {Simulating realistic correlation matrices for financial applications: correlation matrices with the {P}erron--{F}robenius property},
  journal = {Journal of Statistical Computation and Simulation},
  year    = {2019},
  volume  = {89},
  number  = {2},
  pages   = {315--336},
  doi     = {10.1080/00949655.2018.1546861},
  url     = {https://doi.org/10.1080/00949655.2018.1546861}
}
```

**Citazione testuale (APA):** Hüttner, A., & Mai, J.-F. (2019). Simulating realistic correlation matrices for financial applications: correlation matrices with the Perron–Frobenius property. *Journal of Statistical Computation and Simulation*, 89(2), 315–336. https://doi.org/10.1080/00949655.2018.1546861

---

### 5.4 — `00 - ptf_sens_with_VAE.pdf`
**Quantifying Credit Portfolio sensitivity to asset correlations with interpretable generative neural networks**
Sergio Caprioli, Emanuele Cagliero, Riccardo Crupi (Intesa Sanpaolo S.p.A.) — 2024 (preprint 2023) — *Journal of Risk Model Validation*, 18(1) (2024); versione preprint: arXiv:2309.08652 (anche presentato al workshop AIABI'23, CEUR Workshop Proceedings vol. 3650). DOI: [10.21314/JRMV.2024.002](https://doi.org/10.21314/JRMV.2024.002)

**Di cosa parla.** Propone un metodo per quantificare la sensibilità del Value-at-Risk (VaR) di un portafoglio crediti alle variazioni della matrice di correlazione tra asset, generando matrici di correlazione sintetiche con reti neurali generative. Riprende l'approccio GAN di Marti (CorrGAN, cfr. 5.2) ma lo sostituisce con un Variational Autoencoder addestrato su 206 matrici di correlazione storiche (44 indici azionari, rolling window di 100 mesi, 1997-2022), ottenendo uno spazio latente bidimensionale interpretabile. Contributo duplice: (1) le due dimensioni latenti catturano fattori interpretabili legati a PCA/RMT (la prima correlata al primo autovalore/intensità della correlazione di mercato, la seconda alla stabilità nel tempo degli autovettori); (2) il decoder viene usato per costruire una superficie di VaR sulla griglia latente e, tramite bootstrap (semplice e a blocchi) delle coordinate latenti, stimare la distribuzione del VaR a 1 anno senza ripetute simulazioni Monte Carlo. Le matrici sintetiche vengono validate sugli stylized facts (Marchenko-Pastur, Perron-Frobenius, struttura gerarchica, MST scale-free). Paper molto rilevante per il confronto diretto VAE vs GAN nella generazione di matrici di correlazione finanziarie; marginale invece il collegamento al forecasting della volatilità (qui non trattato: il focus è sul VaR di portafoglio crediti).

```bibtex
@article{caprioli2024ptfvae,
  author  = {Caprioli, Sergio and Cagliero, Emanuele and Crupi, Riccardo},
  title   = {Quantifying credit portfolio sensitivity to asset correlations with interpretable generative neural networks},
  journal = {Journal of Risk Model Validation},
  volume  = {18},
  number  = {1},
  year    = {2024},
  doi     = {10.21314/JRMV.2024.002}
}

@misc{caprioli2023ptfvaepreprint,
  author        = {Caprioli, Sergio and Cagliero, Emanuele and Crupi, Riccardo},
  title         = {Quantifying Credit Portfolio sensitivity to asset correlations with interpretable generative neural networks},
  year          = {2023},
  eprint        = {2309.08652},
  archiveprefix = {arXiv},
  primaryclass  = {q-fin.RM},
  note          = {Preprint corrispondente al PDF analizzato; presentato anche al workshop AIABI'23, CEUR Workshop Proceedings, Vol. 3650}
}
```

**Citazione testuale (APA):** Caprioli, S., Cagliero, E., & Crupi, R. (2024). Quantifying credit portfolio sensitivity to asset correlations with interpretable generative neural networks. *Journal of Risk Model Validation*, 18(1). https://doi.org/10.21314/JRMV.2024.002 (Versione preprint open-access: Caprioli, Cagliero & Crupi, 2023, arXiv:2309.08652)

---

## 6. Forecasting della volatilità realizzata

### 6.1 — `11 - Forecasting realized volatility with spillover effects.pdf`
**Forecasting realized volatility with spillover effects: Perspectives from graph neural networks**
Chao Zhang, Xingyue Pu, Mihai Cucuringu, Xiaowen Dong — 2025 — *International Journal of Forecasting*, 41, 377–397 (online 2024). DOI: [10.1016/j.ijforecast.2024.09.002](https://doi.org/10.1016/j.ijforecast.2024.09.002)

**Di cosa parla.** Propone GNNHAR, estensione non parametrica del modello Heterogeneous Autoregressive (HAR) basata su graph neural networks (GNN), per prevedere la volatilità realizzata di titoli USA (componenti DJIA 30 e S&P 100), incorporando esplicitamente effetti di spillover tra asset collegati in un grafo finanziario. Contributo triplice: (i) i vicini "multi-hop" nel grafo non offrono chiaro vantaggio predittivo rispetto ai soli vicini a zero/un hop; (ii) modellare gli spillover in modo non lineare (aggregazione GNN) invece che lineare (modello GHAR) migliora sensibilmente l'accuratezza, soprattutto a breve orizzonte; (iii) allenare con una loss basata sulla quasi-verosimiglianza (QL) invece del MSE produce miglioramenti sostanziali per la migliore gestione dell'eteroschedasticità. Rilevante per la parte di tesi su forecasting della volatilità realizzata; offre uno spunto complementare (grafi) rispetto a PCA/RMT (matrici di correlazione) per rappresentare la dipendenza cross-asset. Non tratta modelli generativi (VAE/GAN).

```bibtex
@article{zhang2025gnnhar,
  author  = {Zhang, Chao and Pu, Xingyue and Cucuringu, Mihai and Dong, Xiaowen},
  title   = {Forecasting realized volatility with spillover effects: Perspectives from graph neural networks},
  journal = {International Journal of Forecasting},
  year    = {2025},
  volume  = {41},
  pages   = {377--397},
  doi     = {10.1016/j.ijforecast.2024.09.002}
}
```

**Citazione testuale (APA):** Zhang, C., Pu, X., Cucuringu, M., & Dong, X. (2025). Forecasting realized volatility with spillover effects: Perspectives from graph neural networks. *International Journal of Forecasting*, 41, 377–397. https://doi.org/10.1016/j.ijforecast.2024.09.002

---

## Bibliografia completa (BibTeX)

Blocco unico con tutte le voci, pronto da incollare in un file `.bib`:

```bibtex
@article{jolliffe2016pca,
  author  = {Jolliffe, Ian T. and Cadima, Jorge},
  title   = {Principal component analysis: a review and recent developments},
  journal = {Philosophical Transactions of the Royal Society A},
  year    = {2016}, volume = {374}, number = {2065}, pages = {20150202},
  doi     = {10.1098/rsta.2015.0202}
}

@misc{plaut2018principal,
  title = {From Principal Subspaces to Principal Components with Linear Autoencoders},
  author = {Plaut, Elad}, year = {2018},
  eprint = {1804.10253}, archivePrefix = {arXiv}, primaryClass = {stat.ML},
  note = {arXiv:1804.10253v3}
}

@misc{bouchaud2009rmt,
  author = {Bouchaud, Jean-Philippe and Potters, Marc},
  title = {Financial Applications of Random Matrix Theory: a short review},
  year = {2009}, eprint = {0910.1205}, archivePrefix = {arXiv}, primaryClass = {q-fin.ST},
  url = {https://arxiv.org/abs/0910.1205}
}

@article{mantegna1998hierarchical,
  author = {Mantegna, Rosario N.},
  title = {Hierarchical Structure in Financial Markets},
  year = {1998}, eprint = {cond-mat/9802256}, archivePrefix = {arXiv}, primaryClass = {cond-mat.stat-mech}
}

@article{plerou1999universal,
  author = {Plerou, Vasiliki and Gopikrishnan, Parameswaran and Rosenow, Bernd and Amaral, Lu{\'\i}s A. Nunes and Stanley, H. Eugene},
  title = {Universal and Non-Universal Properties of Cross-Correlations in Financial Time Series},
  journal = {Physical Review Letters}, year = {1999}, volume = {83}, number = {7}, pages = {1471--1474},
  doi = {10.1103/PhysRevLett.83.1471}, eprint = {cond-mat/9902283}, archiveprefix = {arXiv}
}

@article{plerou2002randommatrix,
  author = {Plerou, Vasiliki and Gopikrishnan, Parameswaran and Rosenow, Bernd and Amaral, Lu{\'i}s A. Nunes and Guhr, Thomas and Stanley, H. Eugene},
  title = {A Random Matrix Approach to Cross-Correlations in Financial Data},
  journal = {Physical Review E}, year = {2002}, volume = {65}, pages = {066126},
  doi = {10.1103/PhysRevE.65.066126}, eprint = {cond-mat/0108023}, archivePrefix = {arXiv}
}

@article{cont2001stylizedfacts,
  author = {Cont, Rama},
  title = {Empirical properties of asset returns: stylized facts and statistical issues},
  journal = {Quantitative Finance}, year = {2001}, volume = {1}, number = {2}, pages = {223--236},
  publisher = {IOP Publishing}
}

@article{kelly2019characteristics,
  author = {Kelly, Bryan T. and Pruitt, Seth and Su, Yinan},
  title = {Characteristics are covariances: A unified model of risk and return},
  journal = {Journal of Financial Economics}, year = {2019}, volume = {134}, pages = {501--524},
  doi = {10.1016/j.jfineco.2019.05.001}
}

@misc{doersch2016vae,
  title = {Tutorial on Variational Autoencoders},
  author = {Doersch, Carl}, year = {2016},
  eprint = {1606.05908}, archivePrefix = {arXiv}, primaryClass = {stat.ML},
  note = {v3, revised January 3, 2021}, url = {https://arxiv.org/abs/1606.05908}
}

@misc{goodfellow2016gantutorial,
  author = {Goodfellow, Ian},
  title = {{NIPS} 2016 Tutorial: Generative Adversarial Networks},
  year = {2016}, eprint = {1701.00160}, archivePrefix = {arXiv}, primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/1701.00160}
}

@misc{wiese2019quantgans,
  title = {Quant {GANs}: Deep Generation of Financial Time Series},
  author = {Wiese, Magnus and Knobloch, Robert and Korn, Ralf and Kretschmer, Peter},
  year = {2019}, eprint = {1907.06673}, archivePrefix = {arXiv}, primaryClass = {q-fin.MF},
  note = {arXiv:1907.06673v2}
}

@article{marti2019corrgan,
  author = {Marti, Gautier},
  title = {CorrGAN: Sampling Realistic Financial Correlation Matrices Using Generative Adversarial Networks},
  journal = {arXiv preprint arXiv:1910.09504}, year = {2019}
}

@article{huttner2019simulating,
  author = {H{\"u}ttner, Amelie and Mai, Jan-Frederik},
  title = {Simulating realistic correlation matrices for financial applications: correlation matrices with the {P}erron--{F}robenius property},
  journal = {Journal of Statistical Computation and Simulation}, year = {2019}, volume = {89}, number = {2}, pages = {315--336},
  doi = {10.1080/00949655.2018.1546861}, url = {https://doi.org/10.1080/00949655.2018.1546861}
}

@article{caprioli2024ptfvae,
  author = {Caprioli, Sergio and Cagliero, Emanuele and Crupi, Riccardo},
  title = {Quantifying credit portfolio sensitivity to asset correlations with interpretable generative neural networks},
  journal = {Journal of Risk Model Validation}, volume = {18}, number = {1}, year = {2024},
  doi = {10.21314/JRMV.2024.002}
}

@misc{caprioli2023ptfvaepreprint,
  author = {Caprioli, Sergio and Cagliero, Emanuele and Crupi, Riccardo},
  title = {Quantifying Credit Portfolio sensitivity to asset correlations with interpretable generative neural networks},
  year = {2023}, eprint = {2309.08652}, archiveprefix = {arXiv}, primaryclass = {q-fin.RM},
  note = {Preprint corrispondente al PDF analizzato; presentato anche al workshop AIABI'23, CEUR Workshop Proceedings, Vol. 3650}
}

@article{zhang2025gnnhar,
  author = {Zhang, Chao and Pu, Xingyue and Cucuringu, Mihai and Dong, Xiaowen},
  title = {Forecasting realized volatility with spillover effects: Perspectives from graph neural networks},
  journal = {International Journal of Forecasting}, year = {2025}, volume = {41}, pages = {377--397},
  doi = {10.1016/j.ijforecast.2024.09.002}
}
```
