# Studio di Architetture per la Ricostruzione e successiva Generazione di Matrici di Correlazione Finanziaria

## 1. Introduzione e Obiettivo del Progetto
L'obiettivo principale della tesi è lo sviluppo e la modellizzazione di architetture neurali generative — nello specifico **Autoencoder (AE)** e **Variational Autoencoder (VAE)** — applicate al contesto finanziario. Il focus è incentrato sulla capacità di questi modelli di apprendere la struttura sottostante delle matrici di correlazione empirica dei log-rendimenti azionari (aziende S&P 500, periodo storico dal 2000 al 2020). 

L'applicazione finale di questo framework risiede nel generare scenari di correlazione sintetici ma statisticamente e geometricamente realistici da utilizzare all'interno di simulazioni stocastiche **Monte Carlo** per la stima accurata del rischio (*VaR*).

---

## 2. Fase Sperimentale Iniziale (Approccio Statica)
Nella prima fase della ricerca, l'addestramento e la valutazione dei modelli sono stati impostati su un dataset strutturato in modo classico.

### 2.1 Configurazione del Dataset
* **Arco Temporale:** Dati storici giornalieri dal 2000 al 2020 (S&P 500).
* **Paniere di Asset:** Un insieme fisso e invariato di aziende selezionate ($N$ asset stabili su tutto l'orizzonte temporale, 362 asset totali).
* **Generazione delle Matrici:** Calcolo delle matrici di correlazione empirica di Pearson mediante una finestra mobile (*rolling window*) di **724 giorni lavorativi** e uno stride molto corto pari a **10 giorni**.
* **Overlap Temporale:** L'elevata sovrapposizione tra finestre contigue ha determinato un overlap informativo pari a circa il **98.5%**.

### 2.2 Evidenze Sperimentali e Problema dello Stallo Non Lineare
Durante il confronto delle performance di ricostruzione (valutate tramite *Mean Squared Error* - MSE), è emerso un comportamento inatteso:

$$\text{MSE}_{\text{PCA}} \approx \text{MSE}_{\text{Linear AE}} \approx \text{MSE}_{\text{Deep Non-Linear AE}}$$

L'introduzione di strati profondi e funzioni di attivazione non lineari (come ReLU o Tanh) **non ha prodotto alcun incremento significativo delle performance** rispetto al Linear Autoencoder o alla PCA (*Principal Component Analysis*) tradizionale. 

### 2.3 Analisi Teorica dello Stallo
L'equivalenza empirica tra modelli non lineari e PCA è riconducibile alla natura intrinseca del dataset iniziale:
1. **Invarianza Strutturale:** L'utilizzo di un paniere fisso di aziende e un overlap temporale iper-denso (98.5%) fa sì che le relazioni macroeconomiche inter-settoriali rimangano pressoché statiche tra un campione e l'altro.
2. **Linearità delle Feature:** I coefficienti di correlazione di Pearson estratti su asset fissi si muovono lungo uno spazio delle feature che non subisce deformazioni geometriche macroscopiche. Di conseguenza, lo spazio latente ottimale è di fatto un iperpiano piatto. 
3. **Collasso Lineare:** L'ottimizzatore della rete neurale profonda, per il principio di parsimonia, esclude i gradi di libertà non lineari e collassa sulla soluzione lineare equivalente alla PCA.

## 2.4 Dettaglio Tecnico e Confronto Matematico dei Modelli di Ricostruzione

Per comprendere i motivi dello stallo prestazionale, è necessario analizzare come la PCA, il Linear Autoencoder (LAE) e il Deep Autoencoder (AE) elaborano e ricostruiscono le matrici di correlazione empirica $R \in \mathbb{R}^{N \times N}$.

### 1. Principal Component Analysis (PCA)
La PCA opera attraverso la decomposizione in autovalori e autovettori della matrice di correlazione (o covarianza) media del dataset, oppure tramite la *Singular Value Decomposition* (SVD) applicata direttamente alle matrici del dataset spianate (*flattened*).

Dato un insieme di matrici di correlazione empirica trasformate in vettori $x \in \mathbb{R}^{d}$ (dove $d = N \times N$), la PCA cerca una matrice di proiezione ortogonale $W_k \in \mathbb{R}^{d \times k}$ (con $k < d$, che rappresenta la dimensione dello spazio latente) che massimizzi la varianza dei dati proiettati.

La ricostruzione lineare $\hat{x}$ della matrice avviene tramite:

$$\hat{x} = W_k W_k^T (x - \mu) + \mu$$

La PCA garantisce la soluzione analitica ottimale che minimizza il Mean Squared Error (MSE) sotto il vincolo di **ortogonalità** delle componenti della matrice di proiezione.

---

### 2. Linear Autoencoder (LAE)
Il Linear Autoencoder è una rete neurale priva di funzioni di attivazione non lineari (funzioni di identità in tutti i livelli). È composto da due matrici di peso principali:
* **Encoder ($W_e \in \mathbb{R}^{k \times d}$):** Mappa il vettore della matrice di correlazione $x$ nello spazio latente lineare $z \in \mathbb{R}^k$.
* **Decoder ($W_d \in \mathbb{R}^{d \times k}$):** Ricostruisce la matrice partendo dallo spazio latente.

L'operazione complessiva del LAE per ottenere la matrice ricostruita $\hat{x}$ è definita da:

$$\hat{x} = W_d (W_e x + b_e) + b_d$$

I parametri $[W_e, W_d, b_e, b_d]$ vengono ottimizzati stocasticamente (tramite algoritmi come *Adam* o *SGD*) minimizzando la loss di ricostruzione MSE sull'intero dataset di matrici:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{i=1}^{M} \|x_i - \hat{x}_i\|^2$$

#### Perché il Linear Autoencoder approssima (ed è equivalente a) la PCA?
Secondo il teorema fondamentale di **Bourlard e Kamp (1988)**, se un Autoencoder ha un solo strato nascosto ed è completamente lineare, **lo spazio sotteso dalle sue mappe di ricostruzione coincide esattamente con il sottospazio principale individuato dalla PCA**. 

Ci sono tuttavia due precisazioni matematiche importanti:
1. **Mancanza di Ortogonalità Diretta:** A differenza della PCA, le matrici di peso del LAE ($W_e$ e $W_d$) non sono costrette a essere ortogonali durante l'addestramento. La rete trova una base generica per l'iperpiano latente, non necessariamente le componenti principali ordinate per autovalore.
2. **Equivalenza nello Spazio di Ricostruzione:** Sebbene i pesi interni siano diversi (il LAE ammette infinite combinazioni di $W_e$ e $W_d$ che danno lo stesso risultato tramite rotazioni nello spazio latente), **l'output finale $\hat{x}$ e il valore minimo della Loss MSE del LAE saranno matematicamente identici a quelli della PCA**. Il LAE impara la stessa identica proiezione lineare della PCA.

---

### 3. Deep Non-Linear Autoencoder (AE)
Il Deep Autoencoder estende il modello lineare introducendo molteplici strati nascosti (*hidden layers*) e funzioni di attivazione non lineari $\sigma$ (ReLU negli Hidden Layer e ReLU/Tanh nell'Output Layer). L'obiettivo teorico è consentire alla rete di apprendere una rappresentazione compressa e non lineare delle matrici di correlazione, proiettando i dati su una varietà (*manifold*) a bassa dimensionalità non catturabile dalle semplici proiezioni ortogonali della PCA.

#### Architettura della Rete Implementata
L'architettura sviluppata prevede un restringimento progressivo e simmetrico delle dimensioni dei layer, strutturata come segue:

* **Encoder:** $$\text{Input } (x \in \mathbb{R}^d) \longrightarrow \text{Layer 1 } (2048) \longrightarrow \text{Layer 2 } (1024) \longrightarrow \text{Layer 3 } (512) \longrightarrow \text{Spazio Latente } (z \in \mathbb{R}^k)$$
  
* **Decoder:** $$\text{Spazio Latente } (z \in \mathbb{R}^k) \longrightarrow \text{Layer 4 } (512) \longrightarrow \text{Layer 5 } (1024) \longrightarrow \text{Layer 6 } (2048) \longrightarrow \text{Output } (\hat{x} \in \mathbb{R}^d)$$

#### Formalizzazione Matematica dell'Architettura
Considerando $h_e^{(i)}$ come l'output del livello $i$-esimo dell'Encoder e $h_d^{(i)}$ l'output del livello $i$-esimo del Decoder, il passaggio dei dati attraverso la rete è governato dalle seguenti equazioni di transizione:

**Fase di Encoding (Compressione):**
1. Primo strato nascosto ($d \to 2048$):  
   $$h_e^{(1)} = \sigma(W_{e1} x + b_{e1})$$
2. Secondo strato nascosto ($2048 \to 1024$):  
   $$h_e^{(2)} = \sigma(W_{e2} h_e^{(1)} + b_{e2})$$
3. Terzo strato nascosto ($1024 \to 512$):  
   $$h_e^{(3)} = \sigma(W_{e3} h_e^{(2)} + b_{e3})$$
4. Collo di bottiglia / Bottleneck ($512 \to k$):  
   $$z = \sigma(W_{ez} h_e^{(3)} + b_{ez})$$

**Fase di Decoding (Ricostruzione):**
5. Quarto strato nascosto ($k \to 512$):  
   $$h_d^{(1)} = \sigma(W_{d1} z + b_{d1})$$
6. Quinto strato nascosto ($512 \to 1024$):  
   $$h_d^{(2)} = \sigma(W_{d2} h_d^{(1)} + b_{d2})$$
7. Sesto strato nascosto ($1024 \to 2048$):  
   $$h_d^{(3)} = \sigma(W_{d3} h_d^{(2)} + b_{d3})$$
8. Strato di Output finale ($2048 \to d$):  
   $$\hat{x} = \sigma(W_{dx} h_d^{(3)} + b_{dx})$$

Dove $\sigma$ rappresenta la funzione di attivazione non lineare applicata elemento per elemento, mentre $W$ e $b$ indicano rispettivamente le matrici dei pesi e i vettori di bias di ciascun livello.

---

### Conclusioni sullo Stallo Prestazionale delle Matrici Fisse
Nonostante la notevole capacità computazionale e l'elevata profondità di questa architettura (in grado di mappare superfici altamente complesse), quando viene addestrata sul nostro dataset a paniere fisso con overlap tra matrici consecutive molto alto (circa del 98.5%), l'ottimizzatore si trova davanti a un problema geometrico degenere. 

Le strutture a blocchi non subiscono deformazioni macroscopiche (spostamenti, rotazioni o variazioni strutturali di topologia) lungo la serie storica. Di fronte a relazioni puramente statiche tra le feature (i coefficienti di Pearson tra gli stessi asset rimangono vincolati alla medesima struttura lineare di mercato), la capacità di "curvare lo spazio" offerta dai tre strati profondi non offre alcun vantaggio matematico per minimizzare l'errore quadratico medio. 

Di conseguenza, l'ottimizzatore apprende a impostare i pesi $W$ in modo da far lavorare le attivazioni $\sigma$ (ReLU/Tanh) quasi esclusivamente nella loro regione di linearità (oppure spegne i neuroni ridondanti), **costringendo di fatto questa complessa architettura profonda a collassare sulle stesse performance simulative del Linear Autoencoder e della PCA tradizionale (notare infatti che la PCA rimane comunque migliore rispetto al LAE/AE per quasi tutti i valori di K scelti)**.

---

### Conclusioni sullo Stallo Prestazionale delle Matrici Fisse
Quando questo Deep AE viene addestrato sulle nostre matrici a paniere fisso con overlap tra matrici consecutive del 98.5%, l'ottimizzatore si trova davanti a un dataset in cui la varianza geometrica è nulla. Le strutture a blocchi non si spostano né cambiano forma nel tempo.

Di fronte a relazioni puramente statiche e lineari tra le feature, la curvatura dello spazio latente offerta dalle funzioni $\sigma$ (ReLU/Tanh) non offre alcun grado di libertà utile a ridurre l'errore. La rete profonda impara a impostare i pesi degli strati intermedi in modo da far lavorare le attivazioni nella loro zona lineare (o spegne i neuroni ridondanti), **collassando di fatto sulle prestazioni del Linear Autoencoder e, di conseguenza, della PCA**.

---

## 3. Analisi della Letteratura: Il Framework *CorrGAN* (Marti, 2019)
Per sbloccare il potenziale delle reti non lineari, è stato condotto uno studio approfondito sul paper *CorrGAN* (Gautier Marti, 2019). L'autore affronta il problema generativo sotto un paradigma metodologico radicalmente diverso dal nostro approccio iniziale.

### 3.1 Data Augmentation Combinatoria e Clustering Gerarchico
In *CorrGAN*, l'autore non lavora su un paniere fisso. Il suo dataset di circa 10.000 matrici empiriche viene costruito tramite un campionamento casuale:
* Per ogni matrice, vengono estratti a sorte $N$ asset (es. $N=100$) dall'intero pool dello S&P 500 su una finestra di 1 anno.
* Prima di passare la matrice alla rete (una DCGAN), viene applicato un algoritmo di **Clustering Gerarchico** ($\pi_H$) per riordinare in modo deterministico righe e colonne.

### 3.2 Invarianza Geometrica ed Effetto Visivo
Grazie al clustering gerarchico, le matrici empiriche assumono una struttura geometrica fissa a blocchi compatti lungo la diagonale principale. La rete non è costretta a imparare le relazioni specifiche di singoli asset (es. Apple con Microsoft), ma impara a modellare la **"geometria universale"** dei mercati finanziari: come i blocchi settoriali nascono, si contraggono, si espandono e interagiscono in regime di normalità o di crisi. In questo scenario ad alta variabilità combinatoria, la non-linearità diventa fondamentale per mappare le deformazioni dello spazio geometrico delle matrici.

---

## 4. Nuova Proposta Metodologica per la Tesi
Alla luce dell'analisi dello stallo lineare e delle soluzioni evidenziate in letteratura, si propone un cambio di paradigma nella costruzione del dataset e nell'architettura di valutazione.

```text
+---------------------------------------+
|   Pool Globale di Asset (es. 500)     |
+---------------------------------------+
                   |
                   v (Campionamento Stratificato / Random)
+---------------------------------------+
|  Sotto-insieme casuale di N asset     |
+---------------------------------------+
                   |
                   v (Calcolo Correlazione)
+---------------------------------------+
|  Matrice Empirica Grezza (N x N)      |
+---------------------------------------+
                   |
                   v (Clustering Gerarchico)
+---------------------------------------+
|    Matrice Ordinata a Blocchi         |
+---------------------------------------+
                   |
                   v
+---------------------------------------+
|          PCA / LAE / AE / VAE         |
+---------------------------------------+
```

### 4.1 Ristrutturazione del Data Loader
Si ipotizza di modificare la pipeline di scomposizione dei dati secondo i seguenti step:
1. **Fissare una dimensione $N \times N$:** Definire una dimensione quadrata fissa (es. $100 \times 100$).
2. **Campionamento Combinatorio:** Per ogni campione, estrarre casualmente $N$ asset dal pool globale. Al fine di arricchire il dataset ed evitare il *survivorship bias*, potremmo includere anche aziende con serie storiche incomplete (es. nate dopo il 2000 o delistate prima del 2020), condizionandone il pescaggio unicamente alle finestre temporali in cui risultavano attive sul mercato.
3. **Clustering Gerarchico:** Applicare il riordinamento su ciascuna matrice estratta prima di sottoporla al collo di bottiglia dell'Autoencoder e del successivo VAE.

### 4.2 Gestione dello Sbilanciamento Settoriale (*Sampling Bias*)
Un potenziale rischio del campionamento puramente casuale è l'estrazione di matrici sbilanciate (es. un numero eccessivo di aziende dello stesso settore all'interno di una singola matrice), che porterebbe alla formazione di un "mega-cluster" dominante in grado di distorcere lo spazio latente del VAE.
Per mitigare questo fenomeno, potremmo implementare un **Campionamento Stratificato**: il data loader pescherà quote fisse di asset da macro-settori industriali predefiniti (es. 20 Tech, 20 Finance, 20 Energy...), garantendo una stabilità della struttura macro-settoriale e lasciando alla rete il solo compito di apprendere le fluttuazioni stocastiche delle intensità di correlazione.

---

## 5. Framework di Integrazione nel Risk Management (Monte Carlo)
L'adozione della metodologia stile *CorrGAN* comporta l'ottenimento di matrici generate dal VAE strutturate a blocchi ma matematicamente **"anonime"** (prive di etichette sui singoli asset). Per testare un portafoglio di investimento reale su tali matrici all'interno di una simulazione Monte Carlo, potremmo procedere secondo il seguente schema quantitativo:

```text
+-------------------------------+       +-------------------------------+
|   Matrice Anonima VAE (NxN)   |       |   Portafoglio Reale Aziende   |
|                               |       |          (es. 10 asset)       |
+-------------------------------+       +-------------------------------+
                |                                       |
                v (Isolamento Blocchi)                  v (Identificazione Settori)
+-------------------------------+       +-------------------------------+
| Proprietà Statistiche Cluster |       | Caratteristiche Macro-Settori |
+-------------------------------+       +-------------------------------+
                \                                       /
                 \---------------------v---------------/
                                       | 
                                       v (Linear Assignment Problem / Algoritmo Ungherese)
                        +-------------------------------+
                        | Mapping Ottimale Asset/Blocco |
                        +-------------------------------+
                                       |
                                       v (Sub-setting Righe/Colonne)
                        +-------------------------------+
                        | Sotto-Matrice Estratta        |
                        |           (es. 10x10)         |
                        +-------------------------------+
                                       |
                                       v
                        +-------------------------------+
                        | Simulazione Monte Carlo / VaR |
                        +-------------------------------+
```

### 5.1 Algoritmo di Mapping e Associazione dei Pesi
1. **Caratterizzazione dei Cluster Sintetici:** Sulla matrice anonima $N \times N$ generata dal VAE, si applica un algoritmo di clustering (es. *Agglomerative Clustering*) per isolare i blocchi geometrici, calcolando per ciascuno di essi la correlazione media intra-cluster e inter-cluster.
2. **Mapping Ottimale (Linear Assignment Problem):** Si mappano le caratteristiche dei settori macroeconomici del nostro portafoglio reale (es. la compattezza storica del Tech o la decorrelazione delle Utilities) sui blocchi sintetici anonimi del VAE che esibiscono proprietà geometriche affini. Questo accoppiamento ottimale può essere formalizzato matematicamente risolvendo il problema tramite l'**Algoritmo Ungherese**.
3. **Iniezione e Sotto-campionamento (*Sub-setting*):** Le aziende specifiche del portafoglio reale (es. se abbiamo un portafoglio di 10 titoli) vengono idealmente posizionate negli slot (righe/colonne) del blocco sintetico corrispondente al loro settore. Ai fini della simulazione stocastica, potremmo evitare di simulare l'intero mercato a $N$ variabili, andandone invece a **estrarre la sotto-matrice analoga $10 \times 10$** formata esclusivamente dalle righe e colonne assegnate ai nostri titoli.

### 5.2 Vantaggi del Framework
Questo approccio permette di estrarre sotto-matrici di covarianza per il Monte Carlo che contengono le relazioni specifiche del nostro portafoglio, ma che risultano strutturalmente condizionate dalle dinamiche globali e dai cambi di regime appresi dallo spazio latente del VAE.
