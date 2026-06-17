## Formulazione della Funzione di Costo Obiettivo per Matrici di Correlazione

L'addestramento dell'architettura Variational Autoencoder (VAE) per la generazione di scenari di mercato richiede una funzione di costo (loss function) in grado di catturare non solo l'errore di ricostruzione elemento per elemento, ma anche la complessa struttura geometrica e spettrale delle matrici di correlazione finanziarie. 

L'approccio standard basato sull'Errore Quadratico Medio (MSE) risulta sub-ottimale in questo dominio: trattando ogni elemento della matrice come indipendente, l'MSE ignora i vincoli di semi-definitezza positiva (PSD) e non pesa adeguatamente l'importanza degli autovalori. Per superare questa limitazione, la funzione di costo implementata sostituisce l'MSE con una metrica derivata dalla Teoria dell'Informazione: la **Divergenza di Jeffreys** (o Divergenza KL simmetrizzata).

La funzione di costo totale $\mathcal{L}_{Total}$ del modello è formulata come la somma pesata di due componenti principali, seguendo il framework del $\beta$-VAE:

$$\mathcal{L}_{Total} = \mathcal{L}_{Recon} + \beta \mathcal{L}_{KL}$$

Dove $\mathcal{L}_{Recon}$ rappresenta l'errore di ricostruzione strutturale, $\mathcal{L}_{KL}$ è il termine di regolarizzazione dello spazio latente e $\beta$ è un iperparametro che controlla il bilanciamento tra l'accuratezza della ricostruzione e la regolarità topologica della distribuzione latente.

### 1. Loss di Ricostruzione: Divergenza di Jeffreys ($\mathcal{L}_{Recon}$)

Assumendo che i rendimenti degli asset finanziari seguano una distribuzione normale multivariata centrata in zero, possiamo caratterizzare le dipendenze del mercato unicamente attraverso la matrice di correlazione (o covarianza). 

Definiamo:
* $R_{orig} \in \mathbb{R}^{k \times k}$: la matrice di correlazione originale (target).
* $R_{recon} \in \mathbb{R}^{k \times k}$: la matrice di correlazione ricostruita in output dal decodificatore.
* $k$: il numero totale di asset considerati.

Le corrispondenti distribuzioni di probabilità sono $P \sim \mathcal{N}(0, R_{orig})$ e $Q \sim \mathcal{N}(0, R_{recon})$. La divergenza di Kullback-Leibler (KL) standard da $P$ a $Q$ è definita come:

$$D_{KL}(P \parallel Q) = \frac{1}{2} \left[ \text{Tr}(R_{recon}^{-1} R_{orig}) - k + \ln\left(\frac{|R_{recon}|}{|R_{orig}|}\right) \right]$$

Poiché la divergenza KL è asimmetrica ($D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$), introduciamo una misura di distanza simmetrica calcolando la **Divergenza di Jeffreys** ($D_{sym}$), definita come la media aritmetica delle due divergenze direzionali:

$$D_{sym}(P, Q) = \frac{1}{2} \Big( D_{KL}(P \parallel Q) + D_{KL}(Q \parallel P) \Big)$$

Sostituendo le rispettive formule, i termini logaritmici dipendenti dai determinanti si elidono reciprocamente, in quanto $\ln(|R_{recon}| / |R_{orig}|) + \ln(|R_{orig}| / |R_{recon}|) = 0$. La formulazione finale della loss di ricostruzione si riduce quindi esclusivamente alla componente dipendente dalla traccia:

$$\mathcal{L}_{Recon} = \frac{1}{4} \left[ \text{Tr}(R_{recon}^{-1} R_{orig}) + \text{Tr}(R_{orig}^{-1} R_{recon}) - 2k \right]$$

**Giustificazione Finanziaria:** L'impiego delle matrici inverse all'interno delle tracce $\text{Tr}(R^{-1} \dots)$ garantisce che il modello venga fortemente penalizzato per gli errori commessi nella stima dei portafogli a varianza minima (corrispondenti agli autovalori più piccoli della matrice di correlazione). In ambito quantitativo, questi autovalori rappresentano i fattori di rischio intrinseco depurati dal rischio di mercato sistemico. A differenza dell'MSE, questa metrica preserva la struttura degli autovalori e garantisce che la geometria dei dati finanziari sia appresa coerentemente.

### 2. Regolarizzazione dello Spazio Latente ($\mathcal{L}_{KL}$)

Il secondo termine della funzione obiettivo costringe la distribuzione appresa dal codificatore $q_\phi(z|x)$ ad approssimare una distribuzione a priori (prior) $p(z)$, tipicamente assunta come una normale standard isotropica $\mathcal{N}(0, I)$. 

Assumendo che il codificatore produca vettori per la media $\mu$ e la varianza in scala logaritmica $\log(\sigma^2)$ di dimensione $d$ (la dimensione dello spazio latente), la divergenza KL analitica calcolata per ogni batch è:

$$\mathcal{L}_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

Durante la fase di inferenza e di visualizzazione degli scenari, si sfrutta unicamente il vettore deterministico della media $\mu$ per generare le proiezioni nello spazio latente, aggirando il campionamento stocastico per garantire la riproducibilità topologica dei regimi di mercato individuati.