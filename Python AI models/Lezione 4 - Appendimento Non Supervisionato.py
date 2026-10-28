# K-means clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Creazione di un dataset artificiale
np.random.seed(0)
X = np.vstack([
    np.random.normal(loc=0.5, scale=0.1, size=(50, 2)),
    np.random.normal(loc=1.5, scale=0.1, size=(50, 2)),
    np.random.normal(loc=0.5, scale=0.1, size=(50, 2)),
])

# Applicazione di K-means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# Visualizzazione dei cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', label="Dati clusterizzati")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', label="Centroidi")
plt.title("Cluster K-means")
plt.xlabel("Caratteristica 1")
plt.ylabel("Caratteristica 2")
plt.legend()
plt.show()


# PCA (Riduzione della dimensionalità)
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Caricamento del dataset di cifre
digits = load_digits()
X = digits.data

# Riduzione a 2 dimensioni con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualizzazione dei dati ridotti
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap='Spectral', s=10)
plt.colorbar(label="Cifra")
plt.title("Riduzione della dimensionalità con PCA")
plt.xlabel("Prima componente principale")
plt.ylabel("Seconda componente principale")
plt.show()


# Anomaly Detection con Isolation Forest
from sklearn.ensemble import IsolationForest

# Creazione di un dataset artificiale con alcune anomalie
np.random.seed(42)
normal_data = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
anomalies = np.random.uniform(low=-3, high=3, size=(10, 2))
X = np.vstack([normal_data, anomalies])

# Applicazione di Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=0)
predictions = iso_forest.fit_predict(X)

# Visualizzazione delle anomalie
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm', label="Dati classificati")
plt.title("Rilevamento anomalie con Isolation Forest")
plt.xlabel("Caratteristica 1")
plt.ylabel("Caratteristica 2")
plt.legend()
plt.show()

*****************************************************************************************************

L'algoritmo **Isolation Forest (Foresta di Isolamento)** è una tecnica di apprendimento non supervisionato utilizzata per rilevare anomalie. Si basa sull'idea che le anomalie sono più facili da isolare rispetto ai punti normali, poiché tendono a essere più rare e distanti dal resto dei dati.

### Principi di funzionamento
L'Isolation Forest utilizza alberi binari decisionali per isolare i punti dati. L'idea chiave è che le anomalie richiedono meno divisioni per essere isolate rispetto ai punti normali.

1. **Creazione di alberi di isolamento:**
   - Per ogni albero, si selezionano casualmente una feature (dimensione) e una soglia di suddivisione all'interno del range di valori di quella feature.
   - Questo processo divide ricorsivamente i dati fino a quando ogni punto è isolato in un nodo foglia.

2. **Misura della lunghezza del cammino:**
   - La lunghezza del cammino per isolare un punto è definita dal numero di suddivisioni necessarie per separarlo.
   - Le anomalie, che sono punti rari e distanti, richiedono un cammino più corto per essere isolate.

3. **Calcolo del punteggio di anomalia:**
   - Viene calcolato un punteggio di anomalia basato sulla lunghezza media del cammino per isolare un punto attraverso tutti gli alberi della foresta.
   - Il punteggio è scalato in modo che i valori vicini a 1 indichino anomalie, mentre valori vicini a 0 indicano punti normali.

### Formula del punteggio di anomalia
Il punteggio di anomalia di un punto \( x \) è dato da:

\[
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
\]

Dove:
- \( E(h(x)) \): lunghezza media del cammino per il punto \( x \) attraverso gli alberi.
- \( c(n) \): valore di normalizzazione basato sul numero di punti \( n \) nel dataset.
- Un valore \( s(x, n) \) vicino a 1 indica una forte probabilità che il punto sia un'anomalia.

### Vantaggi
- **Efficiente:** Adatto per grandi dataset grazie alla sua complessità lineare \( O(n \log n) \).
- **Non richiede ipotesi sui dati:** Non si basa su distribuzioni specifiche.
- **Gestisce bene dati ad alta dimensionalità.**

### Svantaggi
- **Sensibilità ai parametri:** I risultati dipendono dalla contaminazione presunta e dal numero di alberi.
- **Meno efficace per anomalie molto vicine al resto dei dati.**

### Applicazioni
- Rilevamento di frodi.
- Identificazione di guasti in sistemi industriali.
- Analisi di eventi rari in reti di dati o sistemi di sicurezza.

Se hai bisogno, posso fornire un esempio con visualizzazione grafica più dettagliata per illustrare il comportamento dell'algoritmo.