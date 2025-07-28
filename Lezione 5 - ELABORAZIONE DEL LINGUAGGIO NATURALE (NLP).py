# Esempio 1: Tokenizzazione e Named Entity Recognition (NER)
############################################################
import spacy
import matplotlib.pyplot as plt
from collections import Counter

# Caricamento del modello di lingua di Spacy
nlp = spacy.load("en_core_web_sm")

# Testo di esempio
testo = "Barack Obama è stato il 44° presidente degli Stati Uniti d'America. Vive a Washington D.C."

# Elaborazione del testo
doc = nlp(testo)

# Estrazione dei token
tokens = [token.text for token in doc]
print("Token:", tokens)

# Named Entity Recognition (NER)
print("Entità nominate:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Grafico delle entità nominate
labels = [ent.label_ for ent in doc.ents]
conteggio = Counter(labels)

plt.bar(conteggio.keys(), conteggio.values())
plt.title("Distribuzione delle entità nominate")
plt.xlabel("Tipo di entità")
plt.ylabel("Frequenza")
plt.show()
############################################################

#Esempio 2: Analisi del Sentimento con TextBlob
############################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def analyze_sentiment_unsupervised():
    # Testi di esempio in italiano
    testi = [
        "Il prodotto è fantastico, lo consiglio vivamente!",
        "È stato terribile, non lo comprerei mai più.",
        "Non è male, ma potrebbe essere migliorato.",
        "Eccezionale servizio clienti, davvero soddisfatto!",
        "Pessima qualità, soldi sprecati.",
        "Nella media, niente di speciale.",
        "Ottimo rapporto qualità-prezzo!",
        "Mai più, esperienza deludente.",
        "Discreto prodotto, consegna puntuale.",
        "Magnifico, supera le aspettative!"
    ]

    # Dizionario di parole positive e negative per il confronto
    positive_words = set(['fantastico', 'eccezionale', 'ottimo', 'magnifico', 'soddisfatto'])
    negative_words = set(['terribile', 'pessima', 'deludente', 'mai'])

    # Vettorizzazione del testo usando TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=['il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'è', 'non']
    )
    X = vectorizer.fit_transform(testi)

    # Riduzione dimensionalità con SVD
    svd = TruncatedSVD(n_components=2)
    X_reduced = svd.fit_transform(X)

    # Clustering con K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)

    # Calcolo del "sentiment score" basato sulla presenza di parole positive/negative
    def calculate_sentiment(text):
        words = set(text.lower().split())
        positive_score = len(words.intersection(positive_words))
        negative_score = len(words.intersection(negative_words))
        return (positive_score - negative_score) / (positive_score + negative_score + 1)

    sentiment_scores = [calculate_sentiment(text) for text in testi]

    # Visualizzazione dei risultati
    plt.figure(figsize=(15, 5))

    # Plot 1: Scatter plot dei cluster
    plt.subplot(121)
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    for i in range(3):
        mask = clusters == i
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                   c=colors[i], label=f'Cluster {i}', alpha=0.6)
    plt.title('Clustering dei Testi')
    plt.xlabel('Prima componente')
    plt.ylabel('Seconda componente')
    plt.legend()

    # Plot 2: Sentiment scores
    plt.subplot(122)
    bars = plt.bar(range(len(testi)), sentiment_scores, 
                  color=[colors[c] for c in clusters])
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Analisi del Sentimento')
    plt.xlabel('Indice del testo')
    plt.ylabel('Punteggio sentimento')
    plt.xticks(range(len(testi)), [f'Testo {i+1}' for i in range(len(testi))], 
               rotation=45)

    # Aggiusta il layout
    plt.tight_layout()
    plt.show()

    # Stampa risultati dettagliati
    print("\nAnalisi dettagliata:")
    for i, (testo, score, cluster) in enumerate(zip(testi, sentiment_scores, clusters)):
        sentiment = "positivo" if score > 0 else "negativo" if score < 0 else "neutro"
        print(f"\nTesto {i+1}:")
        print(f"Contenuto: {testo}")
        print(f"Cluster: {cluster}")
        print(f"Sentiment score: {score:.2f} ({sentiment})")

if __name__ == "__main__":
    analyze_sentiment_unsupervised()
############################################################

#Esempio 3: Creazione di Word Cloud
############################################################
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Testo di esempio
testo = """
Il Natural Language Processing è un campo dell'Intelligenza Artificiale che permette ai computer di comprendere, 
interpretare e generare linguaggio naturale. Applicazioni includono traduzione automatica, chatbot, e analisi del sentimento.
"""

# Creazione del word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(testo)

# Visualizzazione del word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud NLP")
plt.show()
############################################################

#Esempio 4: Traduzione automatica con transformers
############################################################
from transformers import MarianMTModel, MarianTokenizer

# Caricamento del modello di traduzione
modello = "Helsinki-NLP/opus-mt-it-en"  # Modello per traduzione da italiano a inglese
tokenizer = MarianTokenizer.from_pretrained(modello)
model = MarianMTModel.from_pretrained(modello)

# Testo in italiano
testo_italiano = "L'intelligenza artificiale sta rivoluzionando molti settori."

# Tokenizzazione e traduzione
tokens = tokenizer(testo_italiano, return_tensors="pt", padding=True, truncation=True)
traduzione = model.generate(**tokens)
tradotto = tokenizer.decode(traduzione[0], skip_special_tokens=True)

print(f"Testo originale: {testo_italiano}")
print(f"Testo tradotto: {tradotto}")
