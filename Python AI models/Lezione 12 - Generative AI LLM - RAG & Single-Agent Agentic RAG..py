pip install openai sentence-transformers PyPDF2 scikit-learn numpy

##################################################################################################
from google.colab import files

uploaded = files.upload()

# Move uploaded files to the `documents` directory
import shutil

docs_path = "/content/documents"
if not os.path.exists(docs_path):
    os.makedirs(docs_path)

for filename in uploaded.keys():
    shutil.move(filename, os.path.join(docs_path, filename))

print(f"Uploaded {len(uploaded.keys())} files successfully!")

##################################################################################################

import os
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import PyPDF2
import textwrap
from pathlib import Path

class DocumentProcessor:
    """
    Classe per il processamento dei documenti
    """
    def __init__(self, docs_dir: str):
        """
        Inizializza il processore di documenti
        
        Args:
            docs_dir: Directory contenente i documenti
        """
        self.docs_dir = Path(docs_dir)
        print("dir local",Path(docs_dir))

    def read_pdf(self, file_path: str) -> str:
        """
        Legge un file PDF e ne estrae il testo
        
        Args:
            file_path: Percorso del file PDF
            
        Returns:
            Testo estratto dal PDF
        """
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
        return text

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Divide il testo in chunk di dimensione specificata
        
        Args:
            text: Testo da dividere
            chunk_size: Dimensione desiderata dei chunk
            
        Returns:
            Lista di chunk di testo
        """
        return textwrap.wrap(text, chunk_size, break_long_words=False)

    def process_documents(self) -> List[Dict[str, Any]]:
        """
        Processa tutti i documenti nella directory
        
        Returns:
            Lista di documenti processati con relativi metadati
        """
        documents = []
        for file_path in self.docs_dir.glob('*.pdf'):
            text = self.read_pdf(str(file_path))
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'text': chunk,
                    'source': file_path.name,
                    'chunk_id': i
                })
        return documents

class EmbeddingEngine:
    """
    Classe per la gestione degli embedding dei documenti
    """
    def __init__(self):
        """
        Inizializza il motore di embedding usando SentenceTransformers
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = []
        self.documents = []

    def create_embeddings(self, documents: List[Dict[str, Any]]):
        """
        Crea gli embedding per i documenti forniti
        
        Args:
            documents: Lista di documenti da processare
        """
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        self.embeddings = self.model.encode(texts)

    def find_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Trova i documenti più simili alla query
        
        Args:
            query: Testo della query
            top_k: Numero di documenti da restituire
            
        Returns:
            Lista dei documenti più rilevanti
        """
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                **self.documents[idx],
                'similarity': similarities[idx]
            })
        return results

class RAGEngine:
    """
    Classe principale per il Retrieval Augmented Generation
    """
    def __init__(self, api_key: str, docs_dir: str):
        """
        Inizializza il motore RAG
        
        Args:
            api_key: Chiave API OpenAI
            docs_dir: Directory contenente i documenti
        """
        self.client = OpenAI(
          api_key=""
        )

        self.doc_processor = DocumentProcessor(docs_dir)
        
        self.embedding_engine = EmbeddingEngine()
        
        # Processa i documenti all'inizializzazione
        documents = self.doc_processor.process_documents()
        print("Doc processor: ",documents[1])
        self.embedding_engine.create_embeddings(documents)

    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Genera una risposta utilizzando RAG
        
        Args:
            query: Query dell'utente
            
        Returns:
            Dizionario contenente risposta e fonti utilizzate
        """
        # Recupera i documenti rilevanti
        relevant_docs = self.embedding_engine.find_similar(query)
        
        # Prepara il contesto per il modello
        context = "\n\n".join([f"Documento {i+1}:\n{doc['text']}" 
                             for i, doc in enumerate(relevant_docs)])
        
        # Prepara il prompt
        messages = [
            {"role": "system", "content": """Sei un assistente didattico esperto.
             Usa le informazioni fornite nel contesto per rispondere alle domande.
             Se le informazioni nel contesto non sono sufficienti, dillo chiaramente."""},
            {"role": "user", "content": f"""Contesto:\n{context}\n\nDomanda: {query}
             
             Rispondi alla domanda basandoti sulle informazioni fornite nel contesto."""}
        ]
        
        # Genera la risposta
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )
        
        return {
            'response': response.choices[0].message.content,
            'sources': [{'source': doc['source'], 'similarity': doc['similarity']} 
                       for doc in relevant_docs]
        }

def main():
    """
    Funzione principale di esempio
    """
    # Configurazione
    api_key="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"
    docs_dir = "/content/documents" # Directory contenente i PDF
    
    try:
        # Inizializza il motore RAG
        rag = RAGEngine(api_key, docs_dir)
        
        # Esempio di utilizzo
        while True:
            # Input utente
            query = input("\nInserisci la tua domanda (o 'exit' per uscire): ")
            if query.lower() == 'exit':
                break
                
            # Genera risposta
            result = rag.generate_response(query)
            
            # Stampa risultato
            print("\nRisposta:")
            print("-" * 50)
            print(result['response'])
            print("\nFonti utilizzate:")
            print("-" * 50)
            for source in result['sources']:
                print(f"- {source['source']} (similarità: {source['similarity']:.3f})")
            
    except Exception as e:
        print(f"Errore: {str(e)}")

if __name__ == "__main__":
    main()