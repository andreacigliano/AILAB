import openai
import os
from typing import Dict, List, Any
import json
import time

class LLMAPIHandler:
    def __init__(self, api_key: str):
        """
        Inizializza l'handler per le chiamate API al modello linguistico
        
        Args:
            api_key (str): Chiave API per l'autenticazione
        """
        # Configurazione della chiave API
        openai.api_key = api_key
        
        # Parametri di default per le chiamate
        self.default_params = {
            "model": "gpt-3.5-turbo",  # Modello da utilizzare
            "temperature": 0.7,         # Controllo della creatività (0-1)
            "max_tokens": 1000,         # Lunghezza massima della risposta
            "top_p": 1,                 # Nucleus sampling
            "frequency_penalty": 0,      # Penalità per la ripetizione di token
            "presence_penalty": 0        # Penalità per la ripetizione di argomenti
        }

    def prepare_messages(self, prompt: str, system_prompt: str = None) -> List[Dict[str, str]]:
        """
        Prepara i messaggi per la chiamata API
        
        Args:
            prompt (str): Il prompt principale
            system_prompt (str, optional): Istruzioni di sistema opzionali
            
        Returns:
            List[Dict[str, str]]: Lista di messaggi formattati
        """
        messages = []
        
        # Aggiunge il prompt di sistema se fornito
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Aggiunge il prompt utente
        messages.append({"role": "user", "content": prompt})
        
        return messages

    def call_llm(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Esegue la chiamata API al modello linguistico
        
        Args:
            prompt (str): Il prompt principale
            system_prompt (str, optional): Istruzioni di sistema
            **kwargs: Parametri aggiuntivi per sovrascrivere i default
            
        Returns:
            Dict[str, Any]: Risposta del modello
        """
        try:
            # Prepara i parametri della chiamata
            params = self.default_params.copy()
            params.update(kwargs)
            
            # Prepara i messaggi
            messages = self.prepare_messages(prompt, system_prompt)
            
            # Esegue la chiamata API con retry in caso di errore
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = openai.ChatCompletion.create(
                        messages=messages,
                        **params
                    )
                    return response
                except openai.error.RateLimitError:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Backoff esponenziale
                        time.sleep(wait_time)
                    else:
                        raise
                        
        except Exception as e:
            print(f"Errore durante la chiamata API: {str(e)}")
            raise

    def extract_response(self, response: Dict[str, Any]) -> str:
        """
        Estrae il testo della risposta dal risultato della chiamata API
        
        Args:
            response (Dict[str, Any]): Risposta dell'API
            
        Returns:
            str: Testo della risposta
        """
        return response['choices'][0]['message']['content']

def main():
    """
    Esempio di utilizzo della classe LLMAPIHandler
    """
    # Carica la chiave API da variabile d'ambiente
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY non trovata nelle variabili d'ambiente")

    # Inizializza l'handler
    handler = LLMAPIHandler(api_key)

    # Esempio di prompt
    prompt = """
    Scrivi una funzione Python che calcoli il fattoriale di un numero.
    La funzione deve includere gestione degli errori e commenti esplicativi.
    """

    # Esempio di prompt di sistema
    system_prompt = """
    Sei un esperto programmatore Python. 
    Fornisci risposte chiare e ben commentate, seguendo le best practice di programmazione.
    """

    try:
        # Esegue la chiamata
        response = handler.call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5  # Riduce la creatività per output più precisi
        )

        # Estrae e stampa la risposta
        result = handler.extract_response(response)
        print("Risposta del modello:")
        print("-" * 50)
        print(result)
        print("-" * 50)

        # Salva la risposta su file
        with open("llm_response.txt", "w") as f:
            f.write(result)
        print("\nRisposta salvata in 'llm_response.txt'")

    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")

if __name__ == "__main__":
    main()