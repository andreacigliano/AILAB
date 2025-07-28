#Questo codice mostra diverse tecniche di prompt engineering:

#Prompt con ruolo specifico: Definire chiaramente il ruolo dell'AI
#Prompt strutturati: Richiedere output in formati specifici
#Few-shot learning: Fornire esempi nel prompt
#Concatenazione: Utilizzare risposte precedenti per domande successive
#Controllo della temperatura: Regolare la creatività delle risposte
#Prompt con contesto: Fornire informazioni di contesto per migliori risultati

#Per ogni esempio, puoi personalizzare:

#La temperatura per controllare la creatività
#Il formato dell'output richiesto
#Gli esempi forniti
#Il contesto aggiuntivo
#I parametri del modello


from openai import OpenAI
import os

# Inizializzazione del client
client = OpenAI(
  api_key="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"
)

# ESEMPIO 1: Prompt per analisi del sentimento
def analyze_sentiment(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sei un esperto di analisi del sentimento. Rispondi solo con: POSITIVO, NEGATIVO, o NEUTRO."},
            {"role": "user", "content": f"Analizza il sentimento del seguente testo: {text}"}
        ]
    )
    return response.choices[0].message.content

# ESEMPIO 2: Prompt per generazione di codice
def generate_python_code(task):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """Sei un esperto programmatore Python. 
            Genera solo codice Python valido con commenti.
            Non includere spiegazioni al di fuori dei commenti nel codice."""},
            {"role": "user", "content": f"Scrivi una funzione Python che: {task}"}
        ],
        temperature=0.2  # Ridotto per output più deterministico
    )
    return response.choices[0].message.content

# ESEMPIO 3: Prompt per riassunto strutturato
def structured_summary(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """Crea un riassunto strutturato con:
            - Punti chiave
            - Argomenti principali
            - Conclusioni
            Usa questo formato esatto."""},
            {"role": "user", "content": f"Riassumi il seguente testo: {text}"}
        ]
    )
    return response.choices[0].message.content

# ESEMPIO 4: Prompt per traduzione con contesto
def context_aware_translation(text, context, target_language):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"""Sei un traduttore esperto verso {target_language}. 
            Considera attentamente il contesto fornito prima di tradurre."""},
            {"role": "user", "content": f"Contesto: {context}\nTraduci il seguente testo: {text}"}
        ]
    )
    return response.choices[0].message.content

# ESEMPIO 5: Prompt per domande concatenate
def multi_step_analysis(topic):
    messages = [
        {"role": "system", "content": "Sei un esperto analista. Rispondi in modo conciso."},
        {"role": "user", "content": f"Argomento da analizzare: {topic}"}
    ]
    
    # Prima domanda
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages + [{"role": "user", "content": "Quali sono i 3 aspetti principali?"}]
    )
    aspects = response.choices[0].message.content
    messages.append({"role": "assistant", "content": aspects})
    
    # Seconda domanda basata sulla risposta precedente
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages + [{"role": "user", "content": "Approfondisci il primo aspetto."}]
    )
    return aspects, response.choices[0].message.content

# ESEMPIO 6: Prompt con esempi few-shot
def text_classifier(text, categories):
    examples = """
    Testo: "Il nuovo smartphone ha un'ottima fotocamera"
    Categoria: TECNOLOGIA

    Testo: "La ricetta richiede 200g di farina"
    Categoria: CUCINA
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Classifica il testo in una delle seguenti categorie: {categories}"},
            {"role": "user", "content": f"Esempi di classificazione:\n{examples}\n\nTesto: {text}\nCategoria:"}
        ]
    )
    return response.choices[0].message.content

# Esempio di utilizzo
if __name__ == "__main__":
    try:
        # Test analisi sentimento
        sentiment = analyze_sentiment("Il prodotto è eccezionale!")
        print(f"Sentimento: {sentiment}")
        
        # Test generazione codice
        code = generate_python_code("calcola il fattoriale di un numero")
        print(f"Codice generato:\n{code}")
        
        # Test riassunto
        summary = structured_summary("Lorem ipsum dolor sit amet...")
        print(f"Riassunto:\n{summary}")
        
        # Test traduzione
        translation = context_aware_translation(
            "The application crashed",
            "Software development context",
            "Italiano"
        )
        print(f"Traduzione: {translation}")
        
        # Test analisi multi-step
        aspects, detail = multi_step_analysis("Cambiamento climatico")
        print(f"Aspetti principali:\n{aspects}\nDettaglio:\n{detail}")
        
        # Test classificazione
        category = text_classifier(
            "La borsa è in rialzo del 2%",
            ["FINANZA", "SPORT", "TECNOLOGIA", "CULTURA"]
        )
        print(f"Categoria: {category}")
        
    except Exception as e:
        print(f"Errore: {str(e)}")