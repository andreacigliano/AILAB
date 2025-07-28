import openai

# Imposta la tua API key
openai.api_key = "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"

# Definizione dell'agente AI basato su LLM
class AgenteLLM:
    def __init__(self, obiettivo):
        self.obiettivo = obiettivo
        self.storia = []

    def chiedi_llm(self, prompt):
        risposta = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Sei un agente AI autonomo con l'obiettivo: {self.obiettivo}"},
                {"role": "user", "content": prompt}
            ]
        )
        return risposta.choices[0].message.content

    def esegui(self, passi=3):
        prompt = f"Inizia a svolgere l'obiettivo: '{self.obiettivo}'. Qual è la prima azione che consigli?"

        for passo in range(passi):
            risposta = self.chiedi_llm(prompt)
            self.storia.append(risposta)

            print(f"Passo {passo+1}: {risposta}\n")

            prompt = f"Hai appena fatto: '{risposta}'. Quale sarà il prossimo passo per raggiungere l'obiettivo '{self.obiettivo}'?"

# Esempio di esecuzione
if __name__ == "__main__":
    obiettivo = "Scrivere un breve riassunto sulla Divina Commedia di Dante."
    agente = AgenteLLM(obiettivo)
    agente.esegui()


import openai

# Imposta la tua API key
openai.api_key = "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"

# Classe dell'agente AI con memoria e pianificazione dinamica
class AgenteIntelligente:
    def __init__(self, obiettivo):
        self.obiettivo = obiettivo
        self.memoria = []  # Memorizza le azioni precedenti e le risposte

    def chiedi_llm(self, prompt):
        risposta = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Sei un agente AI intelligente e autonomo. Il tuo obiettivo principale è: {self.obiettivo}. Usa il contesto fornito e la tua memoria per rispondere."},
                {"role": "user", "content": prompt}
            ]
        )
        return risposta.choices[0].message.content

    def pianifica_azione(self):
        prompt = f"Dato l'obiettivo '{self.obiettivo}' e le azioni già eseguite {self.memoria}, quale dovrebbe essere il prossimo passo più efficace?"
        return self.chiedi_llm(prompt)

    def aggiorna_memoria(self, azione, risultato):
        self.memoria.append({"azione": azione, "risultato": risultato})

    def esegui(self, passi=5):
        for passo in range(passi):
            azione = self.pianifica_azione()
            print(f"Passo {passo + 1} - Azione pianificata: {azione}")

            risultato = self.chiedi_llm(f"Esegui l'azione seguente: {azione}. Descrivi brevemente i risultati ottenuti.")
            print(f"Risultato dell'azione: {risultato}\n")

            self.aggiorna_memoria(azione, risultato)

# Esempio di utilizzo
if __name__ == "__main__":
    obiettivo = "Creare una guida introduttiva dettagliata sull'uso dell'AI agentica nella Pubblica Amministrazione."
    agente = AgenteIntelligente(obiettivo)
    agente.esegui(passi=4)

