# 📚 AISE_chatbot — Local NotebookLM

NotebookLM è un assistente digitale che trasforma i tuoi documenti di studio in una knowledge base interrogabile in linguaggio naturale.  
Basato su tecniche di **Retrieval-Augmented Generation (RAG)**, ti permette di caricare PDF, fare domande, ricevere risposte con citazioni delle fonti, ottenere riassunti e guide allo studio, e persino ascoltare i contenuti tramite sintesi vocale.

---

## 👥 Per chi è pensato

- **Studenti**: possono caricare appunti, dispense o libri e ottenere spiegazioni, schemi di studio e quiz.  
- **Docenti e ricercatori**: strumento rapido per consultare i propri materiali didattici e scientifici, utile nella preparazione di lezioni o seminari.  

---

## 🧭 Obiettivo

Fornire un supporto allo studio **affidabile, trasparente e personalizzato**, senza sostituire il ruolo del docente o dello studente, ma agendo come **partner di apprendimento**.  
Le risposte sono sempre accompagnate da citazioni delle fonti, in modo che l’utente possa verificare e approfondire.

---

## 🔐 Principi di progettazione

- **Privacy**: i documenti vengono gestiti e indicizzati localmente in FAISS, senza invio a database esterni.  
- **Trasparenza**: ogni risposta include le fonti da cui è stata tratta.  
- **Affidabilità**: il modello è vincolato a usare solo i contenuti caricati, riducendo il rischio di hallucinations.  
- **Etica**: in linea con i principi europei **ALTAI** e **FASTEP** per un’IA responsabile.  

---

## 🛠️ Tecnologie utilizzate

- `streamlit` — interfaccia utente semplice e interattiva  
- `langchain` + `langchain_community` — orchestrazione pipeline RAG  
- `langchain_google_genai` + `google-generativeai` — integrazione con Gemini (LLM + embeddings)  
- `faiss-cpu` — vector store locale per gli embedding  
- `pypdf`, `PyMuPDF` — parsing documenti PDF  
- `pyttsx3` — sintesi vocale offline  
- `ragas` — valutazione RAG (faithfulness, relevance, precision, recall)  
- `opik` — monitoring e tracing interazioni  
- `datasets`, `pandas`, `openpyxl` — gestione ed esportazione dati  
- `python-dotenv` — gestione delle variabili d’ambiente  

---

## ⚙️ Installazione

1. **Clonare il repository**
   ```bash
   git clone https://github.com/alessiamanna/AISE_chatbot.git
   cd AISE_chatbot
   ```

2. **Creare ed attivare un ambiente virtuale Python**  
   Consigliato **Python 3.11** per compatibilità.

   **Su Linux/macOS**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

   **Su Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Installare le dipendenze**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurare le variabili d’ambiente**
   Creare un file `.env` a partire da `.env.example`:
   ```bash
   cp .env.example .env   # macOS/Linux
   copy .env.example .env # Windows (cmd)
   ```
   Compilare i token richiesti:
   ```
   GOOGLE_API_KEY=...
   OPIK_API_KEY=...
   OPIK_WORKSPACE=...
   ```

5. **Avvio del Chatbot**
   ```bash
   streamlit run app.py
   ```
   L’app sarà disponibile su [http://localhost:8501](http://localhost:8501).

---

## 🐳 Deploy con Docker

1. **Build dell’immagine**
   ```bash
   docker build -t aise-chatbot .
   ```

2. **Run del container**
   ```bash
   docker run --rm -p 8501:8501 --env-file .env aise-chatbot
   ```

3. **Docker Compose** (opzionale)
   ```bash
   docker compose up --build
   ```

---

## 📊 Testing e valutazione

AISE_chatbot integra strumenti per la valutazione delle performance RAG:

- **RAGAS** → metriche quantitative (faithfulness, answer relevance, context precision/recall)  
- **Opik** → tracciamento delle interazioni, feedback umano, auditing  

Gli script dedicati sono:
```bash
python ragas_eval.py
python eval_opik.py
```

---

## ☁️ Deploy online

L’app può essere distribuita su:
- **Streamlit Community Cloud**  
- Container in **Docker Hub** o orchestrati in Kubernetes 

---

## 🔮 Sviluppi futuri

- Integrazione di modelli open-source locali per maggiore privacy e indipendenza.  
- Supporto multimodale (immagini, video, audio).  
- Sistema di autenticazione multi-utente e gestione dei ruoli.  
- Ottimizzazione energetica e scalabilità.  

---

## ⚠️ Disclaimer

NotebookLM non sostituisce lo studio autonomo né il ruolo del docente.  
Le risposte sono generate in base ai documenti forniti, ma devono essere sempre verificate criticamente dall’utente.  

---

## 📄 Esempio `.env.example`

```env
# API keys
GOOGLE_API_KEY=your_google_genai_key_here
OPIK_API_KEY=your_opik_key_here
OPIK_WORKSPACE=your_workspace_name_here
```
