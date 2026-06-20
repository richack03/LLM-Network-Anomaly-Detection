# AI-Driven Wi-Fi Network Anomaly Detection & Explainable AI (XAI)

Questo repository contiene il codice sorgente e la documentazione di un sistema avanzato per il monitoraggio, la diagnostica e l'analisi della sicurezza delle infrastrutture di rete Wi-Fi. Il progetto integra algoritmi di Machine Learning per il rilevamento di anomalie (Anomaly Detection) e architetture multi-agente basate su Large Language Models (LLM) per fornire spiegazioni semantiche in linguaggio naturale (Explainable AI).

---
Riferimenti e Citazioni
Se utilizzi questo codice per scopi accademici o di ricerca, si prega di fare riferimento alla seguente fonte per i dati di rete:

S. P. Sone, J. Lehtomäki, e Z. Khan, "Wireless Traffic Usage Forecasting Using Real Enterprise Network Data: Analysis and Methods", supplementary material, Dataset, 2020. [Online]. Disponibile: https://ieee-dataport.org/open-access/wireless-network-traffic-time-series-enterprise-network

##  Informazioni sul Dataset

L'applicativo elabora telemetrie storiche reali per estrarre insight sulle reti aziendali e universitarie. Di seguito le specifiche tecniche dei dati supportati:

* Il sistema utilizza un dataset in formato MATLAB (.mat).
* Il dataset contiene i dati di misurazione reali raccolti da un totale di 470 access point (AP) distribuiti nel campus di Linnanmaa dell'Università di Oulu, in Finlandia.
* Le misurazioni includono gli ID, le date di raccolta, il numero di utenti, i dati sul traffico ricevuto, i dati sul traffico trasmesso e i nomi delle posizioni di ciascun AP[cite: 1].
* Ogni osservazione fornisce i valori del traffico e il numero di utenti a intervalli esatti di 10 minuti tra il 18 dicembre 2018 e il 12 febbraio 2019[cite: 1].
* Il file ha una dimensione di 25.1 MB (26,338,384 byte) e può essere letto da qualsiasi piattaforma o ambiente che supporti i file .mat, come MATLAB o Octave[cite: 1].
* I tre componenti principali all'interno del dataset sono: il numero di utenti connessi nel momento della raccolta (numb_users), il traffico ricevuto in byte (rxbytes) e il traffico trasmesso in byte (txbytes) per ogni AP[cite: 1].
* Nota fondamentale per la pre-elaborazione: i dati sul traffico ricevuto e trasmesso sono formattati come serie storiche cumulative; pertanto, è necessario calcolare la differenza tra 2 osservazioni consecutive per ottenere i valori effettivi ogni 10 minuti[cite: 1].

---

##  Architettura del Software

L'applicativo è strutturato secondo un'architettura modulare:

* `app_frontend.py`: Gestisce l'interfaccia utente (UI), la renderizzazione dei grafici e l'interazione dinamica con l'utente tramite Streamlit.
* `backend_app.py`: Contiene il motore logico del sistema, la pre-elaborazione dei dati, l'addestramento dei modelli (XGBoost e Isolation Forest) e l'orchestrazione delle chiamate API verso l'LLM per generare i report XAI.
* `requirements.txt`: L'elenco completo delle dipendenze necessarie per l'esecuzione dell'ambiente Python.

---

## Installazione e Setup

Esegui i seguenti comandi nel tuo terminale per configurare e avviare l'intero sistema in locale:

```bash
# 1. Clona il repository e posizionati nella cartella
git clone [https://github.com/richack03/LLM-Network-Anomaly-Detection.git](https://github.com/richack03/LLM-Network-Anomaly-Detection.git)
cd LLM-Network-Anomaly-Detection

# 2. Crea e attiva l'ambiente virtuale (Su Windows usa: venv\Scripts\activate)
python -m venv venv
source venv/bin/activate

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Configura la chiave API (Sostituisci con la tua chiave reale)
echo "OPENAI_API_KEY=la_tua_chiave_api_qui" > .env

# 5. Avvia l'applicazione Streamlit
streamlit run app_frontend.py
