import os
import json
import scipy.io as sio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.ensemble import IsolationForest
import xgboost as xgb  
import streamlit as st
import altair as alt

# --- CONFIGURAZIONE INIZIALE ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configurazione pagina Streamlit
st.set_page_config(page_title="Network Anomaly Detector", page_icon="📡", layout="wide")

# Funzione per caricare il dataset una volta sola
@st.cache_data
def carica_dataset():
    return sio.loadmat('APs_dataset.mat')

try:
    dati_mat = carica_dataset()
except Exception as e:
    st.error(f"Impossibile caricare il file 'APs_dataset.mat'. Errore: {e}")
    st.stop()

# Estrazione della lista degli AP dal file .mat
lista_ap = [chiave for chiave in dati_mat.keys() if not chiave.startswith('__')]
lista_ap.sort()

# --- INTERFACCIA GRAFICA (GUI) ---
st.title(" Sistema Multi-Agente & ML per Anomaly Detection")
st.markdown("---")

# Layout per Input
col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    default_index = lista_ap.index("ap135172") if "ap135172" in lista_ap else 0
    target_ap = st.selectbox("Seleziona l'ID dell'Access Point da verificare:", options=lista_ap, index=default_index)

with col_input2:
    st.write("#") 
    avvia_pipeline = st.button("🚀 Avvia Pipeline Multi-Agente", use_container_width=True)

# SEZIONE DI ESECUZIONE 

if avvia_pipeline:
    with st.spinner(f"Avvio pipeline per {target_ap}... Elaborazione in corso..."):
        
        # --- 1. ESTRAZIONE DATI E CALCOLO STATISTICO (ROUTING) ---
        dati_ap = dati_mat[target_ap][0,0]
        luogo_ap = dati_ap['aploc'][0]
        utenti = dati_ap['numb_users'].flatten()
        rx = dati_ap['rxbytes'].flatten()
        tx = dati_ap['txbytes'].flatten()

        # Creazione DataFrame
        df = pd.DataFrame({'Utenti_Connessi': utenti, 'Traffico_IN': rx, 'Traffico_OUT': tx})
        
        # CALCOLO AUTOCORRELAZIONE
        autocorr_val = df['Traffico_OUT'].autocorr(lag=1)
        if pd.isna(autocorr_val):
            autocorr_val = 0.0

        # Panoramica  per l'Agente 1
        stats = f"""
        - AP: {target_ap} (Luogo: {luogo_ap})
        - Record totali: {len(df)}
        - Media utenti: {df['Utenti_Connessi'].mean():.2f} (Max: {df['Utenti_Connessi'].max()})
        - Traffico IN Max: {df['Traffico_IN'].max()} Byte
        - Indice di Autocorrelazione (Regolarità): {autocorr_val:.2f} (0 = Caotico, 1 = Molto Prevedibile)
        """

        # --- AGENTE 1 (Data Profiler) ---
        prompt_agente1 = "Sei un Data Profiler. Analizza queste statistiche di rete. Evidenzia se il traffico è regolare o caotico basandoti sull'Indice di Autocorrelazione e segnala anomalie logiche. Sii conciso."
        resp1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt_agente1}, {"role": "user", "content": stats}]
        )
        profilo_dati = resp1.choices[0].message.content

        # --- AGENTE 2 (ML Strategist) ---
        prompt_agente2 = """
        Sei un Architetto ML. Leggi il profilo dati e decidi l'algoritmo. 
        REGOLA RIGIDA: Se l'Indice di Autocorrelazione è >= 0.6, devi suggerire 'XGBoost' perché il traffico è ciclico e prevedibile. 
        Se l'Indice è < 0.6, devi suggerire 'IsolationForest' perché il traffico è irregolare e caotico.
        Giustifica la tua scelta.
        """
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt_agente2}, {"role": "user", "content": profilo_dati}]
        )
        strategia = resp2.choices[0].message.content

        # --- AGENTE 3 (Instruction Generator) ---
        prompt_agente3 = """
        Genera un JSON con le istruzioni per l'addestramento. 
        Usa ESCLUSIVAMENTE queste chiavi: 'task', 'model_type', 'features_to_use', 'reasoning'.
        Nella chiave 'model_type' scrivi ESATTAMENTE o "IsolationForest" o "XGBoost", estrapolandolo dalla strategia.
        Per 'features_to_use' usa: ["Utenti_Connessi", "Traffico_IN", "Traffico_OUT"].
        """
        resp3 = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[{"role": "system", "content": prompt_agente3}, {"role": "user", "content": strategia}]
        )
        istruzioni = json.loads(resp3.choices[0].message.content)
        modello_scelto = istruzioni['model_type']

        # --- FASE 4: ESECUZIONE DEL MODELLO (ROUTING ADATTIVO) ---
        features = istruzioni['features_to_use']
        
        if modello_scelto == "XGBoost":
            # Logica XGBoost: Prevediamo il traffico in base all'istante precedente
            df_xgb = df.copy()
            df_xgb['Traffico_OUT_lag'] = df_xgb['Traffico_OUT'].shift(1)
            df_xgb = df_xgb.dropna() # Rimuoviamo la prima riga vuota
            
            X_xgb = df_xgb[['Traffico_OUT_lag']]
            y_xgb = df_xgb['Traffico_OUT']
            
            modello = xgb.XGBRegressor(n_estimators=50, random_state=42)
            modello.fit(X_xgb, y_xgb)
            previsioni = modello.predict(X_xgb)
            
            # Calcoliamo l'errore: se la differenza tra realtà e previsione è enorme, è un'anomalia
            errori = np.abs(y_xgb - previsioni)
            soglia_errore = np.percentile(errori, 99) # Il top 1% degli errori sono anomalie
            
            # Riallineiamo al dataframe originale
            df['Anomalia'] = 1  
            df['Score_Anomalia'] = 0.0
            indici_anomali = df_xgb[errori > soglia_errore].index
            df.loc[indici_anomali, 'Anomalia'] = -1
            df.loc[df_xgb.index, 'Score_Anomalia'] = errori
            
        else:
            # Isolation Forest classico
            X = df[features]
            modello = IsolationForest(contamination=0.01, random_state=42)
            df['Anomalia'] = modello.fit_predict(X)
            df['Score_Anomalia'] = modello.decision_function(X) * -1 #(score alto = anomalia)

        # Estrazione anomalie per il report
        anomalie_trovate = df[df['Anomalia'] == -1]
        peggiori_anomalie = anomalie_trovate.sort_values(by='Score_Anomalia', ascending=False).head(3)

        dati_per_report = f"""
        AP: {target_ap} (Luogo: {luogo_ap})
        Modello utilizzato: {modello_scelto}
        Anomalie trovate: {len(anomalie_trovate)}
        Dettaglio top 3 record anomali:
        {peggiori_anomalie.to_string(index=False)}
        """

        # --- AGENTE 4 (XAI Manager) ---
        prompt_agente4 = """
        Sei un esperto XAI. Scrivi un report manageriale basato sui record anomali forniti.
        Giustifica brevemente al manager perché è stato usato quel modello ML.
        Usa il contesto del luogo dell'AP per fare ipotesi sensate. Includi un piano d'azione. Struttura in Markdown.
        """
        resp4 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt_agente4}, {"role": "user", "content": dati_per_report}]
        )
        report_finale = resp4.choices[0].message.content

    # --- STAMPA DEI RISULTATI SULLA GUI ---
    st.success(f" Pipeline completata! Modello attivato: **{modello_scelto}**")
    
    col_sx, col_dx = st.columns([1.2, 1])
    
    with col_sx:
        st.write("Rilevamento Grafico Anomalie")
        
        # Preparazione dati per il grafico 
        df_grafico = df.copy()
        df_grafico['Stato'] = df_grafico['Anomalia'].apply(lambda x: 'Criticità (Anomalia)' if x == -1 else 'Traffico Normale')
        
        # Grafico Altair
        grafico = alt.Chart(df_grafico).mark_circle(size=80).encode(
            x=alt.X('Utenti_Connessi', title='Utenti Connessi'),
            y=alt.Y('Traffico_IN', title='Traffico IN (Byte)'),
            color=alt.Color('Stato', scale=alt.Scale(
                domain=['Traffico Normale', 'Criticità (Anomalia)'],
                range=['#1f77b4', '#FF0000'] 
            )),
            tooltip=['Utenti_Connessi', 'Traffico_IN', 'Traffico_OUT', 'Score_Anomalia', 'Stato']
        ).interactive()
        
        st.altair_chart(grafico, use_container_width=True)

        st.metric(label="Anomalie Totali Rilevate", value=len(anomalie_trovate))
        st.write("**Top 3 record più critici:**")
        st.dataframe(peggiori_anomalie, use_container_width=True)

        with st.expander(" Visualizza Log degli Agenti (Debug)"):
            st.write(f"**Indice Autocorrelazione Calcolato:** {autocorr_val:.2f}")
            st.write("**Ragionamento Agente 2:**")
            st.info(strategia)
            st.write("**Istruzioni di Codice Generate (Agente 3 - JSON):**")
            st.json(istruzioni)
            
    with col_dx:
        st.write(" Report Manageriale XAI")
        st.markdown(report_finale)