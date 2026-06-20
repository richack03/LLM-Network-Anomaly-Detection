import os
import json
import scipy.io as sio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# --- CONFIGURAZIONE INIZIALE BACKEND ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def carica_dati_raw():
    """Carica il dataset dal disco."""
    return sio.loadmat('APs_dataset.mat')

def elabora_pipeline_completa(target_ap, dati_mat):
    """
    Funzione core del backend: prende in input il nodo,
    fa i calcoli, interroga l'LLM e restituisce i risultati.
    """
    # --- 1. ESTRAZIONE DATI E CALCOLO STATISTICO (ROUTING) ---
    dati_ap = dati_mat[target_ap][0,0]
    luogo_ap = dati_ap['aploc'][0]
    utenti = dati_ap['numb_users'].flatten()
    rx = dati_ap['rxbytes'].flatten()
    tx = dati_ap['txbytes'].flatten()

    df = pd.DataFrame({'Utenti_Connessi': utenti, 'Traffico_IN': rx, 'Traffico_OUT': tx})
    
    autocorr_val = df['Traffico_OUT'].autocorr(lag=1)
    if pd.isna(autocorr_val):
        autocorr_val = 0.0

    stats = f"""
    - AP: {target_ap} (Luogo: {luogo_ap})
    - Record totali: {len(df)}
    - Media utenti: {df['Utenti_Connessi'].mean():.2f} (Max: {df['Utenti_Connessi'].max()})
    - Traffico IN Max: {df['Traffico_IN'].max()} Byte
    - Indice di Autocorrelazione (Regolarità): {autocorr_val:.2f} 
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
    REGOLA RIGIDA: Se l'Indice di Autocorrelazione è >= 0.6, devi suggerire 'XGBoost'. 
    Se l'Indice è < 0.6, devi suggerire 'IsolationForest'. Giustifica la tua scelta.
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
    Nella chiave 'model_type' scrivi ESATTAMENTE o "IsolationForest" o "XGBoost".
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
        df_xgb = df.copy()
        df_xgb['Traffico_OUT_lag'] = df_xgb['Traffico_OUT'].shift(1)
        df_xgb = df_xgb.dropna() 
        
        X_xgb = df_xgb[['Traffico_OUT_lag']]
        y_xgb = df_xgb['Traffico_OUT']
        
        modello = xgb.XGBRegressor(n_estimators=50, random_state=42)
        modello.fit(X_xgb, y_xgb)
        previsioni = modello.predict(X_xgb)
        
        errori = np.abs(y_xgb - previsioni)
        soglia_errore = np.percentile(errori, 99) 
        
        df['Anomalia'] = 1  
        df['Score_Anomalia'] = 0.0
        indici_anomali = df_xgb[errori > soglia_errore].index
        df.loc[indici_anomali, 'Anomalia'] = -1
        df.loc[df_xgb.index, 'Score_Anomalia'] = errori
        
    else:
        X = df[features]
        modello = IsolationForest(contamination=0.01, random_state=42)
        df['Anomalia'] = modello.fit_predict(X)
        df['Score_Anomalia'] = modello.decision_function(X) * -1 

    # Estrazione anomalie
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
    Giustifica al manager perché è stato usato quel modello ML.
    Usa il contesto del luogo dell'AP per fare ipotesi sensate. Includi un piano d'azione in Markdown.
    """
    resp4 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt_agente4}, {"role": "user", "content": dati_per_report}]
    )
    report_finale = resp4.choices[0].message.content

    # Il backend restituisce tutti i risultati in un unico pacchetto pulito
    return {
        "df": df,
        "modello_scelto": modello_scelto,
        "anomalie_trovate": anomalie_trovate,
        "peggiori_anomalie": peggiori_anomalie,
        "autocorr_val": autocorr_val,
        "strategia": strategia,
        "istruzioni": istruzioni,
        "report_finale": report_finale
    }
