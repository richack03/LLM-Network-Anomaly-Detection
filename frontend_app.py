import streamlit as st
import altair as alt
# IMPORTIAMO LA LOGICA DAL NOSTRO BACKEND SEPARATO
from backend_app import carica_dati_raw, elabora_pipeline_completa

# --- CONFIGURAZIONE INTERFACCIA ---
st.set_page_config(page_title="Network Anomaly Detector", page_icon="network-access-point-vector-icon-260nw-2461269453.jpg", layout="wide")

# La cache rimane qui sul frontend per non far faticare la pagina web
@st.cache_data
def get_dataset():
    return carica_dati_raw()

try:
    dati_mat = get_dataset()
except Exception as e:
    st.error(f"Impossibile caricare il file 'APs_dataset.mat'. Errore: {e}")
    st.stop()

lista_ap = [chiave for chiave in dati_mat.keys() if not chiave.startswith('__')]
lista_ap.sort()

# --- COSTRUZIONE DASHBOARD ---
st.title("Sistema Multi-Agente & ML per Anomaly Detection")
st.markdown("---")

col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    default_index = lista_ap.index("ap135172") if "ap135172" in lista_ap else 0
    target_ap = st.selectbox("Seleziona l'ID dell'Access Point da verificare:", options=lista_ap, index=default_index)

with col_input2:
    st.write("#") 
    avvia_pipeline = st.button(" Avvia Pipeline Multi-Agente", use_container_width=True)

# --- AZIONE AL CLICK ---
if avvia_pipeline:
    with st.spinner(f"Avvio pipeline per {target_ap}... Elaborazione in backend in corso..."):
        
        # CHIAMIAMO IL BACKEND (Una sola riga di codice pulitissima!)
        risultati = elabora_pipeline_completa(target_ap, dati_mat)
        
        # --- STAMPA DEI RISULTATI SULLA GUI ---
        st.success(f" Pipeline completata! Modello attivato dal backend: **{risultati['modello_scelto']}**")
        
        col_sx, col_dx = st.columns([1.2, 1])
        
        with col_sx:
            st.write("### Rilevamento Grafico Anomalie")
            
            df_grafico = risultati['df'].copy()
            df_grafico['Stato'] = df_grafico['Anomalia'].apply(lambda x: 'Criticità (Anomalia)' if x == -1 else 'Traffico Normale')
            
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

            st.metric(label="Anomalie Totali Rilevate", value=len(risultati['anomalie_trovate']))
            st.write("**Top 3 record più critici:**")
            st.dataframe(risultati['peggiori_anomalie'], use_container_width=True)

            with st.expander(" Visualizza Log degli Agenti (Debug Backend)"):
                st.write(f"**Indice Autocorrelazione Calcolato:** {risultati['autocorr_val']:.2f}")
                st.write("**Ragionamento Agente 2:**")
                st.info(risultati['strategia'])
                st.write("**Istruzioni di Codice Generate (Agente 3 - JSON):**")
                st.json(risultati['istruzioni'])
                
        with col_dx:
            st.write("### Report Manageriale XAI")
            st.markdown(risultati['report_finale'])
