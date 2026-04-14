import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="AI Threat Detection System",
    page_icon="shield",
    layout="wide"
)

st.title("AI Cybersecurity Threat Detection System")
st.markdown("**MBZUAI Application Project** — CICIDS 2017 Dataset")

@st.cache_resource
def load_models():
    rf = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    return rf, scaler, le

@st.cache_resource
def load_live_models():
    rf = joblib.load('rf_live_model.pkl')
    sc = joblib.load('scaler_live.pkl')
    le = joblib.load('le_live.pkl')
    return rf, sc, le

rf_model, scaler, le = load_models()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "Dashboard",
    "Threat Analysis",
    "Model Comparison",
    "Live Network Analysis"
])

if page == "Dashboard":
    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", "2,212,030")
    col2.metric("Attack Types", "9")
    col3.metric("RF Accuracy", "100%")
    col4.metric("LSTM Accuracy", "97%")

    attack_data = {
        'Attack': ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS',
                   'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'Bot'],
        'Count': [1672017, 230124, 158804, 128025, 10293, 5796, 5499, 1956]
    }
    df_attacks = pd.DataFrame(attack_data)
    fig = px.bar(df_attacks, x='Attack', y='Count',
                 color='Count', color_continuous_scale='Blues',
                 title='Attack Distribution - CICIDS 2017')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Threat Analysis":
    st.header("Upload and Analyze Network Traffic")

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        if 'Label' in df.columns:
            df = df.drop('Label', axis=1)
        if 'Label_encoded' in df.columns:
            df = df.drop('Label_encoded', axis=1)

        st.success(f"Loaded {len(df)} records")

        try:
            X_scaled = scaler.transform(df)
            predictions = rf_model.predict(X_scaled)
            pred_labels = le.inverse_transform(predictions)
            df['Prediction'] = pred_labels

            attack_mask = df['Prediction'] != 'BENIGN'
            st.error(f"{attack_mask.sum()} threats detected!")
            st.success(f"{(~attack_mask).sum()} benign connections")

            fig2 = px.pie(df, names='Prediction', title='Traffic Classification')
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(df[['Prediction']].head(50))

        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Model Comparison":
    st.header("Model Performance Comparison")

    results = {
        'Model': ['Random Forest', 'LSTM', 'Autoencoder'],
        'Accuracy': [1.00, 0.97, None],
        'Weighted F1': [1.00, 0.97, None],
        'Macro F1': [0.91, 0.77, None],
        'Detection Rate': [None, None, 0.45]
    }
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='Random Forest',
                          x=['Accuracy', 'Weighted F1', 'Macro F1'],
                          y=[1.00, 1.00, 0.91]))
    fig3.add_trace(go.Bar(name='LSTM',
                          x=['Accuracy', 'Weighted F1', 'Macro F1'],
                          y=[0.97, 0.97, 0.77]))
    fig3.update_layout(barmode='group', title='Model Comparison')
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Live Network Analysis":
    st.header("Live Network Traffic Analysis")
    st.info("Upload live traffic captured from your network using nfstream")

    rf_live, scaler_live, le_live = load_live_models()

    uploaded_live = st.file_uploader("Upload live_traffic.csv", type=['csv'])

    if uploaded_live:
        df_live = pd.read_csv(uploaded_live)
        st.success(f"Loaded {len(df_live)} live flows")

        features = [
            'bidirectional_duration_ms', 'bidirectional_packets',
            'bidirectional_bytes', 'src2dst_packets', 'src2dst_bytes',
            'dst2src_packets', 'dst2src_bytes', 'dst_port'
        ]

        try:
            X_live = df_live[features].fillna(0)
            X_scaled = scaler_live.transform(X_live)
            preds = rf_live.predict(X_scaled)
            pred_labels = le_live.inverse_transform(preds)
            df_live['Prediction'] = pred_labels

            attacks = df_live[df_live['Prediction'] != 'BENIGN']
            benign = df_live[df_live['Prediction'] == 'BENIGN']

            col1, col2 = st.columns(2)
            col1.error(f"Threats: {len(attacks)}")
            col2.success(f"Benign: {len(benign)}")

            fig4 = px.pie(df_live, names='Prediction',
                          title='Live Traffic Classification')
            st.plotly_chart(fig4, use_container_width=True)

            if len(attacks) > 0:
                st.subheader("Detected Threats")
                st.dataframe(attacks[['src_ip', 'dst_ip', 'dst_port', 'Prediction']].head(20))

        except Exception as e:
            st.error(f"Error: {e}")