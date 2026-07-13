import os
import io
import json
import yaml
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Set up page configurations
st.set_page_config(
    page_title="FinShield - Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    /* Dark theme customizations */
    .reportview-container {
        background: #0f111a;
    }
    .sidebar .sidebar-content {
        background: #141724;
    }
    
    /* Header card design */
    .header-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .header-card h1 {
        margin: 0;
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .header-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.85;
        font-size: 1.1rem;
    }
    
    /* KPI Card styling */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    .kpi-card {
        flex: 1;
        background: #191c2b;
        border-left: 5px solid #2a5298;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
    }
    .kpi-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #8c92ac;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .kpi-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    .badge-fraud { background: rgba(239, 83, 80, 0.2); color: #ef5350; }
    .badge-ok { background: rgba(76, 175, 80, 0.2); color: #4caf50; }
    .badge-info { background: rgba(33, 150, 243, 0.2); color: #2196f3; }
</style>
""", unsafe_allow_html=True)

# Helper function to load configs
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Main Navigation Sidebar
st.sidebar.image("https://img.icons8.com/color/120/shield-with-crown.png", width=60)
st.sidebar.markdown("<h2 style='color: white;'>FinShield Platform</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "Dashboard Overview",
        "Dataset & Statistics",
        "Exploratory Data Analysis",
        "Real-time Predictor",
        "Batch Fraud Predictor",
        "Explainable AI (SHAP)",
        "Model Performance",
        "REST API Tester"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("🤖 **Antigravity AI Fraud System**\nVersion 1.0.0")

# --- NAVIGATION ROUTING ---

if page == "Dashboard Overview":
    st.markdown("""
    <div class="header-card">
        <h1>🛡️ Financial Fraud Detection Dashboard</h1>
        <p>Enterprise real-time transaction monitoring, predictive machine learning, and explainable AI insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    api_online = False
    try:
        r = requests.get(f"{API_URL}/")
        if r.status_code == 200:
            api_online = True
    except:
        pass
        
    # KPI metrics section
    st.markdown("### System Key Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    # 1. API Status
    with col1:
        status_text = "ONLINE" if api_online else "OFFLINE"
        badge_cls = "badge-ok" if api_online else "badge-fraud"
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {'#4caf50' if api_online else '#ef5350'};">
            <div class="kpi-title">Inference Server</div>
            <div class="kpi-value">{status_text}</div>
            <div class="kpi-badge {badge_cls}">FastAPI Backend</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 2. Classifier type
    with col2:
        model_name = "Not Loaded"
        if api_online:
            try:
                model_name = r.json().get("model_name", "Not Loaded")
            except:
                pass
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #2196f3;">
            <div class="kpi-title">Active AI Model</div>
            <div class="kpi-value" style="font-size: 1.4rem; padding-top: 0.5rem; padding-bottom: 0.3rem;">{model_name}</div>
            <div class="kpi-badge badge-info">XGBoost Optimized</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 3. Best Threshold
    with col3:
        threshold = 0.5
        if api_online:
            try:
                threshold = r.json().get("decision_threshold", 0.5)
            except:
                pass
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #ff9800;">
            <div class="kpi-title">Decision Threshold</div>
            <div class="kpi-value">{threshold:.4f}</div>
            <div class="kpi-badge badge-info">Optimized F1</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 4. Total processed dataset
    with col4:
        dataset_status = "Available" if os.path.exists("data/raw/paysim.csv") else "Missing"
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #9c27b0;">
            <div class="kpi-title">Transaction Dataset</div>
            <div class="kpi-value">{dataset_status}</div>
            <div class="kpi-badge badge-info">PaySim synthetic data</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Main workflow instructions
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **Train the Model**: In your local environment, make sure to execute the model pipeline. Run the baseline training and tuning scripts.
    2. **Start FastAPI**: Initialize the server running `uvicorn api.main:app --reload`.
    3. **Try Real-time Predictor**: Use the navigation sidebar to fill out a transaction form and get instant AI fraud assessments.
    4. **Upload a Batch**: Have a transaction journal file? Upload it to the *Batch Fraud Predictor* page to flag multiple records.
    5. **Analyze Explainability**: Review *Explainable AI (SHAP)* to understand the decision boundaries and feature relationships.
    """)

elif page == "Dataset & Statistics":
    st.markdown("## 📊 Dataset & Statistics")
    
    # Locate dataset
    data_path = "data/raw/paysim.csv"
    if not os.path.exists(data_path):
        st.warning("PaySim CSV file was not found under `data/raw/`. Please make sure the data loading phase has been executed.")
    else:
        # Load a sample to describe
        @st.cache_data
        def load_sample():
            df = pd.read_csv(data_path, nrows=5000)
            return df
            
        df_sample = load_sample()
        
        st.markdown("### Raw Data Sample (First 5 Rows)")
        st.dataframe(df_sample.head())
        
        # Display schema explanation
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Schema Metadata")
            schema_info = {
                "step": "Maps a unit of time in the real world (1 step = 1 hour). Total simulation: 744 steps (30 days).",
                "type": "Transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER.",
                "amount": "Value of the transaction in local currency.",
                "nameOrig": "Unique ID of customer initiating the transaction.",
                "oldbalanceOrg": "Initial balance of origin account before transaction.",
                "newbalanceOrig": "New balance of origin account after transaction.",
                "nameDest": "Unique ID of destination account (recipient).",
                "oldbalanceDest": "Recipient balance before transaction.",
                "newbalanceDest": "Recipient balance after transaction.",
                "isFraud": "Flag indicating if transaction was fraudulent (Target column).",
                "isFlaggedFraud": "Rule-based system flag (Flags transfers exceeding 200,000 in a single step)."
            }
            for col, desc in schema_info.items():
                st.markdown(f"**`{col}`**: {desc}")
                
        with col2:
            st.markdown("### Class Imbalance Summary")
            # Create a mock/sample pie chart since dataset is huge, but we know the exact figures from PaySim
            # Or count target in our small sample (or hardcode standard values)
            labels = ['Legitimate', 'Fraudulent']
            values = [6354407, 8213] # Exact PaySim numbers
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker_colors=['#4caf50', '#ef5350'])])
            fig.update_layout(title_text="PaySim Full Target Distribution (0.129% Fraud)")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Exploratory Data Analysis":
    st.markdown("## 🔍 Exploratory Data Analysis (EDA)")
    
    data_path = "data/raw/paysim.csv"
    if not os.path.exists(data_path):
        st.warning("PaySim CSV file was not found under `data/raw/`.")
    else:
        @st.cache_data
        def load_eda_data():
            # Loading 100k rows (small subset containing both fraud and non-fraud) for speedy visualization
            df = pd.read_csv(data_path, nrows=100000)
            return df
            
        df = load_eda_data()
        
        # 1. Plotly Boxplot: Transaction Amount by Type
        st.markdown("### Transaction Amount Distribution by Type")
        fig1 = px.box(df, x="type", y="amount", color="type", log_y=True,
                     title="Transaction Amount Distribution (Log Scale)",
                     color_discrete_sequence=px.colors.qualitative.Dark2)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. Bar Chart: Fraud Cases by Transaction Type
        st.markdown("### Fraud Counts by Transaction Type")
        # In PaySim, fraud only occurs in TRANSFER and CASH_OUT. Let's showcase this
        fraud_df = df[df["isFraud"] == 1]
        if len(fraud_df) == 0:
            # Inject some synthetic fraud counts if our first 100k rows didn't capture them (first 100k has 128 frauds)
            fraud_counts = pd.DataFrame({
                "type": ["TRANSFER", "CASH_OUT"],
                "count": [4097, 4116] # Exact PaySim splits
            })
        else:
            fraud_counts = fraud_df.groupby("type")["isFraud"].count().reset_index(name="count")
            
        fig2 = px.bar(fraud_counts, x="type", y="count", color="type",
                      title="Fraudulent Transactions by Type (PaySim Dataset)",
                      color_discrete_sequence=['#ef5350', '#ff9800'])
        st.plotly_chart(fig2, use_container_width=True)
        
        # 3. Numeric Correlation Heatmap
        st.markdown("### Numeric Features Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig3 = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r",
                         title="Correlation Coefficient Matrix")
        st.plotly_chart(fig3, use_container_width=True)

elif page == "Real-time Predictor":
    st.markdown("## 🛡️ Real-time Transaction Evaluator")
    
    # Form input fields
    st.markdown("### Transaction Input Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        step = st.number_input("Step (Simulation Hour, 1-744)", min_value=1, max_value=744, value=1)
        txt_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=1e9, value=150.0)
    with col2:
        name_orig = st.text_input("Origin Account ID", "C130548614")
        old_balance_org = st.number_input("Origin Initial Balance ($)", min_value=0.0, max_value=1e9, value=1000.0)
        new_balance_org = st.number_input("Origin Final Balance ($)", min_value=0.0, max_value=1e9, value=850.0)
    with col3:
        name_dest = st.text_input("Destination Account ID", "C553264065")
        old_balance_dest = st.number_input("Destination Initial Balance ($)", min_value=0.0, max_value=1e9, value=0.0)
        new_balance_dest = st.number_input("Destination Final Balance ($)", min_value=0.0, max_value=1e9, value=150.0)
        
    if st.button("🚀 Evaluate Transaction", type="primary"):
        # Make API Request
        payload = {
            "step": step,
            "type": txt_type,
            "amount": amount,
            "nameOrig": name_orig,
            "oldbalanceOrg": old_balance_org,
            "newbalanceOrig": new_balance_org,
            "nameDest": name_dest,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest
        }
        
        with st.spinner("Analyzing transaction metrics..."):
            try:
                # 1. Predict Request
                r_pred = requests.post(f"{API_URL}/predict", json=payload)
                
                # 2. SHAP Request
                r_shap = requests.post(f"{API_URL}/shap", json=payload)
                
                if r_pred.status_code == 200:
                    data = r_pred.json()
                    prob = data["probability"]
                    is_fraud = data["is_fraud"]
                    threshold = data["decision_threshold"]
                    
                    st.markdown("---")
                    st.markdown("### AI Inference Result")
                    
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        # 3. Render dial / gauge of fraud risk
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Risk Score (%)", 'font': {'size': 18}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': "#ef5350" if is_fraud else "#4caf50"},
                                'bgcolor': "#1a1d29",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, threshold*100], 'color': 'rgba(76, 175, 80, 0.1)'},
                                    {'range': [threshold*100, 100], 'color': 'rgba(239, 83, 80, 0.1)'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': threshold * 100
                                }
                            }
                        ))
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"})
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with res_col2:
                        if is_fraud:
                            st.error(f"⚠️ **FRAUDULENT TRANSACTION DETECTED**\n\nProbability: **{prob*100:.2f}%** (Exceeds optimized decision threshold of {threshold:.3f})")
                        else:
                            st.success(f"✅ **LEGITIMATE TRANSACTION VERIFIED**\n\nProbability: **{prob*100:.2f}%** (Below optimized decision threshold of {threshold:.3f})")
                            
                        # Show transaction summary
                        st.markdown(f"""
                        * **Source**: Account `{name_orig}` (Old Bal: ${old_balance_org:,.2f} -> New Bal: ${new_balance_org:,.2f})
                        * **Destination**: Account `{name_dest}` (Old Bal: ${old_balance_dest:,.2f} -> New Bal: ${new_balance_dest:,.2f})
                        * **Amount**: ${amount:,.2f}
                        * **Channel**: {txt_type}
                        """)
                        
                    # Render Local SHAP Importance if SHAP request succeeded
                    if r_shap.status_code == 200:
                        st.markdown("### 🧠 Local Explainable AI (SHAP feature impact)")
                        shap_data = r_shap.json()
                        impacts_df = pd.DataFrame(shap_data["impacts"])
                        
                        # Plot SHAP impacts using horizontal bar chart
                        # We only display the top 10 most impactful features for readability
                        top_impacts = impacts_df.head(10).copy()
                        # Sort for bar chart (ascending so highest is at top)
                        top_impacts = top_impacts.sort_values(by="shap_value", key=abs, ascending=True)
                        
                        colors = ['#ef5350' if x >= 0 else '#4caf50' for x in top_impacts['shap_value']]
                        
                        fig_shap = go.Figure(go.Bar(
                            x=top_impacts['shap_value'],
                            y=top_impacts['feature'],
                            orientation='h',
                            marker_color=colors
                        ))
                        fig_shap.update_layout(
                            title="Feature Impact Contribution (Red: Increases Fraud Risk, Green: Reduces Risk)",
                            xaxis_title="SHAP Value Impact",
                            yaxis_title="Feature",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "white"},
                            margin=dict(l=150, r=20, t=40, b=40)
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.error(f"Inference failed with status {r_pred.status_code}: {r_pred.text}")
            except Exception as e:
                st.error(f"Connection error: Could not connect to API server at {API_URL}. Is it running? Details: {e}")

elif page == "Batch Fraud Predictor":
    st.markdown("## 📥 Batch Fraud Predictor")
    st.markdown("Upload a CSV file containing transactions mapping the raw PaySim schema to perform high-throughput predictions.")
    
    # Template download
    template = pd.DataFrame([{
        "step": 1, "type": "TRANSFER", "amount": 181.0, 
        "nameOrig": "C130548614", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
        "nameDest": "C553264065", "oldbalanceDest": 0.0, "newbalanceDest": 0.0
    }])
    
    st.download_button(
        "Download CSV Schema Template",
        data=template.to_csv(index=False),
        file_name="paysim_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"])
    
    if uploaded_file is not None:
        # Load file preview
        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded file summary:** {len(df_upload)} records.")
        st.dataframe(df_upload.head(10))
        
        if st.button("🚀 Run Batch Inference", type="primary"):
            # Prepare file buffer
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.read(), "text/csv")}
            
            with st.spinner("Sending file to FastAPI batch processor..."):
                try:
                    r = requests.post(f"{API_URL}/batch-predict", files=files)
                    if r.status_code == 200:
                        res = r.json()
                        total_records = res["total_records"]
                        total_fraud = res["total_fraud"]
                        predictions = res["predictions"]
                        
                        st.markdown("---")
                        st.markdown("### Inference Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Processed", total_records)
                        with col2:
                            st.metric("Frauds Identified", total_fraud, delta=f"{total_fraud/total_records*100:.3f}% Rate", delta_color="inverse")
                            
                        # Merge predictions back to upload dataframe
                        preds_df = pd.DataFrame(predictions)
                        final_df = df_upload.copy()
                        final_df["is_fraud"] = preds_df["is_fraud"]
                        final_df["fraud_probability"] = preds_df["probability"]
                        
                        # Show only frauds
                        st.markdown("#### Flagged Transactions Details")
                        frauds_only = final_df[final_df["is_fraud"] == True]
                        if len(frauds_only) > 0:
                            st.dataframe(frauds_only)
                        else:
                            st.success("No fraudulent transactions detected in this file.")
                            
                        # Download button for outputs
                        st.download_button(
                            "📥 Download Full Predictions CSV",
                            data=final_df.to_csv(index=False),
                            file_name="paysim_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Batch prediction failed: {r.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

elif page == "Explainable AI (SHAP)":
    st.markdown("## 🧠 Explainable AI (SHAP)")
    st.markdown("Explainable AI metrics validate model trustworthiness. Here, we present global feature importances extracted from the training set.")
    
    # Check if SHAP plots exist on disk
    bar_plot_path = "artifacts/evaluation/shap_summary_bar.png"
    beeswarm_plot_path = "artifacts/evaluation/shap_summary_beeswarm.png"
    
    if os.path.exists(bar_plot_path) and os.path.exists(beeswarm_plot_path):
        st.markdown("### Global Feature Importance (SHAP Bar Plot)")
        st.image(bar_plot_path, caption="Average impact magnitude of features on model outputs.")
        
        st.markdown("### Feature Value Impact Distribution (SHAP Beeswarm Plot)")
        st.image(beeswarm_plot_path, caption="Impact distribution of high/low feature values on fraud classification. Red values represent high feature ranges, while Blue values represent low ranges.")
    else:
        st.warning("SHAP global visualization images were not found in `artifacts/evaluation/`. Please execute the evaluator/explainability modules during training to generate these assets.")

elif page == "Model Performance":
    st.markdown("## 📈 Model Performance Registry")
    
    # Load model comparison
    comparison_path = "artifacts/evaluation/model_comparison.csv"
    metrics_path = "artifacts/evaluation/evaluation_metrics.json"
    
    if os.path.exists(comparison_path):
        st.markdown("### Baseline Algorithms Comparison")
        comp_df = pd.read_csv(comparison_path)
        st.table(comp_df)
        
        # Plotly chart comparing F1 & PR-AUC
        fig = px.bar(comp_df, x="Model", y=["F1-Score", "PR-AUC"], barmode="group",
                     title="Model Candidates Performance Comparison",
                     color_discrete_sequence=['#2196f3', '#00bcd4'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Baseline comparison details are not available yet. Once the baseline trainer runs, they will display here.")
        
    st.markdown("---")
    st.markdown("### Tuned Classifier Curves")
    
    col1, col2 = st.columns(2)
    with col1:
        roc_path = "artifacts/evaluation/roc_curve.png"
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC Curve showing true positive vs false positive trade-off.")
        else:
            st.warning("ROC Curve image not found.")
            
    with col2:
        pr_path = "artifacts/evaluation/precision_recall_curve.png"
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Precision-Recall Curve illustrating performance on imbalanced target.")
        else:
            st.warning("PR Curve image not found.")

elif page == "REST API Tester":
    st.markdown("## 🔌 REST API Tester")
    st.markdown("Perform manual endpoints requests to test the loaded FastAPI endpoints live.")
    
    endpoint = st.selectbox("API Route", ["/", "/health", "/model-info", "/feature-importance"])
    
    if st.button("Send GET Request"):
        with st.spinner("Fetching response..."):
            try:
                r = requests.get(f"{API_URL}{endpoint}")
                st.markdown("#### Status Code")
                st.code(r.status_code)
                st.markdown("#### JSON Response Payload")
                st.json(r.json())
            except Exception as e:
                st.error(f"Request failed: {e}")
