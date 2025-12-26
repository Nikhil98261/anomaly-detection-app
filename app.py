import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import joblib
from utils import (
    load_model, 
    load_data, 
    preprocess_data, 
    predict_anomalies,
    visualize_anomalies,
    create_scatter_plot,
    create_distribution_plot,
    get_model_info,
    check_feature_compatibility,
    get_24_features
)

# Page configuration
st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .anomaly-count {
        color: #e74c3c;
        font-weight: bold;
    }
    .normal-count {
        color: #27ae60;
        font-weight: bold;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .feature-list {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'model_info' not in st.session_state:
    st.session_state.model_info = None

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Anomaly Detection System (24 Features)</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model section
        st.subheader("🤖 Model")
        model_option = st.radio(
            "Select Model Source",
            ["Load from joblib file", "Upload new model"],
            help="Load pre-trained model"
        )
        
        if model_option == "Upload new model":
            uploaded_model = st.file_uploader("Upload model file", type=['pkl', 'joblib'])
            if uploaded_model:
                try:
                    with st.spinner("Loading model..."):
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                            tmp_file.write(uploaded_model.getvalue())
                            tmp_path = tmp_file.name
                        
                        model = joblib.load(tmp_path)
                        st.session_state.model = model
                        st.session_state.model_info = get_model_info(model)
                        
                        st.markdown('<div class="success-box">✅ Model loaded successfully!</div>', unsafe_allow_html=True)
                        st.write(f"**Type:** {st.session_state.model_info['model_type']}")
                        if 'n_features' in st.session_state.model_info:
                            st.write(f"**Expected features:** {st.session_state.model_info['n_features']}")
                        
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            if st.button("📂 Load Pre-trained Model", use_container_width=True):
                with st.spinner("Loading model..."):
                    try:
                        model = load_model()
                        st.session_state.model = model
                        st.session_state.model_info = get_model_info(model)
                        
                        st.markdown('<div class="success-box">✅ Model loaded successfully!</div>', unsafe_allow_html=True)
                        st.write(f"**Type:** {st.session_state.model_info['model_type']}")
                        if 'n_features' in st.session_state.model_info:
                            st.write(f"**Expected features:** {st.session_state.model_info['n_features']}")
                        
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
        
        st.divider()
        
        # Data section
        st.subheader("📊 Data")
        data_option = st.radio(
            "Select Data Source",
            ["Use data.csv", "Upload new data"],
            help="Load dataset for analysis"
        )
        
        if data_option == "Upload new data":
            uploaded_data = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_data:
                try:
                    df = pd.read_csv(uploaded_data)
                    st.session_state.data = df
                    
                    # Auto-select 24 features
                    features = get_24_features(df)
                    st.session_state.features = features
                    
                    st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
                    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                    st.write(f"**Selected features:** {len(features)} features")
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        else:
            if st.button("📂 Load Dataset", use_container_width=True):
                with st.spinner("Loading data..."):
                    try:
                        df = load_data()
                        st.session_state.data = df
                        
                        # Auto-select 24 features
                        features = get_24_features(df)
                        st.session_state.features = features
                        
                        st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
                        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                        st.write(f"**Selected features:** {len(features)} features")
                        
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
        
        # Feature management
        if st.session_state.data is not None and st.session_state.features is not None:
            st.divider()
            st.subheader("🎯 Feature Management")
            
            if st.button("🔄 Auto-select 24 Features", use_container_width=True):
                features = get_24_features(st.session_state.data)
                st.session_state.features = features
                st.rerun()
            
            if st.button("📋 View All Features", use_container_width=True):
                df = st.session_state.data
                st.write(f"Total columns: {len(df.columns)}")
                st.write("All columns:")
                st.dataframe(pd.DataFrame({'Column Names': df.columns}), height=300, width=400)
        
        st.divider()
        
        # Detection settings
        st.subheader("⚡ Detection Settings")
        threshold = st.slider(
            "Anomaly Threshold (%)",
            min_value=1,
            max_value=20,
            value=5,
            help="Percentage of data points to flag as anomalies"
        )
        
        st.session_state.threshold = threshold / 100  # Convert to decimal
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Detection", "📈 Visualizations", "⚙️ Model Info"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Dashboard Overview</h2>', unsafe_allow_html=True)
        
        # Model status
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Model Status")
            if st.session_state.model is not None:
                st.markdown('<div class="success-box">✅ Model Loaded</div>', unsafe_allow_html=True)
                if st.session_state.model_info:
                    st.write(f"**Type:** {st.session_state.model_info['model_type']}")
            else:
                st.markdown('<div class="warning-box">⚠️ No Model Loaded</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Data Status")
            if st.session_state.data is not None:
                df = st.session_state.data
                st.markdown('<div class="success-box">✅ Data Loaded</div>', unsafe_allow_html=True)
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
            else:
                st.markdown('<div class="warning-box">⚠️ No Data Loaded</div>', unsafe_allow_html=True)
        
        # Feature information
        if st.session_state.features is not None:
            st.markdown("### Selected Features")
            features = st.session_state.features
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Total features selected:** {len(features)}")
                if len(features) <= 10:
                    st.write(features)
                else:
                    with st.expander(f"View all {len(features)} features"):
                        st.markdown('<div class="feature-list">', unsafe_allow_html=True)
                        for i, feat in enumerate(features, 1):
                            st.write(f"{i:2}. {feat}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if st.session_state.model is not None:
                    issues = check_feature_compatibility(st.session_state.model, features)
                    if issues:
                        st.markdown('<div class="warning-box">⚠️ Compatibility Issues</div>', unsafe_allow_html=True)
                        for issue in issues:
                            st.write(f"• {issue}")
                    else:
                        st.markdown('<div class="success-box">✅ Compatible</div>', unsafe_allow_html=True)
        
        # Data preview
        if st.session_state.data is not None:
            st.markdown("### Data Preview")
            df = st.session_state.data
            
            # Show first 10 rows
            st.dataframe(df.head(10), width=1200, height=300)
            
            # Statistics
            st.markdown("### Data Statistics")
            st.dataframe(df.describe(), width=1200, height=300)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Anomaly Detection</h2>', unsafe_allow_html=True)
        
        # Check prerequisites
        if st.session_state.model is None:
            st.warning("Please load a model from the sidebar first.")
        elif st.session_state.data is None:
            st.warning("Please load data from the sidebar first.")
        elif st.session_state.features is None or len(st.session_state.features) == 0:
            st.warning("No features selected. Please auto-select features from the sidebar.")
        else:
            # Show current configuration
            st.info(f"""
            **Current Configuration:**
            - Model: {st.session_state.model_info['model_type'] if st.session_state.model_info else 'Unknown'}
            - Features: {len(st.session_state.features)} selected
            - Threshold: {st.session_state.threshold*100}% of data flagged as anomalies
            """)
            
            # Run detection button
            if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
                with st.spinner("Running anomaly detection..."):
                    try:
                        # Get data and features
                        df = st.session_state.data
                        features = st.session_state.features
                        
                        # Preprocess data
                        X_scaled, scaler, features_used = preprocess_data(df, features)
                        st.session_state.features = features_used  # Update with actual features used
                        
                        # Make predictions
                        predictions = predict_anomalies(
                            st.session_state.model, 
                            X_scaled, 
                            st.session_state.threshold
                        )
                        st.session_state.predictions = predictions
                        
                        # Create results
                        results_df = df.copy()
                        results_df['is_anomaly'] = predictions
                        results_df['anomaly_score'] = X_scaled.mean(axis=1)
                        
                        # Display results
                        st.markdown("## 📊 Detection Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total = len(predictions)
                            st.metric("Total Samples", total)
                        with col2:
                            anomalies = int(predictions.sum())
                            st.metric("Anomalies Detected", anomalies)
                        with col3:
                            anomaly_rate = (anomalies / total) * 100
                            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
                        
                        # Visualization
                        fig = go.Figure()
                        fig.add_trace(go.Indicator(
                            mode="gauge+number",
                            value=anomaly_rate,
                            title={'text': "Anomaly Rate"},
                            gauge={'axis': {'range': [0, 100]},
                                  'bar': {'color': "red" if anomaly_rate > 10 else "orange" if anomaly_rate > 5 else "green"},
                                  'steps': [
                                      {'range': [0, 5], 'color': "lightgreen"},
                                      {'range': [5, 10], 'color': "lightyellow"},
                                      {'range': [10, 100], 'color': "lightcoral"}
                                  ]}
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show anomalies
                        st.markdown("### 🔴 Detected Anomalies")
                        anomaly_df = results_df[results_df['is_anomaly'] == True]
                        
                        if not anomaly_df.empty:
                            st.dataframe(anomaly_df.head(20), width=1200, height=400)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results (CSV)",
                                data=csv,
                                file_name="anomaly_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.success("✅ No anomalies detected!")
                        
                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                        st.exception(e)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is None or st.session_state.features is None:
            st.warning("Please load data and select features first.")
        else:
            df = st.session_state.data
            features = st.session_state.features
            
            # Feature distributions
            st.markdown("### 📊 Feature Distributions")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_feature = st.selectbox("Select feature:", features)
            
            if selected_feature in df.columns:
                # Histogram
                fig = px.histogram(
                    df, 
                    x=selected_feature,
                    title=f'Distribution of {selected_feature}',
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                fig = px.box(df, y=selected_feature, title=f'Box Plot of {selected_feature}')
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot if predictions exist
            if st.session_state.predictions is not None:
                st.markdown("### 🔍 Anomaly Visualization")
                
                predictions = st.session_state.predictions
                results_df = df.copy()
                results_df['anomaly'] = predictions
                
                col1, col2 = st.columns(2)
                with col1:
                    x_feat = st.selectbox("X-axis feature:", features, key='x_feat')
                with col2:
                    y_feat = st.selectbox("Y-axis feature:", features, key='y_feat')
                
                if x_feat != y_feat:
                    fig = px.scatter(
                        results_df,
                        x=x_feat,
                        y=y_feat,
                        color='anomaly',
                        title=f'{x_feat} vs {y_feat}',
                        color_discrete_map={True: 'red', False: 'blue'},
                        hover_data=features[:5]  # Show first 5 features in hover
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
        
        if st.session_state.model is None:
            st.warning("No model loaded. Please load a model from the sidebar.")
        else:
            model = st.session_state.model
            model_info = st.session_state.model_info
            
            # Basic info
            st.markdown("### 📋 Model Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model Type:** `{model_info['model_type']}`")
                if 'n_features' in model_info:
                    st.write(f"**Expected Features:** {model_info['n_features']}")
            
            with col2:
                if 'has_feature_importances' in model_info:
                    st.write(f"**Has Feature Importances:** ✅")
                if 'expected_features' in model_info:
                    st.write(f"**Feature Names Stored:** ✅")
            
            # Parameters
            st.markdown("### ⚙️ Model Parameters")
            if model_info['model_params']:
                params_df = pd.DataFrame(list(model_info['model_params'].items()), columns=['Parameter', 'Value'])
                st.dataframe(params_df, width=800, height=400)
            else:
                st.info("No parameters available.")
            
            # Feature importances (FIXED VERSION)
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 🏆 Feature Importances")
                
                if st.session_state.features is not None:
                    features = st.session_state.features
                    importances = model.feature_importances_
                    
                    # Handle different lengths
                    if len(features) == len(importances):
                        # Create importance dataframe
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot top 10
                        fig = px.bar(
                            importance_df.head(10),
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Most Important Features',
                            color='Importance',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show all importances
                        with st.expander("View All Feature Importances"):
                            st.dataframe(importance_df, width=800, height=400)
                    
                    elif len(features) > len(importances):
                        # More features than importances
                        st.warning(f"""
                        **Feature mismatch:**
                        - Model has {len(importances)} importance values
                        - Data has {len(features)} features
                        
                        Showing importance values without feature names:
                        """)
                        
                        # Create generic feature names
                        generic_features = [f'Feature_{i+1}' for i in range(len(importances))]
                        importance_df = pd.DataFrame({
                            'Feature': generic_features,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        st.dataframe(importance_df, width=800, height=400)
                    
                    else:
                        # More importances than features
                        st.warning(f"""
                        **Feature mismatch:**
                        - Model has {len(importances)} importance values
                        - Data has {len(features)} features
                        
                        This usually means the model was trained on different features.
                        """)
                else:
                    st.info("Run anomaly detection to see feature importances.")
            
            # Expected features from model
            if hasattr(model, 'feature_names_in_'):
                st.markdown("### 📝 Model's Expected Features")
                expected_features = list(model.feature_names_in_)
                
                st.write(f"**Total expected features:** {len(expected_features)}")
                
                if len(expected_features) <= 20:
                    st.write(expected_features)
                else:
                    with st.expander(f"View all {len(expected_features)} expected features"):
                        st.markdown('<div class="feature-list">', unsafe_allow_html=True)
                        for i, feat in enumerate(expected_features, 1):
                            st.write(f"{i:2}. {feat}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📋 How to Use This System
    
    1. **Load Model**: In sidebar, load your `joblib_model.pkl` file
    2. **Load Data**: Load your `data.csv` file (will auto-select 24 features)
    3. **Run Detection**: Go to Detection tab and click "Run Anomaly Detection"
    4. **Analyze Results**: View anomalies and visualizations
    5. **Export**: Download results as CSV if needed
    
    **Note**: The system auto-selects 24 features from your data. 
    If you need different features, use the "Auto-select 24 Features" button in the sidebar.
    """)

if __name__ == "__main__":
    main()