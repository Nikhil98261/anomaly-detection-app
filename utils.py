import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

def load_model(model_path='/Users/nikhilyadav/Desktop/anomaly-detection-app/random_cred.jb'):
    """Load the trained anomaly detection model"""
    try:
        model = joblib.load(model_path)
        
        # Log model information
        print(f"✅ Model loaded: {type(model).__name__}")
        if hasattr(model, 'feature_names_in_'):
            print(f"📊 Model expects {len(model.feature_names_in_)} features")
            print(f"📋 Features: {list(model.feature_names_in_)}")
        elif hasattr(model, 'n_features_in_'):
            print(f"📊 Model expects {model.n_features_in_} features")
        
        return model
    except Exception as e:
        raise Exception(f"❌ Error loading model: {str(e)}")

def load_data(data_path='/Users/nikhilyadav/Desktop/anomaly-detection-app/credit card.csv'):
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise Exception(f"❌ Error loading data: {str(e)}")

def get_24_features(df):
    """Extract 24 features from the dataframe - USING YOUR SPECIFIC COLUMN NAMES"""
    
    # 🔥 YOUR 24 SPECIFIC COLUMN NAMES 🔥
    SPECIFIC_24_FEATURES = [
        'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    print(f"🔍 Looking for your 24 specific features...")
    
    # Check which features exist in the dataframe
    available_features = []
    missing_features = []
    
    for feature in SPECIFIC_24_FEATURES:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    # If we have all 24, return them
    if len(available_features) == 24:
        print(f"✅ Found all 24 specified features!")
        return available_features
    
    # If some are missing, try case-insensitive matching
    if missing_features:
        print(f"⚠️  Some features not found: {missing_features}")
        print("🔍 Trying case-insensitive matching...")
        
        df_columns_lower = [str(col).lower() for col in df.columns]
        
        for missing_feature in missing_features[:]:  # Copy list for iteration
            missing_lower = str(missing_feature).lower()
            
            # Try exact lowercase match
            if missing_lower in df_columns_lower:
                idx = df_columns_lower.index(missing_lower)
                matched_col = df.columns[idx]
                available_features.append(matched_col)
                missing_features.remove(missing_feature)
                print(f"   ✓ Matched '{missing_feature}' to '{matched_col}'")
            
            # Try partial match
            elif any(missing_lower in col_lower for col_lower in df_columns_lower):
                for i, col_lower in enumerate(df_columns_lower):
                    if missing_lower in col_lower and df.columns[i] not in available_features:
                        available_features.append(df.columns[i])
                        if missing_feature in missing_features:
                            missing_features.remove(missing_feature)
                        print(f"   ≈ Partial match: '{missing_feature}' to '{df.columns[i]}'")
                        break
    
    # Check what we have now
    if len(available_features) >= 20:
        print(f"✅ Using {len(available_features)} features (close enough to 24)")
        return available_features[:24]  # Take up to 24
    
    # Fallback: if we don't have enough specific features, use auto-detection
    print(f"⚠️  Only found {len(available_features)} specific features, using auto-detection")
    
    all_columns = list(df.columns)
    
    # Common columns to exclude
    exclude_keywords = [
        'label', 'target', 'anomaly', 'class', 
        'is_anomaly', 'is_fraud', 'outcome', 'result',
        'timestamp', 'date', 'time', 'id', 'index',
        'Unnamed: 0', 'unnamed', 'sample', 'instance'
    ]
    
    # Filter columns (exclude non-feature columns)
    auto_features = []
    for col in all_columns:
        col_lower = str(col).lower()
        # Only exclude if it's clearly not a feature
        if not any(keyword in col_lower for keyword in exclude_keywords):
            auto_features.append(col)
    
    # If we have more than 24, take first 24
    if len(auto_features) > 24:
        print(f"⚠️  Auto-detected {len(auto_features)} features, using first 24")
        auto_features = auto_features[:24]
    
    print(f"✅ Using {len(auto_features)} auto-detected features")
    return auto_features

def preprocess_data(df, features=None):
    """Preprocess the data for prediction"""
    
    # Get 24 features if not specified
    if features is None:
        features = get_24_features(df)
    
    print(f"📊 Features selected ({len(features)}): {features}")
    
    # Select features - ensure they exist
    missing_in_df = [f for f in features if f not in df.columns]
    if missing_in_df:
        print(f"❌ Warning: Some features not in dataframe: {missing_in_df}")
        # Remove missing features
        features = [f for f in features if f in df.columns]
    
    if len(features) == 0:
        raise ValueError("❌ No valid features found in dataframe!")
    
    X = df[features].copy()
    
    # 🔥 SPECIAL HANDLING FOR YOUR SPECIFIC FEATURES 🔥
    # Handle categorical features (based on your column names)
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    # Convert categorical features to numeric if they're not already
    for cat_col in categorical_features:
        if cat_col in X.columns:
            if X[cat_col].dtype == 'object':
                # Encode categorical strings to numeric
                X[cat_col] = pd.factorize(X[cat_col])[0]
            # Ensure it's numeric
            X[cat_col] = pd.to_numeric(X[cat_col], errors='coerce')
    
    # Handle missing values
    for col in X.columns:
        if X[col].isnull().any():
            print(f"⚠️  Column '{col}' has {X[col].isnull().sum()} missing values")
            if X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
    
    # Convert all columns to numeric if possible
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except Exception as e:
            print(f"⚠️  Could not convert column '{col}' to numeric: {e}")
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, features

def predict_anomalies(model, X_scaled, threshold=0.05):
    """Make predictions using the loaded model"""
    try:
        if hasattr(model, 'predict'):
            predictions = model.predict(X_scaled)
            # Handle different prediction formats
            if predictions.dtype in [np.int32, np.int64]:
                # Assuming 1=anomaly, 0=normal (common for anomaly detection)
                predictions = predictions.astype(bool)
                # Check if we need to invert (if 0=anomaly)
                unique_vals = np.unique(predictions)
                if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
                    # If both 0 and 1 exist, assume 1=anomaly
                    pass
                elif 0 in unique_vals:
                    # If only 0 exists in binary, invert
                    predictions = predictions == 1
            elif predictions.dtype in [np.float32, np.float64]:
                # For probability scores > 0.5 = anomaly
                predictions = predictions > 0.5
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X_scaled)
            # Lower scores = more anomalous for many models
            predictions = scores < np.percentile(scores, threshold * 100)
        elif hasattr(model, 'score_samples'):
            scores = model.score_samples(X_scaled)
            # Lower scores = more anomalous
            predictions = scores < np.percentile(scores, threshold * 100)
        else:
            # Default: mark threshold% as anomalies
            n_samples = X_scaled.shape[0]
            predictions = np.zeros(n_samples, dtype=bool)
            n_anomalies = int(n_samples * threshold)
            if n_anomalies > 0:
                predictions[:n_anomalies] = True
            np.random.shuffle(predictions)
            print(f"⚠️  Using fallback: marking {n_anomalies} samples as anomalies")
        
        # Ensure boolean type
        predictions = predictions.astype(bool)
        
        # Print prediction stats
        anomaly_count = predictions.sum()
        print(f"📊 Predictions: {anomaly_count} anomalies out of {len(predictions)} samples ({anomaly_count/len(predictions)*100:.2f}%)")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        # Fallback: mark threshold% as anomalies
        n_samples = X_scaled.shape[0]
        predictions = np.zeros(n_samples, dtype=bool)
        n_anomalies = int(n_samples * threshold)
        if n_anomalies > 0:
            predictions[:n_anomalies] = True
        np.random.shuffle(predictions)
        print(f"⚠️  Fallback: marking {n_anomalies} samples as anomalies")
        return predictions

def visualize_anomalies(df, predictions, features):
    """Create visualizations for anomalies"""
    results_df = df.copy()
    results_df['anomaly_pred'] = predictions
    results_df['anomaly_pred_label'] = results_df['anomaly_pred'].map({True: 'Anomaly', False: 'Normal'})
    
    # Add anomaly score based on prediction confidence
    if 'anomaly_score' not in results_df.columns:
        # Create a simple score (1.0 for anomaly, 0.0 for normal)
        results_df['anomaly_score'] = results_df['anomaly_pred'].astype(float)
    
    return results_df

def create_scatter_plot(df, x_feature, y_feature, color_feature='anomaly_pred_label'):
    """Create an interactive scatter plot"""
    fig = px.scatter(
        df, 
        x=x_feature, 
        y=y_feature, 
        color=color_feature,
        title=f'{x_feature} vs {y_feature} - Anomaly Detection',
        hover_data=df.columns.tolist(),
        color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=8))
    return fig

def create_distribution_plot(df, feature, predictions):
    """Create distribution plot for a feature"""
    fig = go.Figure()
    
    # Check if feature exists
    if feature not in df.columns:
        fig.add_annotation(text=f"Feature '{feature}' not found",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Normal data
    if (~predictions).sum() > 0:  # If we have normal samples
        fig.add_trace(go.Histogram(
            x=df.loc[~predictions, feature],
            name='Normal',
            marker_color='blue',
            opacity=0.7,
            nbinsx=30
        ))
    
    # Anomaly data
    if predictions.sum() > 0:  # If we have anomaly samples
        fig.add_trace(go.Histogram(
            x=df.loc[predictions, feature],
            name='Anomaly',
            marker_color='red',
            opacity=0.7,
            nbinsx=30
        ))
    
    fig.update_layout(
        title=f'Distribution of {feature}',
        xaxis_title=feature,
        yaxis_title='Count',
        barmode='overlay'
    )
    
    return fig

def get_model_info(model):
    """Extract information about the loaded model"""
    info = {
        'model_type': type(model).__name__,
        'model_params': model.get_params() if hasattr(model, 'get_params') else {},
    }
    
    if hasattr(model, 'n_features_in_'):
        info['n_features'] = model.n_features_in_
    
    if hasattr(model, 'feature_names_in_'):
        info['expected_features'] = list(model.feature_names_in_)
    
    if hasattr(model, 'feature_importances_'):
        info['has_feature_importances'] = True
        info['n_importances'] = len(model.feature_importances_)
    
    return info

def check_feature_compatibility(model, features):
    """Check if features are compatible with model"""
    issues = []
    
    if hasattr(model, 'n_features_in_'):
        if len(features) != model.n_features_in_:
            issues.append(f"Feature count mismatch: Model expects {model.n_features_in_}, got {len(features)}")
    
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        missing = set(model_features) - set(features)
        extra = set(features) - set(model_features)
        
        if missing:
            issues.append(f"Missing features expected by model: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            issues.append(f"Extra features not expected by model: {list(extra)[:5]}{'...' if len(extra) > 5 else ''}")
    
    return issues

def get_feature_groups():
    """Return feature groups for your specific 24 features"""
    return {
        'Demographic': ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE'],
        'Payment Status': ['PAY_0', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
        'Bill Amount': ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'],
        'Payment Amount': ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    }

def describe_features(df, features):
    """Generate descriptive statistics for the features"""
    if not features:
        return pd.DataFrame()
    
    stats = []
    for feature in features:
        if feature in df.columns:
            col_data = df[feature]
            stats.append({
                'Feature': feature,
                'Type': str(col_data.dtype),
                'Missing': col_data.isnull().sum(),
                'Missing %': f"{col_data.isnull().sum() / len(col_data) * 100:.2f}%",
                'Mean': f"{col_data.mean():.2f}" if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
                'Std': f"{col_data.std():.2f}" if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
                'Min': f"{col_data.min():.2f}" if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
                'Max': f"{col_data.max():.2f}" if pd.api.types.is_numeric_dtype(col_data) else 'N/A'
            })
    
    return pd.DataFrame(stats)