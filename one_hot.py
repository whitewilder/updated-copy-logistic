import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import io
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Logistic Regression Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Title and description
st.title("🔍 Advanced Logistic Regression Dashboard")
st.markdown("---")

# Add tabs for Train Model and Load Model
main_tab1, main_tab2 = st.tabs(["🏗️ Build New Model", "📂 Load Existing Model"])

with main_tab1:
    # Sidebar for data upload and configuration
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # Training data upload
        train_file = st.file_uploader("Upload Training Data (Parquet)", type=['parquet'], key="train_upload")
        if train_file is not None:
            st.session_state.train_data = pd.read_parquet(train_file)
            st.success(f"Training data loaded: {st.session_state.train_data.shape}")
        
        # Testing data upload
        test_file = st.file_uploader("Upload Testing Data (Parquet)", type=['parquet'], key="test_upload")
        if test_file is not None:
            st.session_state.test_data = pd.read_parquet(test_file)
            st.success(f"Testing data loaded: {st.session_state.test_data.shape}")
        
        st.markdown("---")
        
        # Variable selection
        if st.session_state.train_data is not None:
            st.header("🎯 Variable Selection")
            
            # Target variable selection
            all_columns = st.session_state.train_data.columns.tolist()
            st.session_state.target_variable = st.selectbox(
                "Select Target Variable",
                options=all_columns,
                index=0 if all_columns else None
            )
            
            # Feature selection
            if st.session_state.target_variable:
                available_features = [col for col in all_columns if col != st.session_state.target_variable]
                st.session_state.selected_features = st.multiselect(
                    "Select Features",
                    options=available_features,
                    default=available_features[:5] if len(available_features) >= 5 else available_features
                )
            
            st.markdown("---")
            
            # Model parameters
            st.header("⚙️ Model Parameters")
            regularization = st.selectbox("Regularization", ["l2", "l1", "elasticnet"])
            C_value = st.slider("C (Inverse regularization strength)", 0.01, 10.0, 1.0)
            max_iter = st.slider("Max Iterations", 100, 1000, 100)
            
            # Fit model button
            if st.button("🚀 Fit Model", type="primary"):
                if len(st.session_state.selected_features) > 0:
                    with st.spinner("Training model..."):
                        # Prepare data
                        X_train = st.session_state.train_data[st.session_state.selected_features]
                        y_train = st.session_state.train_data[st.session_state.target_variable]
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        
                        # Train model
                        st.session_state.model = LogisticRegression(
                            penalty=regularization,
                            C=C_value,
                            max_iter=max_iter,
                            random_state=42,
                            solver='saga' if regularization == 'elasticnet' else 'lbfgs'
                        )
                        
                        if regularization == 'elasticnet':
                            st.session_state.model.l1_ratio = 0.5
                        
                        st.session_state.model.fit(X_train_scaled, y_train)
                        st.session_state.scaler = scaler
                        st.session_state.model_trained = True
                        st.success("Model trained successfully!")
                else:
                    st.error("Please select at least one feature!")

with main_tab2:
    st.header("📂 Load Saved Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload saved model
        model_file = st.file_uploader("Upload Saved Model (.pkl)", type=['pkl'])
        
        if model_file is not None:
            try:
                # Load model data
                model_data = joblib.load(model_file)
                
                # Extract components
                st.session_state.model = model_data['model']
                st.session_state.scaler = model_data['scaler']
                st.session_state.selected_features = model_data['features']
                st.session_state.target_variable = model_data['target']
                st.session_state.model_trained = True
                
                st.success("Model loaded successfully!")
                
                # Display model information
                st.subheader("Loaded Model Information")
                st.write(f"**Target Variable:** {st.session_state.target_variable}")
                st.write(f"**Number of Features:** {len(st.session_state.selected_features)}")
                st.write(f"**Features:** {', '.join(st.session_state.selected_features)}")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    with col2:
        if st.session_state.model_trained and st.session_state.model is not None:
            st.subheader("Quick Model Stats")
            st.metric("Intercept", f"{st.session_state.model.intercept_[0]:.4f}")
            st.metric("Regularization", st.session_state.model.penalty)
            st.metric("C Value", st.session_state.model.C)
    
    # Upload test data for loaded model
    if st.session_state.model_trained:
        st.markdown("---")
        st.subheader("Upload Test Data for Predictions")
        test_file_loaded = st.file_uploader("Upload Testing Data (Parquet)", type=['parquet'], key="test_upload_loaded")
        if test_file_loaded is not None:
            st.session_state.test_data = pd.read_parquet(test_file_loaded)
            st.success(f"Testing data loaded: {st.session_state.test_data.shape}")

# Main content area - shown for both new and loaded models
if st.session_state.model_trained:
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Model Summary", "📈 Performance Metrics", "🎯 Predictions", 
        "📉 Gini Analysis", "💾 Save Model", "🔧 Model Coefficients",
        "🔬 Model Stability & Sensitivity"
    ])
    
    # Tab 1-6: [Previous tabs remain the same]
    # ... [Keep all previous tab code here] ...
    
    # Tab 7: Model Stability & Sensitivity Analysis
    with tab7:
        st.header("🔬 Model Stability & Sensitivity Analysis")
        
        if st.session_state.test_data is not None:
            # Check for required columns
            required_cols = ['region', 'final_rating', 'model_rating', 'cohort_date']
            missing_cols = [col for col in required_cols if col not in st.session_state.test_data.columns]
            
            if missing_cols:
                st.warning(f"Missing required columns for stability analysis: {', '.join(missing_cols)}")
                st.info("Please ensure your data contains: region, final_rating, model_rating, cohort_date")
            else:
                # Generate predictions if not already done
                X_test = st.session_state.test_data[st.session_state.selected_features]
                X_test_scaled = st.session_state.scaler.transform(X_test)
                
                # Get probability scores
                prob_scores = st.session_state.model.predict_proba(X_test_scaled)[:, 1]
                
                # Add predictions to test data
                analysis_df = st.session_state.test_data.copy()
                analysis_df['predicted_probability'] = prob_scores
                analysis_df['predicted_class'] = st.session_state.model.predict(X_test_scaled)
                
                # Convert cohort_date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(analysis_df['cohort_date']):
                    analysis_df['cohort_date'] = pd.to_datetime(analysis_df['cohort_date'])
                
                # Create subtabs for different analyses
                stab1, stab2, stab3, stab4, stab5 = st.tabs([
                    "📊 Stability Over Time", "🌍 Regional Performance", 
                    "📈 Rating Migration", "🎯 Sensitivity Analysis", "📋 Comprehensive Report"
                ])
                
                # Subtab 1: Stability Over Time
                with stab1:
                    st.subheader("Model Stability Over Time")
                    
                    # Group by cohort date
                    cohort_analysis = analysis_df.groupby(pd.Grouper(key='cohort_date', freq='M')).agg({
                        'predicted_probability': ['mean', 'std', 'count'],
                        st.session_state.target_variable: 'mean'
                    }).reset_index()
                    
                    cohort_analysis.columns = ['cohort_date', 'avg_score', 'std_score', 'count', 'actual_rate']
                    
                    # Plot stability metrics
                    fig_stability = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Average Score Over Time', 'Score Volatility', 
                                      'Sample Size', 'Actual vs Predicted Rate')
                    )
                    
                    # Average score
                    fig_stability.add_trace(
                        go.Scatter(x=cohort_analysis['cohort_date'], 
                                 y=cohort_analysis['avg_score'],
                                 mode='lines+markers', name='Avg Score'),
                        row=1, col=1
                    )
                    
                    # Score volatility
                    fig_stability.add_trace(
                        go.Scatter(x=cohort_analysis['cohort_date'], 
                                 y=cohort_analysis['std_score'],
                                 mode='lines+markers', name='Std Dev',
                                 line=dict(color='red')),
                        row=1, col=2
                    )
                    
                    # Sample size
                    fig_stability.add_trace(
                        go.Bar(x=cohort_analysis['cohort_date'], 
                              y=cohort_analysis['count'],
                              name='Count'),
                        row=2, col=1
                    )
                    
                    # Actual vs Predicted
                    fig_stability.add_trace(
                        go.Scatter(x=cohort_analysis['cohort_date'], 
                                 y=cohort_analysis['actual_rate'],
                                 mode='lines+markers', name='Actual Rate'),
                        row=2, col=2
                    )
                    fig_stability.add_trace(
                        go.Scatter(x=cohort_analysis['cohort_date'], 
                                 y=cohort_analysis['avg_score'],
                                 mode='lines+markers', name='Predicted Rate',
                                 line=dict(dash='dash')),
                        row=2, col=2
                    )
                    
                    fig_stability.update_layout(height=800, showlegend=True)
                    st.plotly_chart(fig_stability, use_container_width=True)
                    
                    # Population Stability Index (PSI)
                    st.subheader("Population Stability Index (PSI)")
                    
                    # Calculate PSI
                    reference_month = cohort_analysis['cohort_date'].min()
                    reference_data = analysis_df[analysis_df['cohort_date'] == reference_month]['predicted_probability']
                    
                    psi_results = []
                    for month in cohort_analysis['cohort_date'].unique():
                        if month != reference_month:
                            current_data = analysis_df[analysis_df['cohort_date'] == month]['predicted_probability']
                            psi = calculate_psi(reference_data, current_data)
                            psi_results.append({
                                'cohort_date': month,
                                'PSI': psi,
                                'Status': 'Stable' if psi < 0.1 else 'Moderate Shift' if psi < 0.25 else 'Significant Shift'
                            })
                    
                    if psi_results:
                        psi_df = pd.DataFrame(psi_results)
                        
                        fig_psi = px.line(psi_df, x='cohort_date', y='PSI', 
                                        title='Population Stability Index Over Time',
                                        color_discrete_sequence=['blue'])
                        fig_psi.add_hline(y=0.1, line_dash="dash", line_color="green", 
                                        annotation_text="Stable Threshold")
                        fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", 
                                        annotation_text="Significant Shift Threshold")
                        st.plotly_chart(fig_psi, use_container_width=True)
                        
                        st.dataframe(psi_df, use_container_width=True)
                
                # Subtab 2: Regional Performance
                with stab2:
                    st.subheader("Regional Performance Analysis")
                    
                    # Regional metrics
                    regional_analysis = analysis_df.groupby('region').agg({
                        'predicted_probability': ['mean', 'std', 'count'],
                        st.session_state.target_variable: ['mean', 'sum'],
                        'predicted_class': 'sum'
                    }).reset_index()
                    
                    regional_analysis.columns = ['region',
                    
                                        regional_analysis.columns = ['region', 'avg_score', 'std_score', 'count', 
                                                'actual_rate', 'actual_positives', 'predicted_positives']
                    
                    # Calculate regional performance metrics
                    regional_metrics = []
                    for region in regional_analysis['region'].unique():
                        region_data = analysis_df[analysis_df['region'] == region]
                        
                        # Calculate metrics
                        y_true = region_data[st.session_state.target_variable]
                        y_pred = region_data['predicted_class']
                        y_prob = region_data['predicted_probability']
                        
                        if len(np.unique(y_true)) == 2:
                            auc = roc_auc_score(y_true, y_prob)
                            gini = 2 * auc - 1
                        else:
                            auc = np.nan
                            gini = np.nan
                        
                        regional_metrics.append({
                            'Region': region,
                            'Sample Size': len(region_data),
                            'Actual Rate': y_true.mean(),
                            'Predicted Rate': y_prob.mean(),
                            'Accuracy': accuracy_score(y_true, y_pred),
                            'AUC': auc,
                            'Gini': gini,
                            'Rate Difference': abs(y_true.mean() - y_prob.mean())
                        })
                    
                    regional_metrics_df = pd.DataFrame(regional_metrics).sort_values('Gini', ascending=False)
                    
                    # Display regional metrics
                    st.dataframe(
                        regional_metrics_df.style.format({
                            'Actual Rate': '{:.2%}',
                            'Predicted Rate': '{:.2%}',
                            'Accuracy': '{:.4f}',
                            'AUC': '{:.4f}',
                            'Gini': '{:.4f}',
                            'Rate Difference': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Regional visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Regional Gini comparison
                        fig_regional_gini = px.bar(
                            regional_metrics_df,
                            x='Region',
                            y='Gini',
                            title='Gini Coefficient by Region',
                            color='Gini',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_regional_gini, use_container_width=True)
                    
                    with col2:
                        # Actual vs Predicted rates by region
                        fig_rates = go.Figure()
                        fig_rates.add_trace(go.Bar(
                            x=regional_metrics_df['Region'],
                            y=regional_metrics_df['Actual Rate'],
                            name='Actual Rate',
                            marker_color='lightblue'
                        ))
                        fig_rates.add_trace(go.Bar(
                            x=regional_metrics_df['Region'],
                            y=regional_metrics_df['Predicted Rate'],
                            name='Predicted Rate',
                            marker_color='darkblue'
                        ))
                        fig_rates.update_layout(
                            title='Actual vs Predicted Rates by Region',
                            barmode='group',
                            yaxis_title='Rate'
                        )
                        st.plotly_chart(fig_rates, use_container_width=True)
                    
                    # Heatmap of regional performance
                    st.subheader("Regional Performance Heatmap")
                    
                    # Prepare data for heatmap
                    heatmap_data = regional_metrics_df[['Region', 'Accuracy', 'AUC', 'Gini', 'Rate Difference']]
                    heatmap_data_scaled = heatmap_data.set_index('Region')
                    
                    # Normalize data for better visualization
                    from sklearn.preprocessing import MinMaxScaler
                    scaler_heatmap = MinMaxScaler()
                    heatmap_normalized = pd.DataFrame(
                        scaler_heatmap.fit_transform(heatmap_data_scaled),
                        index=heatmap_data_scaled.index,
                        columns=heatmap_data_scaled.columns
                    )
                    
                    fig_heatmap = px.imshow(
                        heatmap_normalized.T,
                        labels=dict(x="Region", y="Metric", color="Normalized Score"),
                        title="Regional Performance Heatmap (Normalized)",
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Subtab 3: Rating Migration
                with stab3:
                    st.subheader("Rating Migration Analysis")
                    
                    # Compare final_rating vs model_rating
                    if 'final_rating' in analysis_df.columns and 'model_rating' in analysis_df.columns:
                        # Create migration matrix
                        migration_matrix = pd.crosstab(
                            analysis_df['final_rating'],
                            analysis_df['model_rating'],
                            normalize='index'
                        ) * 100
                        
                        # Display migration matrix
                        st.write("### Rating Migration Matrix (%)")
                        st.dataframe(
                            migration_matrix.style.format('{:.1f}%').background_gradient(cmap='Blues'),
                            use_container_width=True
                        )
                        
                        # Visualize migration
                        fig_migration = px.imshow(
                            migration_matrix,
                            labels=dict(x="Model Rating", y="Final Rating", color="Migration %"),
                            title="Rating Migration Heatmap",
                            color_continuous_scale='Blues',
                            text_auto='.1f'
                        )
                        st.plotly_chart(fig_migration, use_container_width=True)
                        
                        # Rating agreement analysis
                        st.subheader("Rating Agreement Analysis")
                        
                        # Calculate agreement metrics
                        exact_match = (analysis_df['final_rating'] == analysis_df['model_rating']).mean()
                        
                        # One-notch difference
                        rating_diff = abs(pd.Categorical(analysis_df['final_rating']).codes - 
                                        pd.Categorical(analysis_df['model_rating']).codes)
                        one_notch = (rating_diff <= 1).mean()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Exact Match Rate", f"{exact_match:.2%}")
                        with col2:
                            st.metric("Within One Notch", f"{one_notch:.2%}")
                        with col3:
                            st.metric("Average Rating Difference", f"{rating_diff.mean():.2f}")
                        
                        # Rating distribution comparison
                        fig_rating_dist = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('Final Rating Distribution', 'Model Rating Distribution')
                        )
                        
                        # Final rating distribution
                        final_counts = analysis_df['final_rating'].value_counts().sort_index()
                        fig_rating_dist.add_trace(
                            go.Bar(x=final_counts.index, y=final_counts.values, name='Final Rating'),
                            row=1, col=1
                        )
                        
                        # Model rating distribution
                        model_counts = analysis_df['model_rating'].value_counts().sort_index()
                        fig_rating_dist.add_trace(
                            go.Bar(x=model_counts.index, y=model_counts.values, name='Model Rating'),
                            row=1, col=2
                        )
                        
                        fig_rating_dist.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_rating_dist, use_container_width=True)
                
                # Subtab 4: Sensitivity Analysis
                with stab4:
                    st.subheader("Model Sensitivity Analysis")
                    
                    # Feature sensitivity analysis
                    st.write("### Feature Perturbation Analysis")
                    
                    sensitivity_results = []
                    
                    # Select features for sensitivity analysis
                    features_to_analyze = st.multiselect(
                        "Select features for sensitivity analysis",
                        options=st.session_state.selected_features,
                        default=st.session_state.selected_features[:5] if len(st.session_state.selected_features) >= 5 else st.session_state.selected_features
                    )
                    
                    if st.button("Run Sensitivity Analysis"):
                        progress_bar = st.progress(0)
                        
                        for idx, feature in enumerate(features_to_analyze):
                            # Create perturbed datasets
                            X_test_perturbed = X_test.copy()
                            
                            # Calculate perturbations
                            feature_std = X_test[feature].std()
                            perturbations = [-2*feature_std, -feature_std, 0, feature_std, 2*feature_std]
                            
                            feature_sensitivity = []
                            
                            for perturb in perturbations:
                                X_test_perturbed[feature] = X_test[feature] + perturb
                                X_test_perturbed_scaled = st.session_state.scaler.transform(X_test_perturbed)
                                
                                # Get new predictions
                                new_probs = st.session_state.model.predict_proba(X_test_perturbed_scaled)[:, 1]
                                
                                # Calculate change in average probability
                                avg_change = (new_probs.mean() - prob_scores.mean()) / prob_scores.mean() * 100
                                
                                feature_sensitivity.append({
                                    'Feature': feature,
                                    'Perturbation': f"{perturb/feature_std:.1f} std",
                                    'Avg_Probability_Change_%': avg_change
                                })
                            
                            sensitivity_results.extend(feature_sensitivity)
                            progress_bar.progress((idx + 1) / len(features_to_analyze))
                        
                        # Display sensitivity results
                        sensitivity_df = pd.DataFrame(sensitivity_results)
                        
                        # Pivot for better visualization
                        sensitivity_pivot = sensitivity_df.pivot(
                            index='Feature',
                            columns='Perturbation',
                            values='Avg_Probability_Change_%'
                        )
                        
                        # Create sensitivity heatmap
                        fig_sensitivity = px.imshow(
                            sensitivity_pivot,
                            labels=dict(x="Perturbation", y="Feature", color="Avg Probability Change (%)"),
                            title="Feature Sensitivity Heatmap",
                            color_continuous_scale='RdBu',
                            color_continuous_midpoint=0
                        )
                        st.plotly_chart(fig_sensitivity, use_container_width=True)
                        
                        # Feature importance based on sensitivity
                        feature_importance = sensitivity_pivot.abs().mean(axis=1).sort_values(ascending=False)
                        
                        fig_importance = px.bar(
                            x=feature_importance.values,
                            y=feature_importance.index,
                            orientation='h',
                            title='Feature Importance (Based on Sensitivity)',
                            labels={'x': 'Average Absolute Change (%)', 'y': 'Feature'}
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Threshold sensitivity
                    st.write("### Threshold Sensitivity Analysis")
                    
                    thresholds = np.linspace(0.1, 0.9, 9)
                    threshold_metrics = []
                    
                    for threshold in thresholds:
                        y_pred_threshold = (prob_scores >= threshold).astype(int)
                        
                        threshold_metrics.append({
                            'Threshold': threshold,
                            'Accuracy': accuracy_score(analysis_df[st.session_state.target_variable], y_pred_threshold),
                            'Precision': precision_score(analysis_df[st.session_state.target_variable], y_pred_threshold, zero_division=0),
                            'Recall': recall_score(analysis_df[st.session_state.target_variable], y_pred_threshold, zero_division=0),
                            'F1': f1_score(analysis_df[st.session_state.target_variable], y_pred_threshold, zero_division=0),
                            'Positive_Rate': y_pred_threshold.mean()
                        })
                    
                    threshold_df = pd.DataFrame(threshold_metrics)
                    
                    # Plot threshold sensitivity
                    fig_threshold = go.Figure()
                    
                    for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
                        fig_threshold.add_trace(go.Scatter(
                            x=threshold_df['Threshold'],
                            y=threshold_df[metric],
                            mode='lines+markers',
                            name=metric
                        ))
                    
                    fig_threshold.update_layout(
                        title='Model Performance vs Decision Threshold',
                        xaxis_title='Threshold',
                        yaxis_title='Score',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_threshold, use_container_width=True)
                
                # Subtab 5: Comprehensive Report
                with stab5:
                    st.subheader("📋 Comprehensive Stability Report")
                    
                    # Generate comprehensive report
                    report_data = {
                        'Report Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Model Type': 'Logistic Regression',
                        'Number of Features': len(st.session_state.selected_features),
                        'Test Sample Size': len(analysis_df),
                        'Date Range': f"{analysis_df['cohort_date'].min()} to {analysis_df['cohort_date'].max()}",
                        'Number of Regions': analysis_df['region'].nunique(),
                        'Overall Accuracy': accuracy_score(analysis_df[st.session_state.target_variable], 
                                                         analysis_df['predicted_class']),
                        'Overall AUC': roc_auc_score(analysis_df[st.session_state.target_variable], 
                                                   analysis_df['predicted_probability']) if len(np.unique(analysis_df[st.session_state.target_variable])) == 2 else 'N/A'
                    }
                    
                    # Display report summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Model Information")
                        for key, value in list(report_data.items())[:4]:
                            st.write(f"**{key}:** {value}")
                    
                    with col2:
                        st.write("### Performance Summary")
                        for key, value in list(report_data.items())[4:]:
                            st.write(f"**{key}:** {value}")
                    
                    # Key findings
                    st.write("### Key Findings")
                    
                    # Calculate key metrics
                    if 'psi_df' in locals():
                        avg_psi = psi_df['PSI'].mean()
                        max_psi = psi_df['PSI'].max()
                        stability_status = "Stable" if max_psi < 0.1 else "Moderate Shift" if max_psi < 0.25 else
                        
                        
                        
                        
                                         stability_status = "Stable" if max_psi < 0.1 else "Moderate Shift" if max_psi < 0.25 else "Significant Shift"
                    else:
                        avg_psi = "N/A"
                        max_psi = "N/A"
                        stability_status = "Unable to calculate"
                    
                    findings = {
                        'Population Stability': stability_status,
                        'Average PSI': f"{avg_psi:.4f}" if isinstance(avg_psi, float) else avg_psi,
                        'Maximum PSI': f"{max_psi:.4f}" if isinstance(max_psi, float) else max_psi,
                        'Best Performing Region': regional_metrics_df.loc[regional_metrics_df['Gini'].idxmax(), 'Region'] if not regional_metrics_df.empty else "N/A",
                        'Worst Performing Region': regional_metrics_df.loc[regional_metrics_df['Gini'].idxmin(), 'Region'] if not regional_metrics_df.empty else "N/A",
                        'Rating Agreement Rate': f"{exact_match:.2%}" if 'exact_match' in locals() else "N/A",
                        'Most Sensitive Feature': feature_importance.idxmax() if 'feature_importance' in locals() else "N/A"
                    }
                    
                    for finding, value in findings.items():
                        st.write(f"- **{finding}:** {value}")
                    
                    # Generate downloadable report
                    st.write("### Download Full Report")
                    
                    # Create detailed report dataframe
                    full_report = []
                    
                    # Add summary section
                    full_report.append(pd.DataFrame([
                        ['COMPREHENSIVE MODEL STABILITY REPORT', ''],
                        ['Generated Date', report_data['Report Date']],
                        ['', ''],
                        ['MODEL INFORMATION', ''],
                        ['Model Type', report_data['Model Type']],
                        ['Number of Features', report_data['Number of Features']],
                        ['Test Sample Size', report_data['Test Sample Size']],
                        ['', ''],
                        ['STABILITY METRICS', ''],
                        ['Population Stability Status', stability_status],
                        ['Average PSI', avg_psi],
                        ['Maximum PSI', max_psi],
                        ['', '']
                    ], columns=['Metric', 'Value']))
                    
                    # Add regional performance
                    if not regional_metrics_df.empty:
                        full_report.append(pd.DataFrame([['REGIONAL PERFORMANCE', '']], columns=['Metric', 'Value']))
                        regional_summary = regional_metrics_df[['Region', 'Gini', 'AUC', 'Accuracy']].copy()
                        regional_summary.columns = ['Metric', 'Gini', 'AUC', 'Accuracy']
                        full_report.append(regional_summary)
                    
                    # Combine all sections
                    final_report = pd.concat(full_report, ignore_index=True)
                    
                    # Create Excel file with multiple sheets
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Summary sheet
                        final_report.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Regional metrics sheet
                        if not regional_metrics_df.empty:
                            regional_metrics_df.to_excel(writer, sheet_name='Regional Analysis', index=False)
                        
                        # PSI analysis sheet
                        if 'psi_df' in locals():
                            psi_df.to_excel(writer, sheet_name='PSI Analysis', index=False)
                        
                        # Threshold analysis sheet
                        if 'threshold_df' in locals():
                            threshold_df.to_excel(writer, sheet_name='Threshold Analysis', index=False)
                        
                        # Feature sensitivity sheet
                        if 'sensitivity_pivot' in locals():
                            sensitivity_pivot.to_excel(writer, sheet_name='Feature Sensitivity')
                    
                    output.seek(0)
                    
                    st.download_button(
                        label="📥 Download Comprehensive Report (Excel)",
                        data=output,
                        file_name=f"model_stability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Also provide CSV option for the summary
                    csv_report = final_report.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary Report (CSV)",
                        data=csv_report,
                        file_name=f"model_stability_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Please upload test data to perform stability and sensitivity analysis.")

# Helper function for PSI calculation
def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)"""
    def psi_bucket(expected, actual, buckets):
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Bin the data
        expected_counts = np.histogram(expected, breakpoints)[0]
        actual_counts = np.histogram(actual, breakpoints)[0]
        
        # Calculate PSI
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI for each bucket
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        
        return np.sum(psi_values)
    
    return psi_bucket(expected, actual, buckets)

# Add custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Advanced Logistic Regression Dashboard v2.0")


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_dummy_data(n_samples=10000, n_features=10, train_test_split=0.7):
    """
    Create dummy training and testing dataframes with all required columns
    """
    
    # Define regions and ratings
    regions = ['North', 'South', 'East', 'West', 'Central']
    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
    
    # Generate date range for cohort_date (last 12 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Generate base features with some correlation structure
    # Create a correlation matrix
    correlation_matrix = np.random.rand(n_features, n_features)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1)
    
    # Generate correlated features
    mean = np.zeros(n_features)
    features = np.random.multivariate_normal(mean, correlation_matrix, n_samples)
    
    # Create base probability influenced by features
    # Use first 3 features as main drivers
    logits = (
        0.5 * features[:, 0] + 
        0.3 * features[:, 1] - 
        0.4 * features[:, 2] + 
        0.2 * features[:, 3] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Convert to probabilities
    probabilities = 1 / (1 + np.exp(-logits))
    
    # Create target variable (binary)
    target = (probabilities + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add target variable
    df['target'] = target
    
    # Add region (with some correlation to features)
    region_probabilities = np.abs(features[:, 0]) / np.abs(features[:, 0]).max()
    df['region'] = np.random.choice(regions, size=n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    # Add cohort_date
    days_range = (end_date - start_date).days
    random_days = np.random.randint(0, days_range, n_samples)
    df['cohort_date'] = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # Create model scores (correlated with true probability)
    model_scores = probabilities + np.random.normal(0, 0.05, n_samples)
    model_scores = np.clip(model_scores, 0, 1)
    
    # Convert scores to ratings
    def score_to_rating(score):
        if score > 0.9:
            return 'AAA'
        elif score > 0.8:
            return 'AA'
        elif score > 0.7:
            return 'A'
        elif score > 0.6:
            return 'BBB'
        elif score > 0.5:
            return 'BB'
        elif score > 0.4:
            return 'B'
        else:
            return 'CCC'
    
    df['model_rating'] = [score_to_rating(s) for s in model_scores]
    
    # Create final_rating (similar to model_rating but with some noise)
    final_scores = model_scores + np.random.normal(0, 0.1, n_samples)
    final_scores = np.clip(final_scores, 0, 1)
    df['final_rating'] = [score_to_rating(s) for s in final_scores]
    
    # Add some additional realistic columns
    df['customer_id'] = [f'CUST_{i:06d}' for i in range(n_samples)]
    df['account_age_months'] = np.random.randint(1, 120, n_samples)
    df['transaction_count'] = np.random.poisson(50, n_samples)
    df['avg_transaction_amount'] = np.random.lognormal(4, 1.5, n_samples)
    
    # Split into train and test
    split_idx = int(n_samples * train_test_split)
    
    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Ensure test data has more recent dates (for realistic time-based validation)
    test_df['cohort_date'] = test_df['cohort_date'] + timedelta(days=30)
    
    return train_df, test_df

def save_dummy_data_to_parquet():
    """
    Generate and save dummy data to parquet files
    """
    print("Generating dummy data...")
    train_df, test_df = create_dummy_data(
        n_samples=10000,
        n_features=10,
        train_test_split=0.7
    )
    
    # Save to parquet files
    train_df.to_parquet('train_data_dummy.parquet', index=False)
    test_df.to_parquet('test_data_dummy.parquet', index=False)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    print("\nTraining data columns:")
    print(train_df.columns.tolist())
    print("\nData types:")
    print(train_df.dtypes)
    print("\nFirst few rows of training data:")
    print(train_df.head())
    print("\nTarget distribution in training data:")
    print(train_df['target'].value_counts(normalize=True))
    print("\nRegion distribution in training data:")
    print(train_df['region'].value_counts())
    print("\nRating distribution in training data:")
    print(train_df['model_rating'].value_counts())
    
    return train_df, test_df

# Generate and save the dummy data
if __name__ == "__main__":
    train_df, test_df = save_dummy_data_to_parquet()
    
    # Create a sample code to test specific scenarios
    print("\n" + "="*50)
    print("TESTING SCENARIOS")
    print("="*50)
    
    # Scenario 1: Check rating migration
    print("\n1. Rating Migration Matrix Preview:")
    migration = pd.crosstab(test_df['final_rating'], test_df['model_rating'])
    print(migration)
    
    # Scenario 2: Regional distribution
    print("\n2. Regional Distribution:")
    print(test_df.groupby('region')['target'].agg(['count', 'mean']))
    
    # Scenario 3: Time-based distribution
    print("\n3. Monthly Distribution:")
    test_df['month'] = pd.to_datetime(test_df['cohort_date']).dt.to_period('M')
    print(test_df.groupby('month')['target'].agg(['count', 'mean']))
    
    # Create a smaller sample file for quick testing
    train_sample = train_df.sample(n=1000, random_state=42)
    test_sample = test_df.sample(n=300, random_state=42)
    
    train_sample.to_parquet('train_data_sample.parquet', index=False)
    test_sample.to_parquet('test_data_sample.parquet', index=False)
    
    print("\n" + "="*50)
    print("FILES CREATED:")
    print("="*50)
    print("1. train_data_dummy.parquet (7,000 rows)")
    print("2. test_data_dummy.parquet (3,000 rows)")
    print("3. train_data_sample.parquet (1,000 rows) - for quick testing")
    print("4. test_data_sample.parquet (300 rows) - for quick testing")
    
    # Create a data dictionary
    data_dict = {
        'Column': ['target', 'feature_1 to feature_10', 'region', 'cohort_date', 
                   'model_rating', 'final_rating', 'customer_id', 'account_age_months',
                   'transaction_count', 'avg_transaction_amount'],
        'Description': [
            'Binary target variable (0 or 1)',
            'Numerical features for model training',
            'Geographic region (North, South, East, West, Central)',
            'Date when the rating was generated',
            'Rating assigned by the model (AAA to CCC)',
            'Final rating after review (AAA to CCC)',
            'Unique customer identifier',
            'Age of account in months',
            'Number of transactions',
            'Average transaction amount'
        ],
        'Type': ['Binary', 'Numeric', 'Categorical', 'Datetime', 
                 'Ordinal', 'Ordinal', 'String', 'Integer',
                 'Integer', 'Numeric']
    }
    
    data_dict_df = pd.DataFrame(data_dict)
    print("\n" + "="*50)
    print("DATA DICTIONARY:")
    print("="*50)
    print(data_dict_df.to_string(index=False))


