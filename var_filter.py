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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Model Summary", "📈 Performance Metrics", 
                                                   "🎯 Predictions", "📉 Gini Analysis", 
                                                   "💾 Save Model", "🔧 Model Coefficients"])
    
    # Tab 1: Model Summary
    with tab1:
        st.header("Model Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Coefficients")
            coef_df = pd.DataFrame({
                'Feature': st.session_state.selected_features,
                'Coefficient': st.session_state.model.coef_[0],
                'Abs_Coefficient': np.abs(st.session_state.model.coef_[0])
            }).sort_values('Abs_Coefficient', ascending=False)
            
            st.dataframe(coef_df, use_container_width=True)
            
            # Feature importance plot
            fig_importance = px.bar(
                coef_df, 
                x='Coefficient', 
                y='Feature',
                orientation='h',
                title='Feature Coefficients',
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.subheader("Model Information")
            model_info = {
                'Intercept': st.session_state.model.intercept_[0],
                'Number of Features': len(st.session_state.selected_features),
                'Regularization': st.session_state.model.penalty,
                'C Value': st.session_state.model.C,
                'Max Iterations': st.session_state.model.max_iter,
                'Convergence': 'Yes' if st.session_state.model.n_iter_ < st.session_state.model.max_iter else 'No'
            }
            
            for key, value in model_info.items():
                st.metric(key, value)
    
    # Tab 2: Performance Metrics
    with tab2:
        st.header("Performance Metrics")
        
        if st.session_state.test_data is not None:
            # Check if test data has required columns
            missing_features = [f for f in st.session_state.selected_features if f not in st.session_state.test_data.columns]
            
            if missing_features:
                st.error(f"Test data is missing the following features: {', '.join(missing_features)}")
            elif st.session_state.target_variable not in st.session_state.test_data.columns:
                st.error(f"Test data is missing the target variable: {st.session_state.target_variable}")
            else:
                # Prepare test data
                X_test = st.session_state.test_data[st.session_state.selected_features]
                y_test = st.session_state.test_data[st.session_state.target_variable]
                X_test_scaled = st.session_state.scaler.transform(X_test)
                
                # Make predictions
                y_pred = st.session_state.model.predict(X_test_scaled)
                y_pred_proba = st.session_state.model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.4f}")
                
                with col2:
                    precision = precision_score(y_test, y_pred, average='weighted')
                    st.metric("Precision", f"{precision:.4f}")
                
                with col3:
                    recall = recall_score(y_test, y_pred, average='weighted')
                    st.metric("Recall", f"{recall:.4f}")
                
                with col4:
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    st.metric("F1 Score", f"{f1:.4f}")
                
                # ROC AUC
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    st.metric("ROC AUC", f"{roc_auc:.4f}")
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {roc_auc:.4f})'))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                    fig_roc.update_layout(
                        title='ROC Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate'
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
        else:
            st.warning("Please upload test data to see performance metrics.")
    
    # Tab 3: Predictions
    with tab3:
        st.header("Generate Predictions")
        
        if st.session_state.test_data is not None:
            missing_features = [f for f in st.session_state.selected_features if f not in st.session_state.test_data.columns]
            
            if missing_features:
                st.error(f"Test data is missing the following features: {', '.join(missing_features)}")
            else:
                X_test = st.session_state.test_data[st.session_state.selected_features]
                X_test_scaled = st.session_state.scaler.transform(X_test)
                
                # Generate predictions
                predictions = st.session_state.model.predict(X_test_scaled)
                probabilities = st.session_state.model.predict_proba(X_test_scaled)
                
                # Create results dataframe
                results_df = st.session_state.test_data.copy()
                results_df['Predicted'] = predictions
                results_df['Probability_Class_0'] = probabilities[:, 0]
                results_df['Probability_Class_1'] = probabilities[:, 1]
                
                st.subheader("Prediction Results")
                st.dataframe(results_df.head(100), use_container_width=True)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Score distribution
                st.subheader("Score Distribution")
                
                                
                        fig_dist = px.histogram(
                    x=probabilities[:, 1],
                    nbins=50,
                    title="Distribution of Predicted Probabilities (Class 1)",
                    labels={'x': 'Probability', 'y': 'Count'}
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("Please upload test data to generate predictions.")
    
    # Tab 4: Gini Analysis
    with tab4:
        st.header("Gini Coefficient Analysis")
        
        if st.session_state.test_data is not None and st.session_state.train_data is not None:
            # Check if we have training data for Gini analysis
            if st.session_state.train_data is None:
                st.warning("Training data is required for Gini analysis. Please upload training data.")
            else:
                gini_results = []
                
                for feature in st.session_state.selected_features:
                    try:
                        # Train single feature model
                        X_single = st.session_state.train_data[[feature]]
                        y_train = st.session_state.train_data[st.session_state.target_variable]
                        
                        scaler_single = StandardScaler()
                        X_single_scaled = scaler_single.fit_transform(X_single)
                        
                        model_single = LogisticRegression(random_state=42)
                        model_single.fit(X_single_scaled, y_train)
                        
                        # Test on single feature
                        X_test_single = st.session_state.test_data[[feature]]
                        y_test = st.session_state.test_data[st.session_state.target_variable]
                        X_test_single_scaled = scaler_single.transform(X_test_single)
                        
                        y_pred_proba_single = model_single.predict_proba(X_test_single_scaled)[:, 1]
                        
                        # Calculate Gini coefficient
                        if len(np.unique(y_test)) == 2:
                            auc = roc_auc_score(y_test, y_pred_proba_single)
                            gini = 2 * auc - 1
                            
                            gini_results.append({
                                'Feature': feature,
                                'AUC': auc,
                                'Gini': gini,
                                'Coefficient': st.session_state.model.coef_[0][st.session_state.selected_features.index(feature)]
                            })
                    except Exception as e:
                        st.warning(f"Could not calculate Gini for feature {feature}: {str(e)}")
                
                if gini_results:
                    # Display Gini table
                    gini_df = pd.DataFrame(gini_results).sort_values('Gini', ascending=False)
                    st.dataframe(gini_df, use_container_width=True)
                    
                    # Gini visualization
                    fig_gini = px.bar(
                        gini_df,
                        x='Feature',
                        y='Gini',
                        title='Gini Coefficients by Feature',
                        color='Gini',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_gini, use_container_width=True)
                    
                    # Download Gini table
                    csv_gini = gini_df.to_csv(index=False)
                    st.download_button(
                        label="Download Gini Table as CSV",
                        data=csv_gini,
                        file_name=f"gini_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Could not calculate Gini coefficients.")
        else:
            st.warning("Please upload both training and test data to perform Gini analysis.")
    
    # Tab 5: Save Model
    with tab5:
        st.header("Save Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Save Model File")
            model_name = st.text_input("Model Name", value="logistic_regression_model")
            
            if st.button("Save Model"):
                # Save model and scaler
                model_data = {
                    'model': st.session_state.model,
                    'scaler': st.session_state.scaler,
                    'features': st.session_state.selected_features,
                    'target': st.session_state.target_variable,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Create a bytes buffer
                buffer = io.BytesIO()
                joblib.dump(model_data, buffer)
                buffer.seek(0)
                
                # Download button
                st.download_button(
                    label="Download Model",
                    data=buffer,
                    file_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                    mime="application/octet-stream"
                )
                
                st.success("Model saved successfully!")
        
        with col2:
            st.subheader("Model Metadata")
            st.json({
                'model_type': 'Logistic Regression',
                'features': st.session_state.selected_features,
                'target': st.session_state.target_variable,
                'regularization': st.session_state.model.penalty,
                'C_value': st.session_state.model.C,
                'n_features': len(st.session_state.selected_features),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Tab 6: Model Coefficients (Detailed View)
    with tab6:
        st.header("Detailed Model Coefficients")
        
        # Create comprehensive coefficient analysis
        coef_detailed = pd.DataFrame({
            'Feature': st.session_state.selected_features,
            'Coefficient': st.session_state.model.coef_[0],
            'Abs_Coefficient': np.abs(st.session_state.model.coef_[0]),
            'Exp_Coefficient': np.exp(st.session_state.model.coef_[0])
        })
        
        # Add odds ratio interpretation
        coef_detailed['Odds_Ratio'] = coef_detailed['Exp_Coefficient']
        coef_detailed['Percentage_Change'] = (coef_detailed['Odds_Ratio'] - 1) * 100
        
        # Sort by absolute coefficient
        coef_detailed = coef_detailed.sort_values('Abs_Coefficient', ascending=False)
        
        st.subheader("Coefficient Table")
        st.dataframe(
            coef_detailed.style.format({
                'Coefficient': '{:.4f}',
                'Abs_Coefficient': '{:.4f}',
                'Exp_Coefficient': '{:.4f}',
                'Odds_Ratio': '{:.4f}',
                'Percentage_Change': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Interpretation guide
        with st.expander("📖 How to Interpret Coefficients"):
            st.markdown("""
            **Coefficient**: The raw coefficient from the logistic regression model.
            - Positive values indicate that as the feature increases, the log-odds of the positive class increase
            - Negative values indicate that as the feature increases, the log-odds of the positive class decrease
            
            **Odds Ratio (Exp_Coefficient)**: The exponential of the coefficient.
            - Values > 1 indicate increased odds of the positive class
            - Values < 1 indicate decreased odds of the positive class
            - Value = 1 indicates no effect
            
            **Percentage Change**: How much the odds change for a one-unit increase in the feature.
            - Positive percentages mean increased likelihood
            - Negative percentages mean decreased likelihood
            """)
        
        # Visualization of coefficients
        col1, col2 = st.columns(2)
        
        with col1:
            # Coefficient plot
            fig_coef = px.bar(
                coef_detailed.head(20),
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Top 20 Feature Coefficients',
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_coef, use_container_width=True)
        
        with col2:
            # Odds ratio plot
            fig_odds = px.bar(
                coef_detailed.head(20),
                x='Odds_Ratio',
                y='Feature',
                orientation='h',
                title='Top 20 Feature Odds Ratios',
                color='Odds_Ratio',
                color_continuous_scale='Viridis'
            )
            fig_odds.add_vline(x=1, line_dash="dash", line_color="red", 
                             annotation_text="No Effect")
            st.plotly_chart(fig_odds, use_container_width=True)
        
        # Download coefficient analysis
        csv_coef = coef_detailed.to_csv(index=False)
        st.download_button(
            label="Download Coefficient Analysis as CSV",
            data=csv_coef,
            file_name=f"coefficient_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Add a section for quick model loading and coefficient extraction
st.markdown("---")
st.header("🔍 Quick Model Coefficient Extractor")

with st.expander("Extract Coefficients from Saved Model"):
    quick_model_file = st.file_uploader("Upload Model File (.pkl)", type=['pkl'], key="quick_load")
    
    if quick_model_file is not None:
        try:
            # Load model
            model_data = joblib.load(quick_model_file)
            loaded_model = model_data['model']
            loaded_features = model_data['features']
            
            # Extract coefficients
            quick_coef_df = pd.DataFrame({
                'Feature': loaded_features,
                'Coefficient': loaded_model.coef_[0],
                'Abs_Coefficient': np.abs(loaded_model.coef_[0]),
                'Odds_Ratio': np.exp(loaded_model.coef_[0])
            }).sort_values('Abs_Coefficient', ascending=False)
            
            st.success("Model loaded successfully!")
            st.subheader("Model Coefficients")
            st.dataframe(quick_coef_df, use_container_width=True)
            
            # Model info
            st.subheader("Model Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Intercept", f"{loaded_model.intercept_[0]:.4f}")
            with col2:
                st.metric("Number of Features", len(loaded_features))
            with col3:
                st.metric("Model Type", loaded_model.penalty)
            
            # Download coefficients
            csv_quick = quick_coef_df.to_csv(index=False)
            st.download_button(
                label="Download Coefficients",
                data=csv_quick,
                file_name=f"model_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="quick_download"
            )
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Logistic Regression Dashboard v1.0")