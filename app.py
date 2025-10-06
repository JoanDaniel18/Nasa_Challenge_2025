"""
Streamlit dashboard for Ecuador climate prediction system
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import logging

from model_service import model_service
from data_service import data_service
from cache_service import cache
from config import ECUADOR_BBOX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Ecuador Climate Prediction System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data_with_caching(variable: str, days: int = 30):
    """Load data with caching"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    with st.spinner(f"Loading {variable} data..."):
        data = data_service.fetch_harmony_data(variable, start_date, end_date)
    
    return data

def plot_time_series(data: pd.DataFrame, variable: str, title: str):
    """Create time series plot"""
    if data.empty:
        st.warning(f"No data available for {variable}")
        return
    
    # Aggregate spatial data to daily averages
    daily_data = data_service.aggregate_spatial_data(data, method='mean')
    
    fig = px.line(
        daily_data, 
        x='date', 
        y='value',
        title=title,
        labels={'value': f'{variable.title()}', 'date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"{variable.title()}",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_spatial_data(data: pd.DataFrame, variable: str, date_filter: str = None):
    """Create spatial visualization"""
    if data.empty:
        st.warning(f"No spatial data available for {variable}")
        return
    
    # Filter data by date if specified
    if date_filter:
        filtered_date = pd.to_datetime(date_filter)
        plot_data = data[data['date'].dt.date == filtered_date.date()]
    else:
        # Use most recent data
        latest_date = data['date'].max()
        plot_data = data[data['date'] == latest_date]
    
    if plot_data.empty:
        st.warning("No data available for selected date")
        return
    
    fig = px.scatter_mapbox(
        plot_data,
        lat='latitude',
        lon='longitude',
        color='value',
        size='value',
        hover_data=['date', 'value'],
        color_continuous_scale='Viridis',
        size_max=15,
        zoom=6,
        center=dict(
            lat=(ECUADOR_BBOX['north'] + ECUADOR_BBOX['south']) / 2,
            lon=(ECUADOR_BBOX['east'] + ECUADOR_BBOX['west']) / 2
        ),
        title=f"{variable.title()} Distribution"
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    
    st.plotly_chart(fig, use_container_width=True)

def plot_model_comparison(comparison_data: dict):
    """Plot model performance comparison"""
    if not comparison_data or 'best_model' not in comparison_data:
        st.warning("No model comparison data available")
        return
    
    models = [k for k in comparison_data.keys() if k != 'best_model']
    metrics = ['test_rmse', 'test_r2', 'test_mae']
    
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metrics,
        specs=[[{"secondary_y": False}] * len(metrics)]
    )
    
    for i, metric in enumerate(metrics):
        values = [comparison_data[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(name=metric, x=models, y=values),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_predictions_vs_actual(model_key: str, target_variable: str):
    """Plot predictions vs actual values"""
    try:
        # Get recent data for comparison
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        X, y = data_service.prepare_ml_dataset(
            start_date=start_date,
            end_date=end_date,
            target_variable=target_variable
        )
        
        if X.empty:
            st.warning("No data available for validation plot")
            return
        
        # Get model predictions
        model = model_service.get_model(model_key)
        if model is None:
            st.warning("Model not found")
            return
        
        predictions = model_service.predict(model_key, X)
        
        if predictions is None:
            st.warning("Could not generate predictions for validation")
            return
        
        # Create comparison plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=X.index,
            y=y.values,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=X.index,
            y=predictions,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"Predictions vs Actual - {target_variable.title()}",
            xaxis_title="Date",
            yaxis_title=target_variable.title(),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating validation plot: {str(e)}")

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🌤️ Ecuador Climate Prediction System")
    st.markdown("Sistema de entrenamiento y predicción climática usando datos satelitales de NASA HARMONY")
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Data Explorer", "Model Training", "Predictions", "System Status"]
    )
    
    # Variable selection
    variables = ['temperature', 'precipitation', 'humidity', 'pressure']
    selected_variable = st.sidebar.selectbox("Select Variable", variables)
    
    if page == "Dashboard":
        st.header("Climate Data Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Load recent data for metrics
        recent_data = load_data_with_caching(selected_variable, days=7)
        
        if not recent_data.empty:
            latest_value = recent_data.groupby('date')['value'].mean().iloc[-1]
            avg_value = recent_data.groupby('date')['value'].mean().mean()
            max_value = recent_data.groupby('date')['value'].mean().max()
            min_value = recent_data.groupby('date')['value'].mean().min()
            
            with col1:
                st.metric(
                    label=f"Latest {selected_variable.title()}",
                    value=f"{latest_value:.2f}",
                    delta=f"{latest_value - avg_value:.2f}"
                )
            
            with col2:
                st.metric(
                    label="7-Day Average",
                    value=f"{avg_value:.2f}"
                )
            
            with col3:
                st.metric(
                    label="7-Day Maximum",
                    value=f"{max_value:.2f}"
                )
            
            with col4:
                st.metric(
                    label="7-Day Minimum",
                    value=f"{min_value:.2f}"
                )
        
        # Time series plot
        st.subheader(f"{selected_variable.title()} Time Series")
        data_30d = load_data_with_caching(selected_variable, days=30)
        plot_time_series(data_30d, selected_variable, f"{selected_variable.title()} - Last 30 Days")
        
        # Spatial distribution
        st.subheader(f"{selected_variable.title()} Spatial Distribution")
        if not data_30d.empty:
            plot_spatial_data(data_30d, selected_variable)
    
    elif page == "Data Explorer":
        st.header("Data Explorer")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30),
                max_value=datetime.now().date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        if st.button("Load Data"):
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())
            
            with st.spinner("Loading data..."):
                data = data_service.fetch_harmony_data(selected_variable, start_dt, end_dt)
            
            if not data.empty:
                st.success(f"Loaded {len(data)} data points")
                
                # Data summary
                st.subheader("Data Summary")
                daily_data = data_service.aggregate_spatial_data(data, method='mean')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Basic Statistics**")
                    st.dataframe(daily_data['value'].describe())
                
                with col2:
                    st.write("**Data Info**")
                    st.write(f"Total Records: {len(data)}")
                    st.write(f"Date Range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
                    st.write(f"Spatial Points: {len(data.groupby(['latitude', 'longitude']))}")
                
                # Plots
                plot_time_series(data, selected_variable, f"{selected_variable.title()} Time Series")
                
                # Raw data table
                if st.checkbox("Show Raw Data"):
                    st.subheader("Raw Data")
                    st.dataframe(data.head(1000))  # Limit for performance
            else:
                st.error("No data found for the selected parameters")
    
    elif page == "Model Training":
        st.header("Model Training & Evaluation")
        
        # Training controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Configuration")
            target_var = st.selectbox("Target Variable", variables, index=0)
            force_retrain = st.checkbox("Force Retraining", value=False)
            
            if st.button("Start Training"):
                with st.spinner("Training models..."):
                    try:
                        results = model_service.retrain_models(target_var)
                        
                        if 'error' in results:
                            st.error(f"Training failed: {results['error']}")
                        else:
                            st.success("Training completed successfully!")
                            
                            # Display results
                            if 'xgboost' in results and 'error' not in results['xgboost']:
                                st.write("**XGBoost Results:**")
                                xgb_metrics = results['xgboost']['metrics']
                                st.write(f"Test RMSE: {xgb_metrics['test_rmse']:.4f}")
                                st.write(f"Test R²: {xgb_metrics['test_r2']:.4f}")
                            
                            if 'random_forest' in results and 'error' not in results['random_forest']:
                                st.write("**Random Forest Results:**")
                                rf_metrics = results['random_forest']['metrics']
                                st.write(f"Test RMSE: {rf_metrics['test_rmse']:.4f}")
                                st.write(f"Test R²: {rf_metrics['test_r2']:.4f}")
                    
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
        
        with col2:
            st.subheader("Model Comparison")
            comparison = model_service.compare_models(target_var)
            
            if comparison:
                plot_model_comparison(comparison)
                
                if 'best_model' in comparison:
                    st.info(f"Best Model: {comparison['best_model']}")
            else:
                st.info("No trained models found. Start training to see comparison.")
        
        # Model validation
        st.subheader("Model Validation")
        available_models = model_service.get_model_info()['available_models']
        model_keys = [key for key in available_models if target_var in key]
        
        if model_keys:
            selected_model = st.selectbox("Select Model for Validation", model_keys)
            
            if st.button("Generate Validation Plot"):
                plot_predictions_vs_actual(selected_model, target_var)
        else:
            st.info("No trained models available for validation")
    
    elif page == "Predictions":
        st.header("Climate Predictions")
        
        # Prediction controls
        col1, col2 = st.columns(2)
        
        with col1:
            pred_variable = st.selectbox("Variable to Predict", variables)
            days_ahead = st.slider("Days Ahead", min_value=1, max_value=30, value=7)
            
            # Model selection
            comparison = model_service.compare_models(pred_variable)
            if comparison and 'best_model' in comparison:
                default_model = comparison['best_model']
                model_options = [key.split('_')[0] for key in model_service.get_model_info()['available_models'] if pred_variable in key]
                model_type = st.selectbox("Model Type", model_options, index=model_options.index(default_model) if default_model in model_options else 0)
            else:
                st.warning("No trained models found. Please train models first.")
                model_type = None
        
        with col2:
            if model_type and st.button("Generate Predictions"):
                model_key = f"{model_type}_{pred_variable}_model"
                
                with st.spinner("Generating predictions..."):
                    try:
                        result = model_service.predict_future(
                            model_key=model_key,
                            days_ahead=days_ahead,
                            target_variable=pred_variable
                        )
                        
                        if result:
                            st.success("Predictions generated successfully!")
                            
                            # Create prediction plot
                            pred_df = pd.DataFrame({
                                'date': pd.to_datetime(result['dates']),
                                'prediction': result['predictions']
                            })
                            
                            fig = px.line(
                                pred_df,
                                x='date',
                                y='prediction',
                                title=f"{pred_variable.title()} Predictions - Next {days_ahead} Days",
                                markers=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show prediction table
                            st.subheader("Prediction Values")
                            display_df = pred_df.copy()
                            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                            display_df['prediction'] = display_df['prediction'].round(2)
                            st.dataframe(display_df)
                        
                        else:
                            st.error("Failed to generate predictions")
                    
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
    
    elif page == "System Status":
        st.header("System Status")
        
        # Cache statistics
        st.subheader("Cache Statistics")
        cache_stats = cache.get_cache_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", cache_stats['total_files'])
        with col2:
            st.metric("Total Size (MB)", f"{cache_stats['total_size_mb']:.2f}")
        with col3:
            st.metric("Data Files", cache_stats['data_files'])
        with col4:
            st.metric("Model Files", cache_stats['model_files'])
        
        # Cache cleanup
        if st.button("Clean Expired Cache"):
            removed = cache.cleanup_expired()
            st.success(f"Removed {removed} expired cache entries")
        
        # Model information
        st.subheader("Available Models")
        model_info = model_service.get_model_info()
        
        if model_info['available_models']:
            for model_key in model_info['available_models']:
                with st.expander(f"Model: {model_key}"):
                    if model_key in model_info['model_metrics']:
                        metrics = model_info['model_metrics'][model_key]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Performance Metrics**")
                            st.write(f"Test RMSE: {metrics.get('test_rmse', 'N/A'):.4f}")
                            st.write(f"Test R²: {metrics.get('test_r2', 'N/A'):.4f}")
                            st.write(f"Test MAE: {metrics.get('test_mae', 'N/A'):.4f}")
                        
                        with col2:
                            st.write("**Cross-Validation**")
                            st.write(f"CV RMSE (mean): {metrics.get('cv_rmse_mean', 'N/A'):.4f}")
                            st.write(f"CV RMSE (std): {metrics.get('cv_rmse_std', 'N/A'):.4f}")
                        
                        # Feature importance (top 10)
                        if 'feature_importance' in metrics:
                            st.write("**Top 10 Most Important Features**")
                            importance = metrics['feature_importance']
                            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                            
                            importance_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                            fig = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title="Feature Importance"
                            )
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No models available. Please train some models first.")
        
        # System configuration
        st.subheader("System Configuration")
        with st.expander("Configuration Details"):
            st.write("**Ecuador Bounding Box:**")
            st.json(ECUADOR_BBOX)
            
            st.write("**NASA Collections:**")
            st.json(data_service.collections)

if __name__ == "__main__":
    main()
