"""
FastAPI REST API for climate predictions
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import uvicorn

from config import API_CONFIG
from model_service import model_service
from data_service import data_service
from cache_service import cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    target_variable: str = 'temperature'
    days_ahead: int = 7
    model_type: Optional[str] = None  # 'xgboost' or 'random_forest'

class PredictionResponse(BaseModel):
    dates: List[str]
    predictions: List[float]
    target_variable: str
    model_type: str
    confidence_interval: Optional[Dict[str, List[float]]] = None

class TrainingRequest(BaseModel):
    target_variable: str = 'temperature'
    force_retrain: bool = False

class ModelMetrics(BaseModel):
    test_rmse: float
    test_r2: float
    test_mae: float
    cv_rmse_mean: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    available_models: List[str]
    cache_stats: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG['title'],
    version=API_CONFIG['version'],
    description="API for Ecuador climate prediction using NASA HARMONY satellite data"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_info = model_service.get_model_info()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            available_models=model_info['available_models'],
            cache_stats=model_info['cache_stats']
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_climate(request: PredictionRequest):
    """Generate climate predictions"""
    try:
        # Determine which model to use
        if request.model_type:
            model_key = f"{request.model_type}_{request.target_variable}_model"
        else:
            # Use best available model
            comparison = model_service.compare_models(request.target_variable)
            if not comparison or 'best_model' not in comparison:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No trained models found for {request.target_variable}"
                )
            best_model = comparison['best_model']
            model_key = f"{best_model}_{request.target_variable}_model"
        
        # Generate predictions
        result = model_service.predict_future(
            model_key=model_key,
            days_ahead=request.days_ahead,
            target_variable=request.target_variable
        )
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate predictions"
            )
        
        return PredictionResponse(
            dates=result['dates'],
            predictions=result['predictions'],
            target_variable=result['target_variable'],
            model_type=model_key.split('_')[0]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train or retrain models"""
    try:
        # Check if models exist and are recent
        model_info = model_service.get_model_info()
        needs_training = request.force_retrain or not any(
            request.target_variable in key for key in model_info['available_models']
        )
        
        if needs_training:
            # Run training in background
            background_tasks.add_task(
                model_service.retrain_models, 
                request.target_variable
            )
            
            return {
                "message": f"Training initiated for {request.target_variable}",
                "status": "training_started"
            }
        else:
            return {
                "message": f"Models already exist for {request.target_variable}",
                "status": "models_exist",
                "available_models": [
                    key for key in model_info['available_models']
                    if request.target_variable in key
                ]
            }
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{target_variable}")
async def get_model_info(target_variable: str):
    """Get information about models for a specific variable"""
    try:
        comparison = model_service.compare_models(target_variable)
        
        if not comparison:
            raise HTTPException(
                status_code=404,
                detail=f"No models found for {target_variable}"
            )
        
        return {
            "target_variable": target_variable,
            "models": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/{variable}")
async def get_historical_data(
    variable: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    aggregation: str = 'daily'
):
    """Get historical climate data"""
    try:
        # Parse dates
        if start_date:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        else:
            start = datetime.now() - timedelta(days=30)
        
        if end_date:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        else:
            end = datetime.now()
        
        # Fetch data
        data = data_service.fetch_harmony_data(variable, start, end)
        
        if data is None or data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {variable}"
            )
        
        # Aggregate if needed
        if aggregation == 'daily':
            aggregated = data.groupby([data['date'].dt.date])['value'].mean().reset_index()
            aggregated['date'] = aggregated['date'].astype(str)
        else:
            aggregated = data[['date', 'value']].copy()
            aggregated['date'] = aggregated['date'].astype(str)
        
        return {
            "variable": variable,
            "data": aggregated.to_dict('records'),
            "count": len(aggregated),
            "date_range": {
                "start": start.isoformat(),
                "end": end.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache")
async def clear_cache():
    """Clear system cache"""
    try:
        removed_count = cache.cleanup_expired()
        
        return {
            "message": "Cache cleanup completed",
            "removed_entries": removed_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/variables")
async def get_available_variables():
    """Get list of available climate variables"""
    return {
        "variables": list(data_service.collections.keys()),
        "description": {
            "temperature": "Surface temperature (°C)",
            "precipitation": "Precipitation rate (mm)",
            "humidity": "Relative humidity (%)",
            "pressure": "Surface pressure (hPa)"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=True
    )
