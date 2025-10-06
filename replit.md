# Ecuador Climate Prediction System

## Overview

This is a climate prediction system focused on Ecuador that leverages NASA satellite data (via the HARMONY API) to forecast weather patterns. The system provides both a REST API and an interactive Streamlit dashboard for visualizing and predicting climate variables including temperature, precipitation, humidity, and pressure. It uses machine learning models (XGBoost and Random Forest) to generate predictions based on historical satellite data.

The application includes a complete data pipeline: fetching data from NASA's Earth observation satellites, processing and caching the data, training ML models, and serving predictions through both API and web interfaces.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Streamlit Dashboard (`app.py`)**
- Interactive web interface for climate visualization and predictions
- Plotly-based charts for time series analysis and geographic visualization
- Real-time data loading with caching to improve performance
- Responsive layout with custom CSS styling
- Demo application (`weather_demo_app.py`) for simplified weather classification (Sunny vs Cloudy)

**Design Decision**: Streamlit was chosen for rapid development of an interactive dashboard without requiring extensive frontend framework knowledge. The built-in caching and component system allows for efficient data visualization.

### Backend Architecture

**FastAPI REST API (`api.py`)**
- RESTful endpoints for climate predictions
- Pydantic models for request/response validation
- CORS middleware for cross-origin support
- Background task support for async operations
- Health check endpoints with system status

**Service Layer Pattern**:
1. **DataService** (`data_service.py`) - Handles NASA HARMONY API integration and data fetching
2. **ModelService** (`model_service.py`) - Manages ML model training and predictions
3. **CacheService** (`cache_service.py`) - Provides file-based caching for data and models

**Design Decision**: The service layer pattern separates concerns, making each component independently testable and maintainable. FastAPI provides automatic API documentation and async support for better performance.

### Data Processing Pipeline

**NASA HARMONY Integration**:
- Authenticates using NASA Earthdata Login credentials (environment variables)
- Fetches satellite data for Ecuador's bounding box (coordinates: -81°W to -75°W, -5°S to 5°N)
- Supports multiple climate variables through different NASA collections:
  - Precipitation: GPM IMERG (C1598621096-GES_DISC)
  - Temperature/Humidity/Pressure: AIRS (C1276812863-GES_DISC)
- Processes NetCDF4 format satellite data
- Implements retry logic and error handling for API requests

**Data Flow**:
1. Request data from NASA HARMONY API with spatial and temporal parameters
2. Download and parse NetCDF4 files
3. Extract relevant variables and convert to pandas DataFrames
4. Cache processed data for future use
5. Serve to ML models or API consumers

**Design Decision**: Direct integration with NASA's official data sources ensures high-quality, authoritative climate data. NetCDF4 is the standard format for scientific climate data.

### Machine Learning Architecture

**Model Types**:
- **XGBoost Regressor**: Primary model for regression tasks (temperature, precipitation prediction)
- **Random Forest Regressor**: Alternative model with ~88% accuracy for classification tasks

**Training Pipeline**:
- 80/20 train-test split (configurable)
- Cross-validation for model evaluation
- Metrics tracking: RMSE, R², MAE
- Model persistence using joblib

**Feature Engineering**:
- Time-based features (hour, day, month, year)
- Spatial features (latitude, longitude)
- Historical values and trends
- Derived meteorological features

**Design Decision**: XGBoost provides excellent performance for time series prediction with structured data. Random Forest offers a robust baseline. The service supports multiple model types to allow comparison and fallback options.

### Caching Strategy

**File-Based Cache System**:
- Pickle serialization for Python objects (DataFrames, models)
- JSON metadata for cache management
- Configurable retention periods:
  - Data cache: 7 days default
  - Model cache: 30 days default
- Automatic cache invalidation based on age
- Memory limit controls (500MB default)

**Cache Keys**: Generated from variable name, date range, and model type to ensure uniqueness

**Design Decision**: File-based caching is simple, persistent across restarts, and doesn't require external dependencies. For larger scale deployments, this could be upgraded to Redis or similar.

### Configuration Management

**Centralized Config** (`config.py`):
- Environment-based credentials (NASA_EDL_USERNAME, NASA_EDL_PASSWORD)
- Bounding box coordinates for Ecuador
- NASA collection IDs for different climate variables
- Model hyperparameters (n_estimators, max_depth, learning_rate)
- Cache settings
- API settings (host, port)

**Design Decision**: Centralized configuration makes it easy to adjust system behavior without code changes. Environment variables keep sensitive credentials secure.

## External Dependencies

### Third-Party APIs

**NASA Earthdata HARMONY API**:
- Purpose: Primary data source for satellite climate observations
- Authentication: HTTP Basic Auth with NASA Earthdata Login credentials
- Endpoints: https://harmony.earthdata.nasa.gov/
- Data Format: NetCDF4
- Collections Used:
  - GPM IMERG for precipitation data
  - AIRS for temperature, humidity, and pressure

### Python Libraries

**Data Processing**:
- `pandas`: DataFrame manipulation and time series handling
- `numpy`: Numerical computations
- `netCDF4`: Parsing NASA satellite data files
- `requests`: HTTP client for API calls

**Machine Learning**:
- `scikit-learn`: Model training, evaluation, and utilities
- `xgboost`: Gradient boosting models for predictions
- `joblib`: Model serialization

**Web Frameworks**:
- `fastapi`: REST API framework
- `uvicorn`: ASGI server for FastAPI
- `streamlit`: Dashboard web application
- `pydantic`: Data validation and settings management

**Visualization**:
- `plotly`: Interactive charts and graphs

**Development**:
- `python-dotenv`: Environment variable management (likely)

### File Storage

**Local Filesystem**:
- Cache directory for processed data and models
- Model persistence directory
- Temporary files for downloaded NetCDF4 data

**Design Decision**: Currently uses local filesystem for simplicity. For production, could integrate cloud storage (S3, GCS) for scalability.

### Future Integration Considerations

- **Database**: System currently uses file-based storage. A PostgreSQL database could be added for:
  - Historical prediction storage
  - User management
  - Query optimization
  - Relationship management between predictions and source data

- **Message Queue**: For handling long-running model training tasks asynchronously (Celery, RabbitMQ)

- **Monitoring**: Integration with monitoring services (DataDog, Prometheus) for production deployments