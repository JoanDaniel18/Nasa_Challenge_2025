"""
Service for fetching and processing NASA HARMONY satellite data
"""
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import requests
from requests.auth import HTTPBasicAuth
import tempfile
import glob

from config import (
    NASA_EDL_USERNAME, NASA_EDL_PASSWORD, ECUADOR_BBOX, 
    NASA_COLLECTIONS, DEFAULT_TIME_RANGE, DATA_DIR
)
from cache_service import cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataService:
    def __init__(self):
        self.username = NASA_EDL_USERNAME
        self.password = NASA_EDL_PASSWORD
        self.bbox = ECUADOR_BBOX
        self.collections = NASA_COLLECTIONS
        self.session = requests.Session()
        
        if self.username and self.password:
            self.session.auth = HTTPBasicAuth(self.username, self.password)
        
    def _generate_cache_key(self, variable: str, start_date: datetime, end_date: datetime) -> str:
        """Generate unique cache key for data request"""
        return f"{variable}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    
    def fetch_harmony_data(
        self, 
        variable: str, 
        start_date: datetime, 
        end_date: datetime,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from NASA HARMONY API
        """
        cache_key = self._generate_cache_key(variable, start_date, end_date)
        
        # Check cache first
        if not force_refresh:
            cached_data = cache.get(cache_key, 'data')
            if cached_data is not None:
                logger.info(f"Using cached data for {variable}")
                return cached_data
        
        try:
            # For demonstration, we'll simulate the HARMONY API request structure
            # In production, you would use harmony-py client library
            
            collection_id = self.collections.get(variable)
            if not collection_id:
                logger.error(f"No collection found for variable: {variable}")
                return None
            
            # Construct HARMONY API URL
            base_url = "https://harmony.earthdata.nasa.gov"
            url = f"{base_url}/{collection_id}/ogc-api-coverages/1.0.0/collections/all/coverage/rangeset"
            
            params = {
                'subset': [
                    f'lat({self.bbox["south"]}:{self.bbox["north"]})',
                    f'lon({self.bbox["west"]}:{self.bbox["east"]})',
                    f'time("{start_date.isoformat()}":"{end_date.isoformat()}")'
                ],
                'format': 'application/x-netcdf4'
            }
            
            logger.info(f"Fetching {variable} data from HARMONY API...")
            
            # Make request to HARMONY
            response = self.session.get(url, params=params, timeout=300)
            
            if response.status_code == 200:
                # Save response to temporary file
                with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
                    temp_file.write(response.content)
                    temp_path = temp_file.name
                
                # Process NetCDF file
                df = self._process_netcdf_file(temp_path, variable)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                if df is not None:
                    # Cache the processed data
                    cache.set(cache_key, df, 'data')
                    logger.info(f"Successfully fetched and cached {variable} data")
                    return df
                
            else:
                logger.error(f"HARMONY API request failed: {response.status_code}")
                # Fallback to simulated data for demonstration
                return self._generate_sample_data(variable, start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error fetching HARMONY data: {str(e)}")
            # Fallback to simulated data
            return self._generate_sample_data(variable, start_date, end_date)
        
        return None
    
    def _process_netcdf_file(self, file_path: str, variable: str) -> Optional[pd.DataFrame]:
        """
        Process NetCDF file and extract relevant data
        """
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                # Extract coordinates
                if 'lat' in dataset.variables:
                    lats = dataset.variables['lat'][:]
                elif 'latitude' in dataset.variables:
                    lats = dataset.variables['latitude'][:]
                else:
                    logger.error("No latitude variable found")
                    return None
                
                if 'lon' in dataset.variables:
                    lons = dataset.variables['lon'][:]
                elif 'longitude' in dataset.variables:
                    lons = dataset.variables['longitude'][:]
                else:
                    logger.error("No longitude variable found")
                    return None
                
                if 'time' in dataset.variables:
                    times = dataset.variables['time']
                    time_units = times.units if hasattr(times, 'units') else 'days since 1970-01-01'
                    dates = nc.num2date(times[:], time_units)
                else:
                    logger.error("No time variable found")
                    return None
                
                # Find the main data variable
                data_vars = [var for var in dataset.variables.keys() 
                           if var not in ['lat', 'lon', 'latitude', 'longitude', 'time']]
                
                if not data_vars:
                    logger.error("No data variables found")
                    return None
                
                # Use the first data variable
                var_name = data_vars[0]
                data = dataset.variables[var_name][:]
                
                # Convert to DataFrame
                records = []
                for t_idx, date in enumerate(dates):
                    for lat_idx, lat in enumerate(lats):
                        for lon_idx, lon in enumerate(lons):
                            value = data[t_idx, lat_idx, lon_idx] if data.ndim == 3 else data[lat_idx, lon_idx]
                            
                            # Skip masked/invalid values
                            if np.ma.is_masked(value) or np.isnan(value):
                                continue
                            
                            records.append({
                                'date': pd.to_datetime(date),
                                'latitude': float(lat),
                                'longitude': float(lon),
                                'value': float(value),
                                'variable': variable
                            })
                
                if records:
                    df = pd.DataFrame(records)
                    df = df.sort_values(['date', 'latitude', 'longitude'])
                    return df
                else:
                    logger.warning("No valid data points found")
                    return None
                    
        except Exception as e:
            logger.error(f"Error processing NetCDF file: {str(e)}")
            return None
    
    def _generate_sample_data(
        self, 
        variable: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate sample data for demonstration when API is not available
        Note: This is only for development/testing purposes
        """
        logger.warning(f"Generating sample data for {variable} - this should only be used for testing")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Define Ecuador grid points (simplified)
        lat_points = np.linspace(self.bbox['south'], self.bbox['north'], 10)
        lon_points = np.linspace(self.bbox['west'], self.bbox['east'], 10)
        
        records = []
        
        # Variable-specific data generation
        base_values = {
            'temperature': 25.0,  # Celsius
            'precipitation': 2.5,  # mm
            'humidity': 70.0,     # %
            'pressure': 1013.25   # hPa
        }
        
        noise_levels = {
            'temperature': 8.0,
            'precipitation': 3.0,
            'humidity': 20.0,
            'pressure': 15.0
        }
        
        base_val = base_values.get(variable, 0.0)
        noise_level = noise_levels.get(variable, 1.0)
        
        for date in date_range:
            for lat in lat_points:
                for lon in lon_points:
                    # Add some realistic variation
                    seasonal_factor = np.sin(2 * np.pi * date.day_of_year / 365.25)
                    altitude_factor = (lat - self.bbox['south']) * 0.1  # Simple elevation proxy
                    random_noise = np.random.normal(0, noise_level * 0.3)
                    
                    value = base_val + seasonal_factor * noise_level * 0.5 + altitude_factor + random_noise
                    
                    # Ensure realistic bounds
                    if variable == 'humidity':
                        value = np.clip(value, 0, 100)
                    elif variable == 'precipitation':
                        value = np.maximum(value, 0)
                    
                    records.append({
                        'date': date,
                        'latitude': lat,
                        'longitude': lon,
                        'value': value,
                        'variable': variable
                    })
        
        df = pd.DataFrame(records)
        
        # Cache the sample data
        cache_key = self._generate_cache_key(variable, start_date, end_date)
        cache.set(cache_key, df, 'data')
        
        return df
    
    def get_multi_variable_data(
        self, 
        variables: List[str], 
        start_date: datetime, 
        end_date: datetime,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple variables
        """
        results = {}
        
        for variable in variables:
            logger.info(f"Fetching data for variable: {variable}")
            data = self.fetch_harmony_data(variable, start_date, end_date, force_refresh)
            if data is not None:
                results[variable] = data
            else:
                logger.warning(f"No data retrieved for variable: {variable}")
        
        return results
    
    def aggregate_spatial_data(self, df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """
        Aggregate spatial data to time series
        """
        if method == 'mean':
            agg_data = df.groupby(['date', 'variable'])['value'].mean().reset_index()
        elif method == 'max':
            agg_data = df.groupby(['date', 'variable'])['value'].max().reset_index()
        elif method == 'min':
            agg_data = df.groupby(['date', 'variable'])['value'].min().reset_index()
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        return agg_data
    
    def prepare_ml_dataset(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        target_variable: str = 'temperature'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for machine learning training
        """
        if start_date is None:
            start_date = DEFAULT_TIME_RANGE['start']
        if end_date is None:
            end_date = DEFAULT_TIME_RANGE['end']
        
        # Fetch all climate variables
        variables = list(self.collections.keys())
        data_dict = self.get_multi_variable_data(variables, start_date, end_date)
        
        if not data_dict:
            logger.error("No data available for ML dataset preparation")
            return pd.DataFrame(), pd.Series()
        
        # Aggregate spatial data
        aggregated_dfs = []
        for var, df in data_dict.items():
            agg_df = self.aggregate_spatial_data(df)
            agg_df['variable'] = var
            aggregated_dfs.append(agg_df)
        
        # Combine all variables
        combined_df = pd.concat(aggregated_dfs, ignore_index=True)
        
        # Pivot to have variables as columns
        pivot_df = combined_df.pivot(index='date', columns='variable', values='value')
        
        # Forward fill missing values
        pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')
        
        # Create features (lagged values, moving averages, etc.)
        feature_df = pivot_df.copy()
        
        # Add lagged features
        for var in pivot_df.columns:
            for lag in [1, 7, 30]:  # 1 day, 1 week, 1 month lags
                feature_df[f'{var}_lag_{lag}'] = pivot_df[var].shift(lag)
        
        # Add moving averages
        for var in pivot_df.columns:
            feature_df[f'{var}_ma_7'] = pivot_df[var].rolling(window=7).mean()
            feature_df[f'{var}_ma_30'] = pivot_df[var].rolling(window=30).mean()
        
        # Add time-based features
        feature_df['day_of_year'] = feature_df.index.day_of_year
        feature_df['month'] = feature_df.index.month
        feature_df['season'] = ((feature_df.index.month - 1) // 3) + 1
        
        # Drop rows with NaN values
        feature_df = feature_df.dropna()
        
        if feature_df.empty:
            logger.error("No valid data remaining after feature engineering")
            return pd.DataFrame(), pd.Series()
        
        # Separate features and target
        target = feature_df[target_variable]
        features = feature_df.drop(columns=[target_variable])
        
        logger.info(f"Prepared ML dataset with {len(features)} samples and {len(features.columns)} features")
        
        return features, target

# Global data service instance
data_service = DataService()
