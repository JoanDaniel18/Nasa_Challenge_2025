"""
Cache service for storing processed data and trained models
"""
import os
import pickle
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import logging
from config import CACHE_DIR, CACHE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.retention_days = CACHE_CONFIG['data_retention_days']
        self.model_retention_days = CACHE_CONFIG['model_retention_days']
        
    def _get_cache_path(self, key: str, cache_type: str = 'data') -> str:
        """Generate cache file path"""
        return os.path.join(self.cache_dir, f"{cache_type}_{key}.pkl")
    
    def _get_metadata_path(self, key: str, cache_type: str = 'data') -> str:
        """Generate metadata file path"""
        return os.path.join(self.cache_dir, f"{cache_type}_{key}_meta.json")
    
    def set(self, key: str, data: Any, cache_type: str = 'data') -> bool:
        """Store data in cache with metadata"""
        try:
            cache_path = self._get_cache_path(key, cache_type)
            meta_path = self._get_metadata_path(key, cache_type)
            
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'key': key,
                'cache_type': cache_type,
                'size_mb': os.path.getsize(cache_path) / (1024 * 1024)
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Cached {cache_type} data: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching data {key}: {str(e)}")
            return False
    
    def get(self, key: str, cache_type: str = 'data') -> Optional[Any]:
        """Retrieve data from cache if not expired"""
        try:
            cache_path = self._get_cache_path(key, cache_type)
            meta_path = self._get_metadata_path(key, cache_type)
            
            if not os.path.exists(cache_path) or not os.path.exists(meta_path):
                return None
            
            # Check if cache is expired
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            created_at = datetime.fromisoformat(metadata['created_at'])
            retention_days = self.model_retention_days if cache_type == 'model' else self.retention_days
            
            if datetime.now() - created_at > timedelta(days=retention_days):
                logger.info(f"Cache expired for {key}")
                self.delete(key, cache_type)
                return None
            
            # Load and return data
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Retrieved cached {cache_type} data: {key}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving cached data {key}: {str(e)}")
            return None
    
    def delete(self, key: str, cache_type: str = 'data') -> bool:
        """Delete cached data and metadata"""
        try:
            cache_path = self._get_cache_path(key, cache_type)
            meta_path = self._get_metadata_path(key, cache_type)
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            logger.info(f"Deleted cached {cache_type} data: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cached data {key}: {str(e)}")
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files"""
        removed_count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('_meta.json'):
                    meta_path = os.path.join(self.cache_dir, filename)
                    
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    created_at = datetime.fromisoformat(metadata['created_at'])
                    cache_type = metadata.get('cache_type', 'data')
                    retention_days = self.model_retention_days if cache_type == 'model' else self.retention_days
                    
                    if datetime.now() - created_at > timedelta(days=retention_days):
                        key = metadata['key']
                        if self.delete(key, cache_type):
                            removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} expired cache entries")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'data_files': 0,
            'model_files': 0,
            'oldest_entry': None,
            'newest_entry': None
        }
        
        try:
            entries = []
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('_meta.json'):
                    meta_path = os.path.join(self.cache_dir, filename)
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    entries.append(metadata)
            
            if entries:
                stats['total_files'] = len(entries)
                stats['total_size_mb'] = sum(entry.get('size_mb', 0) for entry in entries)
                stats['data_files'] = len([e for e in entries if e.get('cache_type') == 'data'])
                stats['model_files'] = len([e for e in entries if e.get('cache_type') == 'model'])
                
                dates = [datetime.fromisoformat(e['created_at']) for e in entries]
                stats['oldest_entry'] = min(dates).isoformat()
                stats['newest_entry'] = max(dates).isoformat()
        
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
        
        return stats

# Global cache instance
cache = CacheService()
