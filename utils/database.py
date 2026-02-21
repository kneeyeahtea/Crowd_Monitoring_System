import sqlite3
from datetime import datetime
import json
import numpy as np


class CrowdDatabase:
    def __init__(self, db_name='crowd_monitoring.db'):
        """Initialize database connection and create tables"""
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()
    
    def create_tables(self):
        """Create tables if they don't exist"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS crowd_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                location TEXT NOT NULL,
                crowd_count INTEGER NOT NULL,
                alert_status TEXT NOT NULL,
                flow_direction REAL,
                flow_magnitude REAL,
                num_hotspots INTEGER,
                hotspot_info TEXT
            )
        ''')
        self.conn.commit()
        print("Database tables created/verified successfully!")
    
    def insert_record(self, location, crowd_count, alert_status,
                     flow_direction=None, flow_magnitude=None,
                     num_hotspots=0, hotspot_info=None):
        """Insert a new crowd monitoring record"""
        
        # Convert hotspot_info to JSON string - handle numpy arrays and numpy types
        if hotspot_info is not None:
            # Convert any numpy arrays/types to native Python types
            hotspot_info = self._convert_to_serializable(hotspot_info)
            
            if not isinstance(hotspot_info, str):
                hotspot_info = json.dumps(hotspot_info)
        
        # Convert flow_direction and flow_magnitude to native Python float if they're numpy types
        if flow_direction is not None and isinstance(flow_direction, np.floating):
            flow_direction = float(flow_direction)
        if flow_magnitude is not None and isinstance(flow_magnitude, np.floating):
            flow_magnitude = float(flow_magnitude)
        
        self.cursor.execute('''
            INSERT INTO crowd_data 
            (location, crowd_count, alert_status, flow_direction, 
             flow_magnitude, num_hotspots, hotspot_info)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (location, crowd_count, alert_status, flow_direction,
              flow_magnitude, num_hotspots, hotspot_info))
        
        self.conn.commit()
    
    def _convert_to_serializable(self, obj):
        """Recursively convert numpy arrays and numpy types to native Python types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    def get_all_records(self, location=None):
        """Retrieve all records, optionally filtered by location"""
        if location:
            self.cursor.execute('''
                SELECT * FROM crowd_data WHERE location = ?
                ORDER BY timestamp DESC
            ''', (location,))
        else:
            self.cursor.execute('''
                SELECT * FROM crowd_data ORDER BY timestamp DESC
            ''')
        
        return self.cursor.fetchall()
    
    def get_statistics(self, location):
        """Get statistics for a specific location"""
        self.cursor.execute('''
            SELECT 
                AVG(crowd_count) as avg_count,
                MAX(crowd_count) as max_count,
                MIN(crowd_count) as min_count,
                COUNT(*) as record_count
            FROM crowd_data
            WHERE location = ?
        ''', (location,))
        
        return self.cursor.fetchone()
    
    def get_recent_records(self, location, limit=10):
        """Get most recent records for a location"""
        self.cursor.execute('''
            SELECT * FROM crowd_data 
            WHERE location = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (location, limit))
        
        return self.cursor.fetchall()
    
    def delete_location_data(self, location):
        """Delete all data for a specific location"""
        self.cursor.execute('''
            DELETE FROM crowd_data WHERE location = ?
        ''', (location,))
        self.conn.commit()
        print(f"Deleted all records for location: {location}")
    
    def get_alert_history(self, location):
        """Get all alert events for a location"""
        self.cursor.execute('''
            SELECT timestamp, crowd_count, num_hotspots
            FROM crowd_data 
            WHERE location = ? AND alert_status = 'Alert'
            ORDER BY timestamp DESC
        ''', (location,))
        
        return self.cursor.fetchall()
    
    def close(self):
        """Close database connection"""
        self.conn.close()
