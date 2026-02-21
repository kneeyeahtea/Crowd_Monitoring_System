import numpy as np
from sklearn.cluster import DBSCAN
import cv2

class CrowdClusterer:
    def __init__(self, eps=30, min_samples=5):
        """
        Initialize DBSCAN clusterer for hotspot detection
        
        Args:
            eps: Maximum distance between points in a cluster
            min_samples: Minimum points to form a dense region
        """
        self.eps = eps
        self.min_samples = min_samples
    
    def extract_peaks(self, density_map, threshold_percentile=90):
        """
        Extract peak coordinates from density map
        
        Args:
            density_map: Density map from CSRNet
            threshold_percentile: Percentile threshold for peak detection
            
        Returns:
            peaks: Array of (x, y) coordinates
        """
        # Calculate threshold based on percentile
        threshold = np.percentile(density_map, threshold_percentile)
        
        # Find coordinates above threshold
        peak_mask = density_map > threshold
        y_coords, x_coords = np.where(peak_mask)
        
        # Combine into (x, y) pairs
        if len(x_coords) > 0:
            peaks = np.column_stack((x_coords, y_coords))
        else:
            peaks = np.array([])
        
        return peaks
    
    def cluster_hotspots(self, peaks):
        """
        Apply DBSCAN clustering to detect hotspots
        
        Args:
            peaks: Array of peak coordinates
            
        Returns:
            labels: Cluster labels for each point
            n_clusters: Number of clusters found
            cluster_centers: Centers of each cluster
        """
        if len(peaks) == 0:
            return np.array([]), 0, np.array([])
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(peaks)
        
        # Count clusters (excluding noise labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calculate cluster centers
        cluster_centers = []
        for cluster_id in range(n_clusters):
            cluster_points = peaks[labels == cluster_id]
            center = cluster_points.mean(axis=0)
            cluster_centers.append(center)
        
        cluster_centers = np.array(cluster_centers)
        
        return labels, n_clusters, cluster_centers
    
    def visualize_clusters(self, frame, peaks, labels, cluster_centers):
        """
        Visualize clusters on frame
        
        Args:
            frame: Original frame
            peaks: Peak coordinates
            labels: Cluster labels
            cluster_centers: Cluster center coordinates
            
        Returns:
            vis_frame: Frame with cluster visualization
        """
        vis_frame = frame.copy()
        
        if len(peaks) == 0:
            return vis_frame
        
        # Define colors for different clusters
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        # Draw points colored by cluster
        for point, label in zip(peaks, labels):
            x, y = int(point[0]), int(point[1])
            
            if label == -1:
                # Noise points in gray
                color = (128, 128, 128)
            else:
                # Cluster points
                color = colors[label % len(colors)]
            
            cv2.circle(vis_frame, (x, y), 3, color, -1)
        
        # Draw cluster centers
        for idx, center in enumerate(cluster_centers):
            x, y = int(center[0]), int(center[1])
            color = colors[idx % len(colors)]
            
            # Draw large circle for center
            cv2.circle(vis_frame, (x, y), 15, color, 3)
            cv2.putText(vis_frame, f"H{idx+1}", (x-10, y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_frame
    
    def get_hotspot_info(self, labels, peaks, density_map):
        """
        Get detailed information about each hotspot
        
        Args:
            labels: Cluster labels
            peaks: Peak coordinates
            density_map: Density map
            
        Returns:
            hotspot_info: List of dictionaries with hotspot details
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        hotspot_info = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_points = peaks[cluster_mask]
            
            # Calculate center
            center = cluster_points.mean(axis=0)
            
            # Estimate density at cluster
            x, y = int(center[0]), int(center[1])
            if 0 <= y < density_map.shape[0] and 0 <= x < density_map.shape[1]:
                avg_density = density_map[y, x]
            else:
                avg_density = 0
            
            hotspot_info.append({
                'id': cluster_id + 1,
                'center': center,
                'num_points': len(cluster_points),
                'density': avg_density
            })
        
        return hotspot_info
