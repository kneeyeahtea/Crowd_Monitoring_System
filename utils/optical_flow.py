import cv2
import numpy as np

class OpticalFlowTracker:
    def __init__(self, method='farneback'):
        """
        Initialize optical flow tracker
        
        Args:
            method: 'farneback' or 'lucas-kanade'
        """
        self.method = method
        self.prev_gray = None
        
        # Farneback parameters
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def compute_flow(self, frame):
        """
        Compute optical flow between consecutive frames
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            flow: Optical flow field (dx, dy)
            magnitude: Flow magnitude
            angle: Flow direction in degrees
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If first frame, store and return None
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None, None
        
        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            **self.farneback_params
        )
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Convert angle to degrees
        angle_deg = angle * 180 / np.pi
        
        # Update previous frame
        self.prev_gray = gray
        
        return flow, magnitude, angle_deg
    
    def visualize_flow(self, frame, flow, magnitude, angle):
        """
        Create visualization of optical flow
        
        Args:
            frame: Original frame
            flow: Optical flow field
            magnitude: Flow magnitude
            angle: Flow direction in degrees
            
        Returns:
            flow_vis: Colored visualization of flow
        """
        if flow is None:
            return frame
        
        # Create HSV image for visualization
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255  # Full saturation
        
        # Hue represents direction, Value represents magnitude
        hsv[..., 0] = angle / 2  # Scale to 0-180
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV to BGR
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_vis
    
    def get_average_direction(self, flow, magnitude, threshold=1.0):
        """
        Calculate average flow direction in the scene
        
        Args:
            flow: Optical flow field
            magnitude: Flow magnitude
            threshold: Minimum magnitude to consider
            
        Returns:
            avg_direction: Average direction in degrees
            avg_magnitude: Average magnitude
        """
        if flow is None:
            return None, None
        
        # Filter by magnitude threshold
        mask = magnitude > threshold
        
        if not np.any(mask):
            return 0, 0
        
        # Get valid flow vectors
        valid_flow = flow[mask]
        valid_mag = magnitude[mask]
        
        # Calculate weighted average
        avg_dx = np.average(valid_flow[:, 0], weights=valid_mag)
        avg_dy = np.average(valid_flow[:, 1], weights=valid_mag)
        avg_magnitude = np.mean(valid_mag)
        
        # Calculate direction
        avg_direction = np.arctan2(avg_dy, avg_dx) * 180 / np.pi
        
        return avg_direction, avg_magnitude
    
    def reset(self):
        """Reset the tracker (clear previous frame)"""
        self.prev_gray = None
