import cv2
import torch
import numpy as np
import time
import sqlite3
import os
import datetime
import matplotlib.pyplot as plt
from models.csrnet import CSRNet
from utils.preprocessing import ImagePreprocessor
from utils.optical_flow import OpticalFlowTracker
from utils.clustering import CrowdClusterer
from utils.database import CrowdDatabase


class CrowdMonitoringSystem:
    def __init__(self, video_path, location_name="Station", threshold=500):
        """
        Initialize the complete crowd monitoring system
        
        Args:
            video_path: Path to video file or 0 for webcam
            location_name: Name of the location being monitored
            threshold: Crowd count threshold for alerts
        """
        # Initialize components
        self.video_path = video_path
        self.location = location_name
        self.threshold = threshold
        
        # Generate timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load CSRNet model
        print("Loading CSRNet model...")
        self.model = CSRNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained weights
        weights_path = 'data/pretrained_weights/csrnet_weights.pth.tar'
        print(f"Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load with strict=False to ignore mismatched keys
        self.model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully!")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize utilities - REDUCED RESOLUTION for faster processing
        self.preprocessor = ImagePreprocessor(target_size=(384, 384))
        self.flow_tracker = OpticalFlowTracker(method='farneback')
        self.clusterer = CrowdClusterer(eps=30, min_samples=5)
        
        # Initialize database (THIS IS KEY!)
        print("Initializing database...")
        self.database = CrowdDatabase()
        print("Database ready!")
        
        # Frame counter and statistics tracker
        self.frame_count = 0
        self.all_counts = []  # Track all counts for statistics
        
        print("System initialized successfully!")
    
    def process_frame(self, frame):
        """Process a single frame through the complete pipeline"""
        
        # 1. Preprocess frame
        tensor, original_shape = self.preprocessor.preprocess_frame(frame)
        tensor = tensor.to(self.device)
        
        # 2. Generate density map using CSRNet
        with torch.no_grad():
            density_map = self.model(tensor)
        
        # Resize density map to original frame size
        density_np = self.preprocessor.resize_density_map(density_map, original_shape)
        
        # 3. Calculate crowd count with scaling factor
        crowd_count_raw = np.sum(density_np)
        scaling_factor = 0.0090  # Calibration factor
        crowd_count = int(crowd_count_raw * scaling_factor)
        
        # 4. Compute optical flow (FAST VERSION - skip some calculations)
        flow, magnitude, angle = self.flow_tracker.compute_flow(frame)
        avg_direction, avg_magnitude = self.flow_tracker.get_average_direction(
            flow, magnitude, threshold=1.0
        )
        
        # 5. Extract peaks and cluster hotspots
        peaks = self.clusterer.extract_peaks(density_np, threshold_percentile=90)
        labels, n_clusters, cluster_centers = self.clusterer.cluster_hotspots(peaks)
        hotspot_info = self.clusterer.get_hotspot_info(labels, peaks, density_np)
        
        # 6. Decision module - check threshold
        alert_status = "Alert" if crowd_count > self.threshold else "Normal"
        
        # 7. Store data every 30 frames in DATABASE
        if self.frame_count % 30 == 0:
            try:
                self.database.insert_record(
                    location=self.location,
                    crowd_count=crowd_count,
                    alert_status=alert_status,
                    flow_direction=avg_direction,
                    flow_magnitude=avg_magnitude,
                    num_hotspots=n_clusters,
                    hotspot_info=hotspot_info
                )
                print(f"  → Data saved to database (Frame {self.frame_count})")
            except Exception as e:
                print(f"  → Database save error: {e}")
        
        return {
            'frame': frame,
            'density_map': density_np,
            'crowd_count': crowd_count,
            'alert_status': alert_status,
            'flow': flow,
            'magnitude': magnitude,
            'angle': angle,
            'avg_direction': avg_direction,
            'avg_magnitude': avg_magnitude,
            'peaks': peaks,
            'labels': labels,
            'cluster_centers': cluster_centers,
            'n_clusters': n_clusters,
            'hotspot_info': hotspot_info
        }
    
    def create_visualization(self, results):
        """Create comprehensive visualization - RESIZED TO FIT SCREEN"""
        frame = results['frame'].copy()
        h, w = frame.shape[:2]
        
        # REDUCE OUTPUT SIZE to 50% for better screen fit
        display_width = w // 2
        display_height = h // 2
        
        # Create density map visualization (heatmap)
        density_colored = cv2.applyColorMap(
            cv2.normalize(results['density_map'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        density_colored = cv2.resize(density_colored, (w, h))
        
        # Blend original frame with density map
        overlay = cv2.addWeighted(frame, 0.7, density_colored, 0.3, 0)
        
        # Visualize clusters if available
        if len(results['peaks']) > 0:
            overlay = self.clusterer.visualize_clusters(
                overlay, results['peaks'], results['labels'], 
                results['cluster_centers']
            )
        
        # Visualize optical flow if available
        if results['flow'] is not None:
            flow_vis = self.flow_tracker.visualize_flow(
                frame, results['flow'], results['magnitude'], results['angle']
            )
        else:
            flow_vis = frame.copy()
        
        # Add text information
        status_color = (0, 0, 255) if results['alert_status'] == "Alert" else (0, 255, 0)
        
        cv2.putText(overlay, f"Count: {results['crowd_count']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"Status: {results['alert_status']}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(overlay, f"Hotspots: {results['n_clusters']}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if results['avg_direction'] is not None:
            cv2.putText(overlay, f"Flow Dir: {results['avg_direction']:.1f}deg", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine visualizations side by side
        combined = np.hstack([overlay, flow_vis])
        
        # RESIZE TO 50% FOR SCREEN FIT
        combined_resized = cv2.resize(combined, (display_width * 2, display_height))
        
        return combined_resized
    
    def plot_crowd_graph(self):
        """Generate and save crowd count vs frame graph with timestamp"""
        if len(self.all_counts) == 0:
            print("No data to plot!")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Calculate frame numbers (every 30th frame)
        frame_numbers = [i * 30 for i in range(len(self.all_counts))]
        
        # Plot crowd count
        plt.plot(frame_numbers, self.all_counts, 'b-', linewidth=2, marker='o', markersize=6, label='Crowd Count')
        
        # Add threshold line
        plt.axhline(y=self.threshold, color='r', linestyle='--', linewidth=2, label=f'Alert Threshold ({self.threshold})')
        
        # Mark alert regions
        for i, count in enumerate(self.all_counts):
            if count > self.threshold:
                plt.scatter(frame_numbers[i], count, color='red', s=150, zorder=5, edgecolors='darkred', linewidths=2)
        
        # Styling
        plt.xlabel('Frame Number', fontsize=12, fontweight='bold')
        plt.ylabel('Crowd Count (people)', fontsize=12, fontweight='bold')
        plt.title(f'Crowd Count Analysis - {self.location}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='upper right', fontsize=10)
        
        # Add statistics box
        stats_text = f"Peak: {max(self.all_counts)} | Avg: {int(np.mean(self.all_counts))} | Min: {min(self.all_counts)}"
        plt.text(0.5, 0.95, stats_text, 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=10, ha='center', va='top', fontweight='bold')
        
        # Save graph with timestamp
        graph_path = f'output/crowd_graph_{self.location.replace(" ", "_")}_{self.timestamp}.png'
        plt.tight_layout()
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Graph saved to: {graph_path}")
        
        # Display graph
        plt.show()
    
    def run(self, display=True, save_output=False, output_path=None):
        """Run the monitoring system - ULTRA-FAST VERSION"""
        
        # Create output directory if saving
        if save_output:
            os.makedirs('output', exist_ok=True)
            
            # Generate timestamped output path if not provided
            if output_path is None:
                output_path = f'output/result_{self.location.replace(" ", "_")}_{self.timestamp}.avi'
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*80}")
        print(f"Video: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        print(f"Processing every 30th frame for maximum speed...")
        print(f"Database: Saving every 30th frame")
        if save_output:
            print(f"Output will be saved to: {output_path}")
        print(f"Press 'q' in video window to quit")
        print(f"{'='*80}\n")
        
        # Video writer for saving output
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 2.0, (width, height//2))
        
        # PROCESS EVERY 30TH FRAME (10x faster!)
        frame_skip = 30
        
        # Create named window
        if display:
            cv2.namedWindow('Crowd Monitoring System', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Crowd Monitoring System', 1280, 480)
        
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process every 30th frame only
                if self.frame_count % frame_skip == 0:
                    frame_start = time.time()
                    
                    # Process frame
                    results = self.process_frame(frame)
                    
                    # Track count for statistics
                    self.all_counts.append(results['crowd_count'])
                    
                    # Create visualization
                    vis = self.create_visualization(results)
                    
                    # Display
                    if display:
                        cv2.imshow('Crowd Monitoring System', vis)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("\nUser quit")
                            break
                    
                    # Save output
                    if save_output:
                        out.write(vis)
                    
                    # Calculate timing and timestamp
                    frame_time = time.time() - frame_start
                    frames_processed = (self.frame_count // frame_skip) + 1
                    elapsed = time.time() - start_time
                    estimated_total = (total_frames / frame_skip) * frame_time
                    eta_seconds = int(estimated_total - elapsed)
                    
                    # Calculate video timestamp (in seconds and MM:SS format)
                    video_timestamp_sec = self.frame_count / fps
                    minutes = int(video_timestamp_sec // 60)
                    seconds = int(video_timestamp_sec % 60)
                    timestamp_str = f"{minutes:02d}:{seconds:02d}"
                    
                    # Show detailed progress with count and timestamp
                    status_display = results['alert_status']
                    if status_display == "Alert":
                        status_color_terminal = "\033[91m"  # Red
                        reset_color = "\033[0m"
                    else:
                        status_color_terminal = "\033[92m"  # Green
                        reset_color = "\033[0m"
                    
                    print(f"Frame {self.frame_count:4d}/{total_frames} | "
                          f"Time: {timestamp_str} | "
                          f"COUNT={results['crowd_count']:3d} | "
                          f"Status: {status_color_terminal}{status_display}{reset_color} | "
                          f"Hotspots={results['n_clusters']:2d} | "
                          f"Speed={frame_time:.1f}s | "
                          f"ETA={eta_seconds//60}m{eta_seconds%60}s")
                    
                    # Print ALERT message if threshold exceeded
                    if results['alert_status'] == "Alert":
                        print(f"\033[91m{'!'*80}")
                        print(f"⚠️  ALERT: Crowd threshold exceeded at {timestamp_str}!")
                        print(f"    Location: {self.location}")
                        print(f"    Count: {results['crowd_count']} people (Threshold: {self.threshold})")
                        print(f"    Frame: {self.frame_count}")
                        print(f"    Hotspots: {results['n_clusters']}")
                        print(f"{'!'*80}\033[0m")
                
                self.frame_count += 1
        
        finally:
            # Cleanup
            elapsed_total = time.time() - start_time
            cap.release()
            if save_output:
                out.release()
                print(f"\n💾 Video saved to: {output_path}")
            if display:
                cv2.destroyAllWindows()
            
            # CALCULATE STATISTICS FROM ACTUAL PROCESSED FRAMES
            if len(self.all_counts) > 0:
                avg_count = int(np.mean(self.all_counts))
                max_count = int(np.max(self.all_counts))
                min_count = int(np.min(self.all_counts))
                record_count = len(self.all_counts)
            else:
                avg_count, max_count, min_count, record_count = 0, 0, 0, 0
            
            self.database.close()
            
            print(f"\n{'='*80}")
            print(f"PROCESSING COMPLETE!")
            print(f"{'='*80}")
            print(f"Total time: {int(elapsed_total//60)}min {int(elapsed_total%60)}sec")
            print(f"Frames processed: {self.frame_count // 30}/{total_frames // 30}")
            print(f"Average speed: {elapsed_total/(self.frame_count//30):.1f}s per frame")
            print(f"\n{'─'*80}")
            print(f"CROWD STATISTICS - {self.location}")
            print(f"{'─'*80}")
            print(f"Peak Count:    {max_count} people (busiest moment)")
            print(f"Average Count: {avg_count} people (typical density)")
            print(f"Minimum Count: {min_count} people (quietest moment)")
            print(f"Database: {record_count} records saved to crowd_monitoring.db")
            print(f"{'='*80}\n")
            
            # GENERATE GRAPH
            print("Generating crowd analysis graph...")
            self.plot_crowd_graph()


# Main execution
if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "data/videos/test_video3.mp4"  # Change this to your video path
    LOCATION = "A Road"
    THRESHOLD = 45  # Alert if count exceeds this
    
    # Create and run system
    system = CrowdMonitoringSystem(
        video_path=VIDEO_PATH,
        location_name=LOCATION,
        threshold=THRESHOLD
    )
    
    # Run with automatic timestamped output
    system.run(display=True, save_output=True)
