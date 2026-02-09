"""
Standalone Posture Data Processor
Receives batched frame data via WebSocket, stores in memory, and generates debug files.
This is completely independent from main.py and can be deleted without affecting main.py.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from scipy import interpolate


# ============================================================================
# INCREMENTAL STREAMING FILTERS
# ============================================================================

class StreamingEMA:
    """Exponential Moving Average - O(1) per update, no history storage"""
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.ema = None
    
    def update(self, new_value: float) -> float:
        if new_value is None:
            return self.ema if self.ema is not None else None
        
        if self.ema is None:
            self.ema = new_value
        else:
            self.ema = self.alpha * new_value + (1 - self.alpha) * self.ema
        return self.ema
    
    def reset(self):
        self.ema = None


class StreamingKalman:
    """Kalman Filter - O(1) per update, optimal for noisy sensor data"""
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.estimate = None
        self.error_estimate = 1.0
        self.Q = process_noise
        self.R = measurement_noise
    
    def update(self, measurement: float) -> float:
        if measurement is None:
            return self.estimate
        
        if self.estimate is None:
            self.estimate = measurement
            return self.estimate
        
        # Prediction step
        predicted_estimate = self.estimate
        predicted_error = self.error_estimate + self.Q
        
        # Update step
        kalman_gain = predicted_error / (predicted_error + self.R)
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.error_estimate = (1 - kalman_gain) * predicted_error
        
        return self.estimate
    
    def reset(self):
        self.estimate = None
        self.error_estimate = 1.0


class StreamingDoubleEMA:
    """Double Exponential Smoothing (Holt's method) - Handles trends"""
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = 0
    
    def update(self, new_value: float) -> float:
        if new_value is None:
            return self.level if self.level is not None else None
        
        if self.level is None:
            self.level = new_value
            return self.level
        
        prev_level = self.level
        self.level = self.alpha * new_value + (1 - self.alpha) * (self.level + self.trend)
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        
        return self.level + self.trend
    
    def reset(self):
        self.level = None
        self.trend = 0


class RollingWindowFilter:
    """Rolling window filter (Gaussian/Savitzky-Golay) - O(window_size) per update"""
    def __init__(self, window_size: int = 7, filter_mode: str = 'gaussian', sigma: float = 1.5):
        self.window_size = window_size
        self.filter_mode = filter_mode
        self.sigma = sigma
        self.buffer = []
        
        # Precompute coefficients
        if filter_mode == 'gaussian':
            self._compute_gaussian_kernel()
        elif filter_mode == 'savgol':
            self._compute_savgol_coefficients()
    
    def _compute_gaussian_kernel(self):
        """Precompute Gaussian kernel weights"""
        import math
        half_window = self.window_size // 2
        kernel = []
        kernel_sum = 0
        
        for i in range(-half_window, half_window + 1):
            weight = math.exp(-(i * i) / (2 * self.sigma * self.sigma))
            kernel.append(weight)
            kernel_sum += weight
        
        self.kernel = [k / kernel_sum for k in kernel]
    
    def _compute_savgol_coefficients(self):
        """Precompute Savitzky-Golay coefficients"""
        coefficients = {
            5: [-3, 12, 17, 12, -3],
            7: [-2, 3, 6, 7, 6, 3, -2],
            9: [-21, 14, 39, 54, 59, 54, 39, 14, -21]
        }
        
        if self.window_size == 5:
            self.kernel = [c / 35 for c in coefficients[5]]
        elif self.window_size == 7:
            self.kernel = [c / 21 for c in coefficients[7]]
        elif self.window_size == 9:
            self.kernel = [c / 231 for c in coefficients[9]]
        else:
            self.kernel = [c / 21 for c in coefficients[7]]
    
    def update(self, new_value: float) -> float:
        if new_value is None:
            if len(self.buffer) > 0:
                return self.buffer[-1]
            return None
        
        # Add to buffer
        self.buffer.append(new_value)
        
        # Keep only window_size elements
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Not enough data yet - return raw value
        if len(self.buffer) < self.window_size:
            return new_value
        
        # Apply convolution
        filtered_value = sum(self.buffer[i] * self.kernel[i] for i in range(self.window_size))
        return filtered_value
    
    def reset(self):
        self.buffer.clear()


class IncrementalHybridFilter:
    """
    Hybrid incremental filter combining multiple stages:
    Stage 1: Kalman (removes sensor noise)
    Stage 2: Rolling Gaussian (smooths movement)
    
    Memory: ~10 floats per joint
    Computation: O(window_size) per update (constant for fixed window)
    """
    def __init__(self):
        self.kalman = StreamingKalman(process_noise=0.01, measurement_noise=0.1)
        self.gaussian = RollingWindowFilter(window_size=7, filter_mode='gaussian', sigma=1.5)
    
    def update(self, raw_angle: float) -> float:
        if raw_angle is None:
            return None
        
        # Stage 1: Kalman removes sensor noise
        kalman_filtered = self.kalman.update(raw_angle)
        
        # Stage 2: Gaussian smooths movement
        final_filtered = self.gaussian.update(kalman_filtered)
        
        return final_filtered
    
    def reset(self):
        self.kalman.reset()
        self.gaussian.reset()


def create_incremental_filter(filter_type: str, config: Dict):
    """Factory function to create appropriate incremental filter"""
    if filter_type == 'ema':
        return StreamingEMA(alpha=config['ema_alpha'])
    elif filter_type == 'double_ema':
        return StreamingDoubleEMA(
            alpha=config['double_ema_alpha'],
            beta=config['double_ema_beta']
        )
    elif filter_type == 'kalman':
        return StreamingKalman(
            process_noise=config['kalman_process_noise'],
            measurement_noise=config['kalman_measurement_noise']
        )
    elif filter_type == 'savgol':
        return RollingWindowFilter(
            window_size=config['savgol_window_size'],
            filter_mode='savgol'
        )
    elif filter_type == 'gaussian':
        return RollingWindowFilter(
            window_size=config['gaussian_window_size'],
            filter_mode='gaussian',
            sigma=config['gaussian_sigma']
        )
    elif filter_type == 'hybrid':
        return IncrementalHybridFilter()
    else:
        # Default to hybrid
        return IncrementalHybridFilter()


# ============================================================================
# CONFIGURATION
# ============================================================================

# Smoothing configuration (matching main.py SMOOTHING_CONFIG)
SMOOTHING_CONFIG = {
    'filter_type': 'gaussian',      # Options: 'savgol', 'ema', 'double_ema', 'gaussian', 'kalman', 'hybrid'
    'savgol_window_size': 7,        # 5, 7, or 9
    'ema_alpha': 0.3,               # 0.1-0.5
    'double_ema_alpha': 0.3,
    'double_ema_beta': 0.1,
    'gaussian_window_size': 7,
    'gaussian_sigma': 1.5,
    'kalman_process_noise': 0.01,
    'kalman_measurement_noise': 0.1
}


# ============================================================================
# DATA STORAGE CLASS
# ============================================================================

class JointDataStore:
    """Stores angle data points for a single joint (either trainer or user)"""
    
    def __init__(self, joint_name: str):
        self.joint_name = joint_name
        self.timestamps: List[float] = []
        self.angles: List[float] = []              # Raw angles (with confidence filtering)
        self.angles_smoothed: List[float] = []     # Incrementally filtered angles
        self.confidences: List[float] = []
        self.interpolated_60fps: List[float] = []  # Auto-interpolated 60 FPS data
        self.interpolated_filtered: List[float] = []  # Filtered interpolated data
        self.last_interpolated_index: int = 0      # Track which raw data index we last interpolated up to
        self.last_interpolated_timestamp: float = 0.0  # Last 60 FPS timestamp we generated
        
        # Create incremental filter instance (maintains state, no reprocessing)
        self.filter = create_incremental_filter(
            SMOOTHING_CONFIG['filter_type'],
            SMOOTHING_CONFIG
        )
        
        # Create separate filter for interpolated data
        self.interpolated_filter = create_incremental_filter(
            SMOOTHING_CONFIG['filter_type'],
            SMOOTHING_CONFIG
        )
    
    def add_data_point(self, timestamp: float, angle: float, confidence: float):
        """Add a new data point and apply incremental filter"""
        self.timestamps.append(timestamp)
        self.angles.append(angle)
        self.confidences.append(confidence)
        
        # Apply filter incrementally (O(1) or O(window_size), no reprocessing)
        smoothed = self.filter.update(angle)
        self.angles_smoothed.append(smoothed)
        
        # Auto-interpolate after adding data
        self._update_interpolation()
    
    def _update_interpolation(self):
        """Internal method to update 60 FPS interpolation incrementally"""
        if len(self.timestamps) < 2:
            self.interpolated_60fps = []
            self.last_interpolated_index = 0
            self.last_interpolated_timestamp = 0.0
            return
        
        fps = 60
        frame_duration = 1.0 / fps  # 0.016666... seconds per frame
        current_data_count = len(self.timestamps)
        
        # First time interpolation - do full interpolation
        if self.last_interpolated_index == 0:
            timestamps = np.array(self.timestamps)
            angles = np.array(self.angles_smoothed)
            
            first_timestamp = timestamps[0]
            last_timestamp = timestamps[-1]
            
            # Align start time to nearest 60 FPS frame boundary at or before first timestamp
            whole_second = int(np.floor(first_timestamp))
            offset_within_second = first_timestamp - whole_second
            frame_index_within_second = int(np.floor(offset_within_second / frame_duration))
            aligned_start_time = whole_second + (frame_index_within_second * frame_duration)
            
            # Generate timestamps at exactly 60 FPS from aligned start
            duration = last_timestamp - aligned_start_time
            num_frames = int(np.floor(duration / frame_duration)) + 1
            
            interpolated_timestamps = np.array([
                aligned_start_time + (i * frame_duration) 
                for i in range(num_frames)
            ])
            
            interpolated_timestamps = interpolated_timestamps[interpolated_timestamps <= last_timestamp]
            
            if len(interpolated_timestamps) == 0:
                return
            
            # Perform linear interpolation
            interpolator = interpolate.interp1d(
                timestamps, 
                angles, 
                kind='linear',
                fill_value='extrapolate'
            )
            interpolated_values = interpolator(interpolated_timestamps)
            
            # Store interpolated values and apply filter to each
            self.interpolated_60fps = []
            self.interpolated_filtered = []
            for value in interpolated_values.tolist():
                self.interpolated_60fps.append(value)
                # Apply filter to interpolated value and round to integer
                filtered_value = self.interpolated_filter.update(value)
                self.interpolated_filtered.append(round(filtered_value))
            
            self.last_interpolated_index = current_data_count - 1
            self.last_interpolated_timestamp = interpolated_timestamps[-1]
            return
        
        # Incremental interpolation - only process new data
        # Use 2 boundary points from previous data for smooth interpolation
        boundary_points = 2
        start_idx = max(0, self.last_interpolated_index - boundary_points + 1)
        
        # Check if we have new data
        if current_data_count <= self.last_interpolated_index:
            return  # No new data
        
        # Get slice of data including boundary points and new data
        timestamps_slice = np.array(self.timestamps[start_idx:])
        angles_slice = np.array(self.angles_smoothed[start_idx:])
        
        if len(timestamps_slice) < 2:
            return
        
        last_timestamp = timestamps_slice[-1]
        
        # Calculate next 60 FPS timestamp after last interpolated point
        next_start_time = self.last_interpolated_timestamp + frame_duration
        
        # Check if we have enough new data for at least one frame
        if last_timestamp < next_start_time:
            return
        
        # Generate new 60 FPS timestamps from next_start_time to last_timestamp
        duration = last_timestamp - next_start_time
        num_new_frames = int(np.floor(duration / frame_duration)) + 1
        
        new_interpolated_timestamps = np.array([
            next_start_time + (i * frame_duration)
            for i in range(num_new_frames)
        ])
        
        new_interpolated_timestamps = new_interpolated_timestamps[new_interpolated_timestamps <= last_timestamp]
        
        if len(new_interpolated_timestamps) == 0:
            return
        
        # Interpolate only the new timestamps using the slice (includes boundary points)
        interpolator = interpolate.interp1d(
            timestamps_slice,
            angles_slice,
            kind='linear',
            fill_value='extrapolate'
        )
        new_interpolated_values = interpolator(new_interpolated_timestamps)
        
        # Append only the new interpolated values and apply filter to each
        new_values_list = new_interpolated_values.tolist()
        for value in new_values_list:
            self.interpolated_60fps.append(value)
            # Apply filter to interpolated value and round to integer
            filtered_value = self.interpolated_filter.update(value)
            self.interpolated_filtered.append(round(filtered_value))
        
        self.last_interpolated_index = current_data_count - 1
        self.last_interpolated_timestamp = new_interpolated_timestamps[-1]
    
# ============================================================================
# JOINT CONFIGURATION
# ============================================================================

# 15 joint names
JOINT_NAMES = [
    "Left Shoulder",
    "Left Elbow",
    "Left Wrist",
    "Right Shoulder",
    "Right Elbow",
    "Right Wrist",
    "Left Hip",
    "Left Knee",
    "Left Ankle",
    "Right Hip",
    "Right Knee",
    "Right Ankle",
    "Left Spine",
    "Right Spine",
    "Neck"
]

# Mapping from frontend short keys to full joint names
JOINT_KEY_MAPPING = {
    'ls': 'Left Shoulder',
    'le': 'Left Elbow',
    'lw': 'Left Wrist',
    'rs': 'Right Shoulder',
    're': 'Right Elbow',
    'rw': 'Right Wrist',
    'lh': 'Left Hip',
    'lk': 'Left Knee',
    'la': 'Left Ankle',
    'rh': 'Right Hip',
    'rk': 'Right Knee',
    'ra': 'Right Ankle',
    'lsp': 'Left Spine',
    'rsp': 'Right Spine',
    'n': 'Neck'
}

# Initialize storage for all 30 joint stores (15 trainer + 15 user)
trainer_stores: Dict[str, JointDataStore] = {
    joint_name: JointDataStore(f"Trainer_{joint_name}")
    for joint_name in JOINT_NAMES
}

user_stores: Dict[str, JointDataStore] = {
    joint_name: JointDataStore(f"User_{joint_name}")
    for joint_name in JOINT_NAMES
}

# Confidence threshold (matching main.py line 1023)
CONFIDENCE_THRESHOLD = 40

# Track last valid angles for confidence filtering
last_valid_trainer_angles: Dict[str, float] = {}  # {joint_name: last_valid_angle}
last_valid_user_angles: Dict[str, float] = {}     # {joint_name: last_valid_angle}

# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_batch(batch_data: Dict) -> Dict:
    """
    Process a batch of frames and store the data
    Returns statistics about what was processed
    """
    frames = batch_data.get('frames', [])
    frame_count = batch_data.get('frameCount', len(frames))
    batch_timestamp = batch_data.get('batchTimestamp', datetime.now().timestamp())
    
    stats = {
        'batch_timestamp': batch_timestamp,
        'frame_count': frame_count,
        'frames_processed': 0,
        'trainer_joints_found': set(),
        'user_joints_found': set(),
        'sample_values': {}
    }
    
    # Process each frame
    for frame in frames:
        frame_timestamp = frame.get('t', datetime.now().timestamp()) / 1000.0  # Convert ms to seconds
        stats['frames_processed'] += 1
        
        # Process each joint key in the frame
        for key, value in frame.items():
            if key == 't' or key == 'gest':  # Skip timestamp and gesture
                continue
            
            # Determine if this is trainer or user data
            if key.startswith('tr_'):
                prefix = 'tr_'
                is_trainer = True
                stores_dict = trainer_stores
            elif key.startswith('u_'):
                prefix = 'u_'
                is_trainer = False
                stores_dict = user_stores
            else:
                continue
            
            # Extract the joint key (e.g., 'ls' from 'tr_ls' or 'u_ls')
            joint_key = key[len(prefix):]
            
            # Map to full joint name
            joint_name = JOINT_KEY_MAPPING.get(joint_key)
            if not joint_name:
                print(f"[WARNING] Unknown joint key: {joint_key}")
                continue
            
            # Extract angle and confidence from [angle, confidence] array
            if isinstance(value, list) and len(value) >= 2:
                angle = value[0]
                confidence = value[1]
                
                # Apply confidence threshold filtering (matching main.py logic)
                angle_to_store = angle
                
                if is_trainer:
                    # Trainer data
                    if confidence >= CONFIDENCE_THRESHOLD:
                        # Good confidence - use actual angle
                        angle_to_store = angle
                        last_valid_trainer_angles[joint_name] = angle
                    else:
                        # Low confidence
                        if joint_name in last_valid_trainer_angles:
                            # Use previous valid angle
                            angle_to_store = last_valid_trainer_angles[joint_name]
                        else:
                            # First frame with low confidence - store anyway as baseline
                            angle_to_store = angle
                            last_valid_trainer_angles[joint_name] = angle
                else:
                    # User data
                    if confidence >= CONFIDENCE_THRESHOLD:
                        # Good confidence - use actual angle
                        angle_to_store = angle
                        last_valid_user_angles[joint_name] = angle
                    else:
                        # Low confidence
                        if joint_name in last_valid_user_angles:
                            # Use previous valid angle
                            angle_to_store = last_valid_user_angles[joint_name]
                        else:
                            # First frame with low confidence - store anyway as baseline
                            angle_to_store = angle
                            last_valid_user_angles[joint_name] = angle
                
                # Store the data (filtered angle with original confidence)
                if joint_name in stores_dict:
                    stores_dict[joint_name].add_data_point(frame_timestamp, angle_to_store, confidence)
                    
                    # Track which joints were found
                    if is_trainer:
                        stats['trainer_joints_found'].add(joint_name)
                    else:
                        stats['user_joints_found'].add(joint_name)
                    
                    # Store sample values for the first frame
                    if stats['frames_processed'] == 1:
                        if joint_name not in stats['sample_values']:
                            stats['sample_values'][joint_name] = {}
                        
                        if is_trainer:
                            stats['sample_values'][joint_name]['trainer'] = {
                                'angle': angle,
                                'confidence': confidence
                            }
                        else:
                            stats['sample_values'][joint_name]['user'] = {
                                'angle': angle,
                                'confidence': confidence
                            }
    
    # Convert sets to lists for JSON serialization
    stats['trainer_joints_found'] = sorted(list(stats['trainer_joints_found']))
    stats['user_joints_found'] = sorted(list(stats['user_joints_found']))
    
    return stats


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Posture Data Processor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections for plot updates
plot_websocket_connections = []

async def broadcast_update():
    """Broadcast update notification to all connected plot viewers"""
    for connection in plot_websocket_connections[:]:
        try:
            await connection.send_json({"type": "data_updated"})
        except Exception as e:
            print(f"[BROADCAST ERROR] Failed to send: {e}")
            plot_websocket_connections.remove(connection)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to receive batched frame data from frontend
    Expected format: {frames: [...], frameCount: N, batchTimestamp: T}
    """
    await websocket.accept()
    print("\n[WEBSOCKET] Client connected to /ws")
    print("[WEBSOCKET] Waiting for data...")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON data
                batch_data = json.loads(data)
                
                # Process the batch and get statistics
                stats = process_batch(batch_data)
                      
                # Broadcast update to all connected plot viewers
                await broadcast_update()
                
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse JSON: {e}")
                await websocket.send_json({
                    "status": "error",
                    "message": f"Invalid JSON: {str(e)}"
                })
            except Exception as e:
                print(f"[ERROR] Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({
                    "status": "error",
                    "message": f"Processing error: {str(e)}"
                })
    
    except WebSocketDisconnect:
        print("[WEBSOCKET] Client disconnected")
    except Exception as e:
        print(f"[WEBSOCKET ERROR] {e}")
        import traceback
        traceback.print_exc()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Posture Data Processor is running",
        "websocket_endpoint": "/ws",
        "plot_endpoint": "/plot",
        "status": "active"
    }

@app.get("/plot", response_class=HTMLResponse)
async def get_plot_page():
    """HTML page with uPlot charts showing all stored data"""
    
    # Get active filter info for display
    filter_descriptions = {
        'savgol': 'Savitzky-Golay - Preserves peaks and valleys',
        'ema': 'Exponential Moving Average - Fast response, minimal lag',
        'double_ema': 'Double EMA - Handles trends well',
        'gaussian': 'Gaussian - Heavy noise reduction, smooth curves',
        'kalman': 'Kalman - Optimal for noisy sensor data',
        'hybrid': 'Hybrid - Best overall (Kalman + Savitzky-Golay + EMA)'
    }
    filter_type = SMOOTHING_CONFIG['filter_type']
    filter_desc = filter_descriptions.get(filter_type, 'Unknown filter')
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Posture Data - Live Plot</title>
        <script src="https://unpkg.com/uplot@1.6.24/dist/uPlot.iife.min.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/uplot@1.6.24/dist/uPlot.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                max-width: 1600px;
                margin: 0 auto;
            }
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .chart-container {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .chart-container h3 {
                margin-top: 0;
                margin-bottom: 10px;
                color: #555;
                font-size: 14px;
            }
            .u-legend {
                font-size: 11px;
            }
            .info-box {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
                font-size: 14px;
            }
            .refresh-button {
                display: block;
                margin: 20px auto;
                padding: 10px 30px;
                background: #2196f3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
            }
            .refresh-button:hover {
                background: #1976d2;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Posture Data - Live Plot (Raw Data)</h1>
            
            <div class="info-box">
                <strong>Data Source:</strong> Stored in Python JointDataStore objects<br>
                <strong>Visualization:</strong> Raw (confidence-filtered) + Smoothed angles<br>
                <strong>Active Filter:</strong> <code>" + filter_type.upper() + "</code> - " + filter_desc + "<br>
                <strong>Confidence Threshold:</strong> ≥40%<br>
                <strong>Time Window:</strong> Last 10 seconds of data<br>
                <strong>Updates:</strong> Real-time via WebSocket (updates as data arrives)
            </div>
            
            <button class="refresh-button" onclick="loadData()">Manual Refresh</button>
            
            <div class="charts-grid" id="charts-container"></div>
        </div>

        <script>
            // Joint mapping
            const joints = {
                'Left Shoulder': 'Left Shoulder',
                'Left Elbow': 'Left Elbow',
                'Left Wrist': 'Left Wrist',
                'Right Shoulder': 'Right Shoulder',
                'Right Elbow': 'Right Elbow',
                'Right Wrist': 'Right Wrist',
                'Left Hip': 'Left Hip',
                'Left Knee': 'Left Knee',
                'Left Ankle': 'Left Ankle',
                'Right Hip': 'Right Hip',
                'Right Knee': 'Right Knee',
                'Right Ankle': 'Right Ankle',
                'Left Spine': 'Left Spine',
                'Right Spine': 'Right Spine',
                'Neck': 'Neck'
            };
            
            const charts = {};
            const chartsContainer = document.getElementById('charts-container');
            
            // Create a chart for each joint
            Object.keys(joints).forEach(jointName => {
                // Create container
                const container = document.createElement('div');
                container.className = 'chart-container';
                container.innerHTML = `<h3>${jointName} (Count: <span id="count-${jointName.replace(/\\s/g, '-')}">0</span>)</h3>`;
                chartsContainer.appendChild(container);
                
                // uPlot options
                const opts = {
                    width: 400,
                    height: 200,
                    scales: {
                        x: {
                            time: true
                        },
                        y: {
                            range: [0, 180]
                        }
                    },
                    series: [
                        {
                            label: "Time",
                            value: (u, v) => {
                                if (v == null) return "-";
                                const date = new Date(v * 1000);
                                const minutes = String(date.getMinutes()).padStart(2, '0');
                                const seconds = String(date.getSeconds()).padStart(2, '0');
                                const milliseconds = String(date.getMilliseconds()).padStart(3, '0');
                                return `${minutes}:${seconds}.${milliseconds}`;
                            }
                        },
                        {
                            label: "User Raw",
                            stroke: "#2196F3",
                            width: 1.5,
                            points: { show: false }
                        },
                        {
                            label: "Trainer Raw",
                            stroke: "#4CAF50",
                            width: 1.5,
                            points: { show: false }
                        },
                        {
                            label: "User Smoothed",
                            stroke: "#FF6B6B",
                            width: 2.5,
                            points: { show: false },
                            
                        },
                        {
                            label: "Trainer Smoothed",
                            stroke: "#FFA500",
                            width: 2.5,
                            points: { show: false },
                            
                        }
                    ],
                    axes: [
                        {
                            stroke: "#64748b",
                            grid: { stroke: "#e2e8f0", width: 1 }
                        },
                        {
                            stroke: "#64748b",
                            grid: { stroke: "#e2e8f0", width: 1 },
                            label: "Angle (°)"
                        }
                    ],
                    legend: {
                        show: true
                    }
                };
                
                // Create chart with initial empty data (5 series)
                const data = [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]
                ];
                
                charts[jointName] = new uPlot(opts, data, container);
            });
            
            // Function to load data from the API
            async function loadData() {
                try {
                    const response = await fetch('/api/plot-data');
                    const data = await response.json();
                    
                    // Update each chart with stored data
                    Object.keys(joints).forEach(jointName => {
                        const trainerStore = data.trainer_stores[jointName];
                        const userStore = data.user_stores[jointName];
                        
                        // Get timestamps and angles
                        const trainerTimestamps = trainerStore.timestamps || [];
                        const trainerAngles = trainerStore.angles || [];
                        const trainerAnglesSmoothed = trainerStore.angles_smoothed || [];
                        const userTimestamps = userStore.timestamps || [];
                        const userAngles = userStore.angles || [];
                        const userAnglesSmoothed = userStore.angles_smoothed || [];
                        
                        // Combine timestamps (union of both)
                        const allTimestamps = [...new Set([...trainerTimestamps, ...userTimestamps])].sort();
                        
                        // If no data, use dummy data
                        if (allTimestamps.length === 0) {
                            charts[jointName].setData([[0], [null], [null], [null], [null]]);
                            const countElement = document.getElementById(`count-$\{jointName.replace(/\\s/g, '-')}`);
                            if (countElement) {
                                countElement.textContent = '0';
                            }
                            return;
                        }
                        
                        // Prepare data for uPlot - map to combined timeline
                        const userAnglesAligned = allTimestamps.map(t => {
                            const idx = userTimestamps.indexOf(t);
                            return idx >= 0 ? userAngles[idx] : null;
                        });
                        const trainerAnglesAligned = allTimestamps.map(t => {
                            const idx = trainerTimestamps.indexOf(t);
                            return idx >= 0 ? trainerAngles[idx] : null;
                        });
                        const userAnglesSmoothedAligned = allTimestamps.map(t => {
                            const idx = userTimestamps.indexOf(t);
                            return idx >= 0 ? userAnglesSmoothed[idx] : null;
                        });
                        const trainerAnglesSmoothedAligned = allTimestamps.map(t => {
                            const idx = trainerTimestamps.indexOf(t);
                            return idx >= 0 ? trainerAnglesSmoothed[idx] : null;
                        });
                        
                        // Update chart with 5 series
                        const chartData = [
                            allTimestamps,
                            userAnglesAligned,
                            trainerAnglesAligned,
                            userAnglesSmoothedAligned,
                            trainerAnglesSmoothedAligned
                        ];
                        
                        charts[jointName].setData(chartData);
                        
                        // Update count
                        const totalCount = trainerStore.count + userStore.count;
                        const countElement = document.getElementById(`count-$\{jointName.replace(/\\s/g, '-')}`);
                        if (countElement) {
                            countElement.textContent = totalCount;
                        }
                    });
                    
                    console.log('Charts updated successfully');
                } catch (error) {
                    console.error('Error loading data:', error);
                }
            }
            
            // Connect to WebSocket for real-time updates
            let plotWs = null;
            
            function connectPlotWebSocket() {
                plotWs = new WebSocket('ws://127.0.0.1:8000/ws/plot-updates');
                
                plotWs.onopen = () => {
                    console.log('[PLOT WS] Connected to plot updates');
                };
                
                plotWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'data_updated') {
                        console.log('[PLOT WS] Data updated, refreshing charts...');
                        loadData();
                    }
                };
                
                plotWs.onerror = (error) => {
                    console.error('[PLOT WS] Error:', error);
                };
                
                plotWs.onclose = () => {
                    console.log('[PLOT WS] Disconnected, reconnecting in 2 seconds...');
                    setTimeout(connectPlotWebSocket, 2000);
                };
            }
            
            // Load data initially
            loadData();
            
            // Connect to WebSocket for real-time updates
            connectPlotWebSocket();
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/api/plot-data")
async def get_plot_data():
    """Get last 10 seconds of stored data for plotting"""
    import time
    current_time = time.time()
    ten_seconds_ago = current_time - 5
    
    def filter_last_10_seconds(store):
        """Filter data to only include last 10 seconds"""
        if len(store.timestamps) == 0:
            return {
                "timestamps": [],
                "angles": [],
                "confidences": [],
                "count": 0
            }
        
        # Find indices where timestamp >= ten_seconds_ago
        filtered_indices = [i for i, t in enumerate(store.timestamps) if t >= ten_seconds_ago]
        
        if not filtered_indices:
            return {
                "timestamps": [],
                "angles": [],
                "confidences": [],
                "count": 0
            }
        
        return {
            "timestamps": [store.timestamps[i] for i in filtered_indices],
            "angles": [store.angles[i] for i in filtered_indices],
            "angles_smoothed": [store.angles_smoothed[i] for i in filtered_indices],
            "confidences": [store.confidences[i] for i in filtered_indices],
            "count": len(filtered_indices)
        }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "trainer_stores": {
            joint_name: filter_last_10_seconds(store)
            for joint_name, store in trainer_stores.items()
        },
        "user_stores": {
            joint_name: filter_last_10_seconds(store)
            for joint_name, store in user_stores.items()
        }
    }

@app.get("/interpolated-plot", response_class=HTMLResponse)
async def get_interpolated_plot_page():
    """HTML page with uPlot charts showing 60 FPS interpolated data for all 15 joints"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>60 FPS Interpolated Data - Live Plot</title>
        <script src="https://unpkg.com/uplot@1.6.24/dist/uPlot.iife.min.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/uplot@1.6.24/dist/uPlot.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                max-width: 1600px;
                margin: 0 auto;
            }
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .chart-container {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .chart-container h3 {
                margin-top: 0;
                margin-bottom: 10px;
                color: #555;
                font-size: 14px;
            }
            .u-legend {
                font-size: 11px;
            }
            .info-box {
                background: #d1f2eb;
                border-left: 4px solid #00C853;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
                font-size: 14px;
            }
            .warning-box {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 10px 15px;
                margin-bottom: 20px;
                border-radius: 4px;
                font-size: 13px;
            }
            .refresh-button {
                display: block;
                margin: 20px auto;
                padding: 10px 30px;
                background: #00C853;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
            }
            .refresh-button:hover {
                background: #00A844;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 60 FPS Interpolated Data - Live Plot</h1>
            
            <div class="info-box">
                <strong>Data Type:</strong> 60 FPS Interpolated (Second-Boundary Aligned) + Filtered<br>
                <strong>Interpolation:</strong> Linear interpolation with incremental updates<br>
                <strong>Filtering:</strong> Same filter applied to interpolated data for smoothness<br>
                <strong>FPS:</strong> Exactly 60 frames per second (0.0167s intervals)<br>
                <strong>Alignment:</strong> Indices 0-59 = 1st second, 60-119 = 2nd second, etc.<br>
                <strong>Display:</strong> Last 5 seconds (300 frames) shown for both raw interpolated and filtered<br>
                <strong>Updates:</strong> Real-time via WebSocket (auto-updates as batches arrive)
            </div>
            
            <div class="warning-box">
                ⚠️ <strong>Note:</strong> X-axis shows frame indices (not timestamps). Each point is exactly 1/60th of a second apart. Lighter colors show filtered data for smoother visualization.
            </div>
            
            <button class="refresh-button" onclick="loadData()">🔄 Manual Refresh</button>
            
            <div class="charts-grid" id="charts-container"></div>
        </div>

        <script>
            // Joint names
            const joints = [
                'Left Shoulder',
                'Left Elbow',
                'Left Wrist',
                'Right Shoulder',
                'Right Elbow',
                'Right Wrist',
                'Left Hip',
                'Left Knee',
                'Left Ankle',
                'Right Hip',
                'Right Knee',
                'Right Ankle',
                'Left Spine',
                'Right Spine',
                'Neck'
            ];
            
            const charts = {};
            const chartsContainer = document.getElementById('charts-container');
            
            // Create a chart for each joint
            joints.forEach(jointName => {
                // Create container
                const container = document.createElement('div');
                container.className = 'chart-container';
                container.innerHTML = `<h3>${jointName} (<span id="count-${jointName.replace(/\\s/g, '-')}">0</span> frames @ 60fps)</h3>`;
                chartsContainer.appendChild(container);
                
                // uPlot options
                const opts = {
                    width: 400,
                    height: 200,
                    scales: {
                        x: {
                            time: false
                        },
                        y: {
                            range: [0, 180]
                        }
                    },
                    series: [
                        {
                            label: "Frame Index",
                            value: (u, v) => v != null ? `Frame ${v}` : "-"
                        },
                        {
                            label: "User (60fps)",
                            stroke: "#FF1744",
                            width: 1.5,
                            points: { show: false }
                        },
                        {
                            label: "User Filtered",
                            stroke: "#FF8A80",
                            width: 2,
                            points: { show: false }
                        },
                        {
                            label: "Trainer (60fps)",
                            stroke: "#00C853",
                            width: 1.5,
                            points: { show: false }
                        },
                        {
                            label: "Trainer Filtered",
                            stroke: "#69F0AE",
                            width: 2,
                            points: { show: false }
                        }
                    ],
                    axes: [
                        {
                            stroke: "#64748b",
                            grid: { stroke: "#e2e8f0", width: 1 },
                            label: "Frame Index"
                        },
                        {
                            stroke: "#64748b",
                            grid: { stroke: "#e2e8f0", width: 1 },
                            label: "Angle (°)"
                        }
                    ],
                    legend: {
                        show: true
                    }
                };
                
                // Create chart with initial empty data
                const data = [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]
                ];
                
                charts[jointName] = new uPlot(opts, data, container);
            });
            
            // Function to load data from the API
            async function loadData() {
                try {
                    const response = await fetch('/api/interpolated-data');
                    const data = await response.json();
                    
                    // Update each chart with interpolated data
                    joints.forEach(jointName => {
                        const trainerData = data.trainer[jointName]?.interpolated || [];
                        const trainerFiltered = data.trainer[jointName]?.filtered || [];
                        const userData = data.user[jointName]?.interpolated || [];
                        const userFiltered = data.user[jointName]?.filtered || [];
                        
                        // Use the longest array length for frame indices
                        const maxLength = Math.max(
                            trainerData.length, 
                            trainerFiltered.length,
                            userData.length,
                            userFiltered.length
                        );
                        
                        if (maxLength === 0) {
                            charts[jointName].setData([[0], [null], [null], [null], [null]]);
                            const countElement = document.getElementById(`count-${jointName.replace(/\\s/g, '-')}`);
                            if (countElement) {
                                countElement.textContent = '0';
                            }
                            return;
                        }
                        
                        // Create frame indices (0, 1, 2, 3, ...)
                        const frameIndices = Array.from({length: maxLength}, (_, i) => i);
                        
                        // Pad shorter arrays with nulls
                        const userDataPadded = [...userData];
                        const userFilteredPadded = [...userFiltered];
                        const trainerDataPadded = [...trainerData];
                        const trainerFilteredPadded = [...trainerFiltered];
                        
                        while (userDataPadded.length < maxLength) userDataPadded.push(null);
                        while (userFilteredPadded.length < maxLength) userFilteredPadded.push(null);
                        while (trainerDataPadded.length < maxLength) trainerDataPadded.push(null);
                        while (trainerFilteredPadded.length < maxLength) trainerFilteredPadded.push(null);
                        
                        // Update chart
                        const chartData = [
                            frameIndices,
                            userDataPadded,
                            userFilteredPadded,
                            trainerDataPadded,
                            trainerFilteredPadded
                        ];
                        
                        charts[jointName].setData(chartData);
                        
                        // Update count
                        const countElement = document.getElementById(`count-${jointName.replace(/\\s/g, '-')}`);
                        if (countElement) {
                            const userCount = userData.length;
                            const trainerCount = trainerData.length;
                            countElement.innerHTML = `U:${userCount} / T:${trainerCount}`;
                        }
                    });
                    
                    console.log('Interpolated charts updated successfully');
                } catch (error) {
                    console.error('Error loading interpolated data:', error);
                }
            }
            
            // Connect to WebSocket for real-time updates
            let plotWs = null;
            
            function connectPlotWebSocket() {
                plotWs = new WebSocket('ws://127.0.0.1:8000/ws/plot-updates');
                
                plotWs.onopen = () => {
                    console.log('[INTERPOLATED PLOT WS] Connected');
                };
                
                plotWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'data_updated') {
                        loadData();
                    }
                };
                
                plotWs.onerror = (error) => {
                    console.error('[INTERPOLATED PLOT WS] Error:', error);
                };
                
                plotWs.onclose = () => {
                    console.log('[INTERPOLATED PLOT WS] Disconnected, reconnecting in 2 seconds...');
                    setTimeout(connectPlotWebSocket, 2000);
                };
            }
            
            // Load data initially
            loadData();
            
            // Connect to WebSocket for real-time updates
            connectPlotWebSocket();
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/api/interpolated-data")
async def get_interpolated_data():
    """Get 60 FPS interpolated data for all joints (last 5 seconds = 300 points)"""
    return {
        "timestamp": datetime.now().isoformat(),
        "trainer": {
            joint_name: {
                "interpolated": store.interpolated_60fps[-300:] if len(store.interpolated_60fps) > 0 else [],
                "filtered": store.interpolated_filtered[-300:] if len(store.interpolated_filtered) > 0 else []
            }
            for joint_name, store in trainer_stores.items()
        },
        "user": {
            joint_name: {
                "interpolated": store.interpolated_60fps[-300:] if len(store.interpolated_60fps) > 0 else [],
                "filtered": store.interpolated_filtered[-300:] if len(store.interpolated_filtered) > 0 else []
            }
            for joint_name, store in user_stores.items()
        }
    }

@app.get("/ifil/{role}/{joint_name}")
async def get_interpolated_filtered_angle(role: str, joint_name: str):
    """Get filtered interpolated data array for a specific joint and role"""
    if role.lower() not in ['trainer', 'user']:
        return {"error": "Role must be 'trainer' or 'user'"}
    
    stores = trainer_stores if role.lower() == "trainer" else user_stores
    
    if joint_name not in stores:
        return {
            "error": f"Joint '{joint_name}' not found for role '{role}'",
            "available_joints": list(stores.keys())
        }
    
    store = stores[joint_name]
    return store.interpolated_filtered if len(store.interpolated_filtered) > 0 else []

@app.websocket("/ws/plot-updates")
async def plot_updates_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time plot updates"""
    await websocket.accept()
    plot_websocket_connections.append(websocket)
    print("[PLOT WS] Plot viewer connected")
    
    try:
        while True:
            # Keep connection alive and wait for disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        plot_websocket_connections.remove(websocket)
        print("[PLOT WS] Plot viewer disconnected")
    except Exception as e:
        if websocket in plot_websocket_connections:
            plot_websocket_connections.remove(websocket)
        print(f"[PLOT WS ERROR] {e}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("POSTURE DATA PROCESSOR - STANDALONE APPLICATION")
    print("="*80)
    print(f"Initialized {len(trainer_stores)} trainer joint stores")
    print(f"Initialized {len(user_stores)} user joint stores")
    print("="*80 + "\n")
    
    # Run the FastAPI app on port 8000
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
