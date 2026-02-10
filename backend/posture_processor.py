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
        
        # Cumulative average tracking (batch-based for efficiency)
        self.cumulative_average: Optional[float] = None  # Running average of all raw angles from start
        self.raw_angles_count: int = 0                  # Count of raw angles used in cumulative average
        self.batch_buffer: List[float] = []             # Temporary buffer for current batch angles
        self.statistics_reset_index: int = 0            # Index in interpolated_filtered where statistics were last reset
        
        # Cumulative standard deviation tracking
        self.cumulative_sum_of_squares: float = 0.0     # Running sum of squared values for std dev calculation
        
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
        
        # Add to batch buffer for cumulative average calculation
        self.batch_buffer.append(angle)
        
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
    
    def finalize_batch(self):
        """
        Finalize the current batch by updating cumulative average.
        This should be called after a batch of data points has been added.
        Uses weighted average formula to efficiently update the cumulative average.
        """
        if len(self.batch_buffer) == 0:
            return  # No new data in this batch
        
        # Calculate average and sum of squares of current batch
        batch_avg = sum(self.batch_buffer) / len(self.batch_buffer)
        batch_count = len(self.batch_buffer)
        batch_sum_of_squares = sum(x * x for x in self.batch_buffer)
        
        # Update cumulative average using weighted formula
        if self.cumulative_average is None:
            # First batch
            self.cumulative_average = batch_avg
            self.raw_angles_count = batch_count
            self.cumulative_sum_of_squares = batch_sum_of_squares
        else:
            # Weighted average: new_avg = (old_avg * old_count + batch_avg * batch_count) / (old_count + batch_count)
            self.cumulative_average = (
                (self.cumulative_average * self.raw_angles_count) + (batch_avg * batch_count)
            ) / (self.raw_angles_count + batch_count)
            self.raw_angles_count += batch_count
            self.cumulative_sum_of_squares += batch_sum_of_squares
        
        # Clear the batch buffer
        self.batch_buffer.clear()
    
    def get_cumulative_average(self) -> Optional[float]:
        """Get the cumulative average of all raw angles from start (rounded to integer)"""
        if self.cumulative_average is not None:
            return round(self.cumulative_average)
        return None
    
    def get_cumulative_average_precise(self) -> Optional[float]:
        """Get the precise cumulative average (not rounded)"""
        return self.cumulative_average
    
    def get_cumulative_std_dev(self) -> Optional[float]:
        """Get the cumulative standard deviation of all raw angles from start (rounded to 2 decimal places)"""
        if self.cumulative_average is not None and self.raw_angles_count > 0:
            # Calculate variance: var = (sum(x^2) / n) - mean^2
            mean_of_squares = self.cumulative_sum_of_squares / self.raw_angles_count
            variance = mean_of_squares - (self.cumulative_average ** 2)
            # Handle floating point errors that might make variance slightly negative
            variance = max(0, variance)
            std_dev = variance ** 0.5
            return round(std_dev, 2)
        return None
    
    def get_cumulative_std_dev_precise(self) -> Optional[float]:
        """Get the precise cumulative standard deviation (not rounded)"""
        if self.cumulative_average is not None and self.raw_angles_count > 0:
            mean_of_squares = self.cumulative_sum_of_squares / self.raw_angles_count
            variance = mean_of_squares - (self.cumulative_average ** 2)
            variance = max(0, variance)
            return variance ** 0.5
        return None
    
    def get_window_average(self, window_size: int) -> Optional[float]:
        """
        Calculate average of the last N filtered interpolated angle data points.
        Uses the filtered_interpolated array for smoother results.
        Only includes data from after the last statistics reset.
        
        Args:
            window_size: Number of points to average (e.g., 120 for last 120 interpolated points)
        
        Returns:
            Average of last window_size points, or None if not enough data (requires minimum 2 points)
        """
        # Only use data from after the last reset point
        valid_data = self.interpolated_filtered[self.statistics_reset_index:]
        
        # Require at least 2 points for meaningful average
        if len(valid_data) < 2:
            return None
        
        if len(valid_data) < window_size:
            # Use all available data if less than window size
            window_size = len(valid_data)
        
        # Get last window_size points from valid data
        last_angles = valid_data[-window_size:]
        return sum(last_angles) / len(last_angles)
    
    def get_window_average_rounded(self, window_size: int) -> Optional[int]:
        """Get window average rounded to integer"""
        avg = self.get_window_average(window_size)
        if avg is not None:
            return round(avg)
        return None
    
    def get_window_average_count(self, window_size: int) -> int:
        """Get the actual number of elements used in window average calculation"""
        # Only use data from after the last reset point
        valid_data = self.interpolated_filtered[self.statistics_reset_index:]
        
        if len(valid_data) == 0:
            return 0
        
        # Return the actual number of elements used (min of requested window_size and available data)
        return min(len(valid_data), window_size)
    
    def get_window_std_dev(self, window_size: int) -> Optional[float]:
        """Get the standard deviation of the last N filtered interpolated angle data points (rounded to 2 decimal places)"""
        # Only use data from after the last reset point
        valid_data = self.interpolated_filtered[self.statistics_reset_index:]
        
        # Require at least 2 points for meaningful std dev
        if len(valid_data) < 2:
            return None
        
        # Use actual window size (min of requested and available)
        actual_window_size = min(len(valid_data), window_size)
        window_data = valid_data[-actual_window_size:]
        
        # Calculate mean
        mean = sum(window_data) / len(window_data)
        
        # Calculate variance
        variance = sum((x - mean) ** 2 for x in window_data) / len(window_data)
        
        # Return standard deviation
        std_dev = variance ** 0.5
        return round(std_dev, 2)
    
    def get_window_std_dev_precise(self, window_size: int) -> Optional[float]:
        """Get the precise standard deviation of the last N points (not rounded)"""
        valid_data = self.interpolated_filtered[self.statistics_reset_index:]
        
        # Require at least 2 points for meaningful std dev
        if len(valid_data) < 2:
            return None
        
        actual_window_size = min(len(valid_data), window_size)
        window_data = valid_data[-actual_window_size:]
        
        mean = sum(window_data) / len(window_data)
        variance = sum((x - mean) ** 2 for x in window_data) / len(window_data)
        return variance ** 0.5
    
    def clear_average_values(self):
        """Clear only average values, marking current position as reset point"""
        # Clear cumulative average tracking
        self.cumulative_average = None
        self.raw_angles_count = 0
        self.batch_buffer.clear()
        self.cumulative_sum_of_squares = 0.0
        
        # Mark the current position in interpolated data as reset point
        # Window averages will only use frames from this index onwards
        self.statistics_reset_index = len(self.interpolated_filtered)
    
    def clear_all_data(self):
        """Clear all data including raw, smoothed, interpolated, and averages"""
        self.timestamps.clear()
        self.angles.clear()
        self.angles_smoothed.clear()
        self.confidences.clear()
        self.interpolated_60fps.clear()
        self.interpolated_filtered.clear()
        self.cumulative_average = None
        self.raw_angles_count = 0
        self.batch_buffer.clear()
        self.last_interpolated_index = 0
        self.last_interpolated_timestamp = 0.0
        self.filter.reset()
        self.interpolated_filter.reset()

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

# Track pause state for statistics reset
pause_flag = False  # True when video is paused, False when playing

# Track exercise change gestures (Thumb_Up -> Thumb_Down within 3 seconds)
thumbs_up_timestamp = None  # Timestamp of LAST (latest) Thumb_Up detected
EXERCISE_CHANGE_WINDOW = 3.0  # seconds - window to detect Thumb_Down after Thumb_Up
exercise_change_count = 0  # Counter for number of exercise changes

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
        'sample_values': {},
        'exercise_change_detected': False
    }
    
    # Process each frame
    for frame in frames:
        frame_timestamp = frame.get('t', datetime.now().timestamp()) / 1000.0  # Convert ms to seconds
        stats['frames_processed'] += 1
        
        # === EXERCISE CHANGE DETECTION via Gesture Sequence ===
        gesture = frame.get('gest')
        
        global thumbs_up_timestamp, exercise_change_count
        
        if gesture == "Thumb_Up":
            # Store the LAST (latest) Thumb_Up timestamp
            thumbs_up_timestamp = frame_timestamp
            print(f"[GESTURE] Thumb_Up detected at {frame_timestamp}, waiting for Thumb_Down within 3 seconds")
        
        elif gesture == "Thumb_Down" and thumbs_up_timestamp is not None:
            # Check if Thumb_Down is within 3 seconds of the last Thumb_Up
            elapsed = frame_timestamp - thumbs_up_timestamp
            
            if elapsed <= EXERCISE_CHANGE_WINDOW:
                # EXERCISE CHANGE DETECTED - Reset all statistics
                print("\n" + "="*80)
                print(f"🔄 EXERCISE CHANGE DETECTED!")
                print(f"   Gesture Sequence: Thumb_Up → Thumb_Down ({elapsed:.2f}s)")
                print(f"   Action: Resetting all statistics")
                print(f"   New Exercise Session: #{exercise_change_count + 1}")
                print("="*80 + "\n")
                
                for store in trainer_stores.values():
                    store.clear_average_values()
                for store in user_stores.values():
                    store.clear_average_values()
                
                exercise_change_count += 1
                thumbs_up_timestamp = None
                stats['exercise_change_detected'] = True
                stats['exercise_change_number'] = exercise_change_count
            else:
                # Thumb_Down came too late, reset waiting state
                thumbs_up_timestamp = None
        
        # Check if Thumb_Up gesture sequence expired
        if thumbs_up_timestamp is not None and (frame_timestamp - thumbs_up_timestamp) > EXERCISE_CHANGE_WINDOW:
            thumbs_up_timestamp = None
        
        # Extract isPlaying from frame and detect pause state transitions
        is_playing = frame.get('isPlaying', True)
        
        global pause_flag
        
        if not is_playing and not pause_flag:
            # Video just paused - clear average values
            pause_flag = True
            print("[PAUSE] Video paused - clearing average values")
            for store in trainer_stores.values():
                store.clear_average_values()
            for store in user_stores.values():
                store.clear_average_values()
            thumbs_up_timestamp = None  # Reset gesture state
            stats['statistics_reset_at_frame'] = stats['frames_processed']
        
        elif is_playing and pause_flag:
            # Video just resumed - clear average values IMMEDIATELY before processing this frame
            pause_flag = False
            print("[RESUME] Video resumed - clearing average values only")
            for store in trainer_stores.values():
                store.clear_average_values()
            for store in user_stores.values():
                store.clear_average_values()
            thumbs_up_timestamp = None  # Reset gesture state
            stats['statistics_reset_at_frame'] = stats['frames_processed']
        
        # Process each joint key in the frame
        for key, value in frame.items():
            if key in ('t', 'gest', 'isPlaying'):  # Skip timestamp, gesture, and isPlaying
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
    
    # Finalize batch: Update cumulative averages for all joints
    for store in trainer_stores.values():
        store.finalize_batch()
    for store in user_stores.values():
        store.finalize_batch()
    
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

@app.get("/statistics")
async def get_statistics():
    """Get cumulative statistics (average & std dev) for all joints (calculated from raw data)"""
    return {
        "timestamp": datetime.now().isoformat(),
        "trainer_statistics": {
            joint_name: {
                "joint_name": joint_name,
                "cumulative_average": store.get_cumulative_average(),
                "cumulative_average_precise": store.get_cumulative_average_precise(),
                "cumulative_std_dev": store.get_cumulative_std_dev(),
                "cumulative_std_dev_precise": store.get_cumulative_std_dev_precise(),
                "total_data_points": store.raw_angles_count
            }
            for joint_name, store in trainer_stores.items()
        },
        "user_statistics": {
            joint_name: {
                "joint_name": joint_name,
                "cumulative_average": store.get_cumulative_average(),
                "cumulative_average_precise": store.get_cumulative_average_precise(),
                "cumulative_std_dev": store.get_cumulative_std_dev(),
                "cumulative_std_dev_precise": store.get_cumulative_std_dev_precise(),
                "total_data_points": store.raw_angles_count
            }
            for joint_name, store in user_stores.items()
        }
    }

@app.get("/last-stats")
async def get_last_stats(window_size: int = 120):
    """
    Get window statistics (average & std dev) for all joints (calculated from filtered interpolated data)
    
    Args:
        window_size: Number of interpolated points to average (default: 120)
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "window_size": window_size,
        "trainer_last_average": {
            joint_name: {
                "joint_name": joint_name,
                "window_average": store.get_window_average_rounded(window_size),
                "window_average_precise": store.get_window_average(window_size),
                "window_std_dev": store.get_window_std_dev(window_size),
                "window_std_dev_precise": store.get_window_std_dev_precise(window_size),
                "elements_used": store.get_window_average_count(window_size),
                "available_data_points": len(store.interpolated_filtered)
            }
            for joint_name, store in trainer_stores.items()
        },
        "user_last_average": {
            joint_name: {
                "joint_name": joint_name,
                "window_average": store.get_window_average_rounded(window_size),
                "window_average_precise": store.get_window_average(window_size),
                "window_std_dev": store.get_window_std_dev(window_size),
                "window_std_dev_precise": store.get_window_std_dev_precise(window_size),
                "elements_used": store.get_window_average_count(window_size),
                "available_data_points": len(store.interpolated_filtered)
            }
            for joint_name, store in user_stores.items()
        }
    }

def calculate_primary_joints(stores_dict: dict) -> dict:
    """
    Calculate primary joints based on standard deviation analysis.
    
    Algorithm:
    1. Get std dev for all joints (excluding Left Wrist and Right Wrist)
    2. Sort by std dev in descending order
    3. Calculate differences between consecutive std devs
    4. Find the maximum difference
    5. All joints above (before) that max difference are considered primary
    
    Returns:
        dict with 'primary_joints' list and 'all_joints_sorted' list with details
    """
    # Joints to ignore in primary joint analysis
    ignored_joints = {"Left Wrist", "Right Wrist"}
    
    # Collect std dev for all joints
    joint_std_devs = []
    for joint_name, store in stores_dict.items():
        # Skip ignored joints
        if joint_name in ignored_joints:
            continue
            
        std_dev = store.get_cumulative_std_dev_precise()
        if std_dev is not None:
            joint_std_devs.append({
                "joint_name": joint_name,
                "std_dev": std_dev,
                "cumulative_average": store.get_cumulative_average_precise(),
                "data_points": store.raw_angles_count
            })
    
    # Sort by std dev descending
    joint_std_devs.sort(key=lambda x: x["std_dev"], reverse=True)
    
    # Calculate differences between consecutive std devs
    differences = []
    for i in range(len(joint_std_devs) - 1):
        diff = joint_std_devs[i]["std_dev"] - joint_std_devs[i + 1]["std_dev"]
        differences.append({
            "index": i,
            "diff": diff,
            "from_joint": joint_std_devs[i]["joint_name"],
            "to_joint": joint_std_devs[i + 1]["joint_name"],
            "from_std_dev": joint_std_devs[i]["std_dev"],
            "to_std_dev": joint_std_devs[i + 1]["std_dev"]
        })
    
    # Find maximum difference
    primary_joints = []
    max_diff_info = None
    
    if differences:
        max_diff_info = max(differences, key=lambda x: x["diff"])
        cutoff_index = max_diff_info["index"]
        
        # All joints up to and including the cutoff_index are primary
        primary_joints = [joint["joint_name"] for joint in joint_std_devs[:cutoff_index + 1]]
    
    return {
        "primary_joints": primary_joints,
        "primary_count": len(primary_joints),
        "total_joints": len(joint_std_devs),
        "all_joints_sorted": joint_std_devs,
        "max_difference": max_diff_info,
        "all_differences": differences
    }

@app.get("/primary-joints")
async def get_primary_joints():
    """
    Analyze and return primary joints for both trainer and user based on standard deviation.
    Primary joints are those with significantly higher std dev (indicating actual movement).
    """
    trainer_analysis = calculate_primary_joints(trainer_stores)
    user_analysis = calculate_primary_joints(user_stores)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "trainer": trainer_analysis,
        "user": user_analysis,
        "summary": {
            "trainer_primary_count": trainer_analysis["primary_count"],
            "trainer_total_joints": trainer_analysis["total_joints"],
            "user_primary_count": user_analysis["primary_count"],
            "user_total_joints": user_analysis["total_joints"]
        }
    }

@app.get("/exercise-info")
async def get_exercise_info():
    """Get current exercise change tracking information"""
    import time
    
    time_remaining = None
    if thumbs_up_timestamp is not None:
        current_time = time.time()
        elapsed = current_time - thumbs_up_timestamp
        if elapsed <= EXERCISE_CHANGE_WINDOW:
            time_remaining = EXERCISE_CHANGE_WINDOW - elapsed
    
    return {
        "timestamp": datetime.now().isoformat(),
        "exercise_change_count": exercise_change_count,
        "thumbs_up_waiting": thumbs_up_timestamp is not None,
        "time_remaining": round(time_remaining, 2) if time_remaining is not None else None,
        "exercise_change_window": EXERCISE_CHANGE_WINDOW
    }

@app.get("/live-primary-joints", response_class=HTMLResponse)
async def get_live_primary_joints():
    """HTML page showing live primary joints analysis with auto-refresh"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Primary Joints Analysis</title>
    </head>
    <body>
        <h1>Live Primary Joints Analysis</h1>
        <p id="updateStatus">Loading...</p>
        
        <h2>Trainer Primary Joints</h2>
        <p id="trainerSummary">Loading...</p>
        <table id="trainerTable" border="1">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Joint Name</th>
                    <th>Std Dev</th>
                    <th>Cumulative Avg</th>
                    <th>Data Points</th>
                    <th>Is Primary</th>
                </tr>
            </thead>
            <tbody id="trainerBody">
                <tr><td colspan="6">Loading...</td></tr>
            </tbody>
        </table>
        
        <h2>User Primary Joints</h2>
        <p id="userSummary">Loading...</p>
        <table id="userTable" border="1">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Joint Name</th>
                    <th>Std Dev</th>
                    <th>Cumulative Avg</th>
                    <th>Data Points</th>
                    <th>Is Primary</th>
                </tr>
            </thead>
            <tbody id="userBody">
                <tr><td colspan="6">Loading...</td></tr>
            </tbody>
        </table>
        
        <h2>Max Difference Analysis</h2>
        <h3>Trainer</h3>
        <table id="trainerMaxDiff" border="1">
            <thead>
                <tr>
                    <th>From Joint</th>
                    <th>From Std Dev</th>
                    <th>To Joint</th>
                    <th>To Std Dev</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody id="trainerMaxDiffBody">
                <tr><td colspan="5">Loading...</td></tr>
            </tbody>
        </table>
        
        <h3>User</h3>
        <table id="userMaxDiff" border="1">
            <thead>
                <tr>
                    <th>From Joint</th>
                    <th>From Std Dev</th>
                    <th>To Joint</th>
                    <th>To Std Dev</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody id="userMaxDiffBody">
                <tr><td colspan="5">Loading...</td></tr>
            </tbody>
        </table>

        <script>
            let updateCount = 0;
            
            function formatValue(val) {
                if (val === null || val === undefined) return '—';
                if (typeof val === 'number') return val.toFixed(2);
                return val;
            }
            
            async function updatePrimaryJoints() {
                try {
                    const response = await fetch('/primary-joints');
                    const data = await response.json();
                    
                    // Update trainer summary
                    document.getElementById('trainerSummary').textContent = 
                        `Primary Joints: ${data.trainer.primary_count} / ${data.trainer.total_joints}`;
                    
                    // Update user summary
                    document.getElementById('userSummary').textContent = 
                        `Primary Joints: ${data.user.primary_count} / ${data.user.total_joints}`;
                    
                    // Update trainer table
                    let trainerHtml = '';
                    data.trainer.all_joints_sorted.forEach((joint, index) => {
                        const isPrimary = data.trainer.primary_joints.includes(joint.joint_name);
                        trainerHtml += `
                            <tr>
                                <td>${index + 1}</td>
                                <td>${joint.joint_name}</td>
                                <td>${formatValue(joint.std_dev)}</td>
                                <td>${formatValue(joint.cumulative_average)}</td>
                                <td>${joint.data_points}</td>
                                <td>${isPrimary ? '✓ PRIMARY' : ''}</td>
                            </tr>
                        `;
                    });
                    document.getElementById('trainerBody').innerHTML = trainerHtml;
                    
                    // Update user table
                    let userHtml = '';
                    data.user.all_joints_sorted.forEach((joint, index) => {
                        const isPrimary = data.user.primary_joints.includes(joint.joint_name);
                        userHtml += `
                            <tr>
                                <td>${index + 1}</td>
                                <td>${joint.joint_name}</td>
                                <td>${formatValue(joint.std_dev)}</td>
                                <td>${formatValue(joint.cumulative_average)}</td>
                                <td>${joint.data_points}</td>
                                <td>${isPrimary ? '✓ PRIMARY' : ''}</td>
                            </tr>
                        `;
                    });
                    document.getElementById('userBody').innerHTML = userHtml;
                    
                    // Update trainer max difference
                    if (data.trainer.max_difference) {
                        const md = data.trainer.max_difference;
                        document.getElementById('trainerMaxDiffBody').innerHTML = `
                            <tr>
                                <td>${md.from_joint}</td>
                                <td>${formatValue(md.from_std_dev)}</td>
                                <td>${md.to_joint}</td>
                                <td>${formatValue(md.to_std_dev)}</td>
                                <td>${formatValue(md.diff)}</td>
                            </tr>
                        `;
                    } else {
                        document.getElementById('trainerMaxDiffBody').innerHTML = 
                            '<tr><td colspan="5">No difference data</td></tr>';
                    }
                    
                    // Update user max difference
                    if (data.user.max_difference) {
                        const md = data.user.max_difference;
                        document.getElementById('userMaxDiffBody').innerHTML = `
                            <tr>
                                <td>${md.from_joint}</td>
                                <td>${formatValue(md.from_std_dev)}</td>
                                <td>${md.to_joint}</td>
                                <td>${formatValue(md.to_std_dev)}</td>
                                <td>${formatValue(md.diff)}</td>
                            </tr>
                        `;
                    } else {
                        document.getElementById('userMaxDiffBody').innerHTML = 
                            '<tr><td colspan="5">No difference data</td></tr>';
                    }
                    
                    updateCount++;
                    document.getElementById('updateStatus').textContent = 
                        `Last updated: ${new Date().toLocaleTimeString()} (Update #${updateCount})`;
                    
                } catch (error) {
                    document.getElementById('updateStatus').textContent = 
                        `Error: ${error.message}`;
                }
            }
            
            // Initial update
            updatePrimaryJoints();
            
            // Update every 1 second
            setInterval(updatePrimaryJoints, 1000);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/live-statistics", response_class=HTMLResponse)
async def get_live_statistics(window_size: int = 120):
    """HTML page showing live statistics table that updates every 500ms"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Statistics - Posture Analysis</title>
    </head>
    <body>
        <h1>Live Statistics Monitor</h1>
        <p>Window Size: {window_size} frames | Auto-refresh: 500ms</p>
        <p id="updateStatus">Connecting...</p>
        
        <table id="statsTable" border="1">
            <thead>
                <tr>
                    <th>Joint Name</th>
                    <th>Role</th>
                    <th>Cumulative Avg</th>
                    <th>Cumulative Std Dev</th>
                    <th>Window Avg (Last {window_size})</th>
                    <th>Window Std Dev</th>
                    <th>Data Points (Total/Window)</th>
                </tr>
            </thead>
            <tbody id="statsBody">
                <tr><td colspan="7">Loading data...</td></tr>
            </tbody>
        </table>

        <script>
            const WINDOW_SIZE = {window_size};
            let updateCount = 0;
            
            function formatValue(value, decimals = 1) {{
                if (value === null || value === undefined) {{
                    return '—';
                }}
                return value.toFixed(decimals);
            }}
            
            function formatValueWithPrecise(value, preciseValue) {{
                if (value === null || value === undefined) {{
                    return '—';
                }}
                let text = value;
                if (preciseValue !== null && preciseValue !== undefined) {{
                    text += ` (${{preciseValue.toFixed(4)}})`;
                }}
                return text;
            }}
            
            async function updateStatistics() {{
                try {{
                    // Fetch both endpoints in parallel
                    const [cumulativeRes, windowRes] = await Promise.all([
                        fetch('/statistics'),
                        fetch(`/last-stats?window_size=${{WINDOW_SIZE}}`)
                    ]);
                    
                    const cumulativeData = await cumulativeRes.json();
                    const windowData = await windowRes.json();
                    
                    // Build table rows
                    let html = '';
                    
                    // Process trainer data
                    const trainerStats = cumulativeData.trainer_statistics;
                    const trainerWindow = windowData.trainer_last_average;
                    
                    Object.keys(trainerStats).forEach(jointName => {{
                        const cumStats = trainerStats[jointName];
                        const winStats = trainerWindow[jointName];
                        
                        html += `
                            <tr>
                                <td>${{jointName}}</td>
                                <td>TRAINER</td>
                                <td>${{formatValueWithPrecise(cumStats.cumulative_average, cumStats.cumulative_average_precise)}}</td>
                                <td>${{formatValueWithPrecise(cumStats.cumulative_std_dev, cumStats.cumulative_std_dev_precise)}}</td>
                                <td>${{formatValueWithPrecise(winStats.window_average, winStats.window_average_precise)}}</td>
                                <td>${{formatValueWithPrecise(winStats.window_std_dev, winStats.window_std_dev_precise)}}</td>
                                <td>${{cumStats.total_data_points}} / ${{winStats.elements_used}}</td>
                            </tr>
                        `;
                    }});
                    
                    // Process user data
                    const userStats = cumulativeData.user_statistics;
                    const userWindow = windowData.user_last_average;
                    
                    Object.keys(userStats).forEach(jointName => {{
                        const cumStats = userStats[jointName];
                        const winStats = userWindow[jointName];
                        
                        html += `
                            <tr>
                                <td>${{jointName}}</td>
                                <td>USER</td>
                                <td>${{formatValueWithPrecise(cumStats.cumulative_average, cumStats.cumulative_average_precise)}}</td>
                                <td>${{formatValueWithPrecise(cumStats.cumulative_std_dev, cumStats.cumulative_std_dev_precise)}}</td>
                                <td>${{formatValueWithPrecise(winStats.window_average, winStats.window_average_precise)}}</td>
                                <td>${{formatValueWithPrecise(winStats.window_std_dev, winStats.window_std_dev_precise)}}</td>
                                <td>${{cumStats.total_data_points}} / ${{winStats.elements_used}}</td>
                            </tr>
                        `;
                    }});
                    
                    document.getElementById('statsBody').innerHTML = html;
                    
                    updateCount++;
                    document.getElementById('updateStatus').textContent = 
                        `Live | Updates: ${{updateCount}} | Last: ${{new Date().toLocaleTimeString()}}`;
                    
                }} catch (error) {{
                    console.error('Error fetching statistics:', error);
                    document.getElementById('updateStatus').textContent = 'Connection Error';
                }}
            }}
            
            // Initial update
            updateStatistics();
            
            // Update every 500ms
            setInterval(updateStatistics, 500);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

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
