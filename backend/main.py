from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import json
import os
from contextlib import asynccontextmanager
import statistics


class JointDataStore:
    """Class to store and manage data for a single joint (trainer or user)."""
    
    def __init__(self, joint_name):
        self.joint_name = joint_name
        self.data = []  # Flat array to store angle values only
    
    def add_data_point(self, angle):
        """Add a single angle value to the array."""
        self.data.append(angle)
    
    def get_all_data(self):
        """Return all data points."""
        return self.data
    
    def get_last_n_points(self, n):
        """Return the last N data points."""
        return self.data[-n:] if len(self.data) >= n else self.data
    
    def find_std_deviation(self, seconds=None):
        """Calculate standard deviation.
        
        Args:
            seconds: If None, use all data. Otherwise, use last seconds*60 elements.
        
        Returns:
            Standard deviation of the specified data range, or None if insufficient data.
        """
        if seconds is None:
            # Use all data
            data_to_analyze = self.data
        else:
            # Use last seconds*60 elements
            num_elements = seconds * 60
            data_to_analyze = self.data[-num_elements:] if len(self.data) >= num_elements else self.data
        
        if len(data_to_analyze) < 2:
            return None
        
        return statistics.stdev(data_to_analyze)
    
    def find_average(self, seconds=None):
        """Calculate average/mean.
        
        Args:
            seconds: If None, use all data. Otherwise, use last seconds*60 elements.
        
        Returns:
            Average of the specified data range, or -1 if insufficient data.
        """
        if seconds is None:
            # Use all data
            data_to_analyze = self.data
        else:
            # Use last seconds*60 elements
            num_elements = seconds * 60
            if len(self.data) < num_elements:
                return -1
            data_to_analyze = self.data[-num_elements:]
        
        if len(data_to_analyze) == 0:
            return -1
        
        return statistics.mean(data_to_analyze)
    
    def count_reps(self):
        """Count repetitions in the angle data.
        
        Returns:
            Number of repetitions detected in the signal.
        
        TODO: Implement peak detection or threshold crossing algorithm.
        """
        # Placeholder implementation
        # TODO: Implement actual rep counting algorithm (e.g., peak detection)
        return 0
    
    def display_all_data(self):
        """Display all data points in terminal."""
        print(f"\n{'='*80}")
        print(f"Joint: {self.joint_name}")
        print(f"Total points: {len(self.data)}")
        print(f"{'='*80}")
        if self.data:
            print(f"{'Index':<8} {'Angle':<15}")
            print('-' * 80)
            for i, angle in enumerate(self.data):
                print(f"{i:<8} {angle:<15.2f}")
        else:
            print("No data available")
        print(f"{'='*80}\n")
    
    def display_last_n_points(self, n):
        """Display the last N data points in terminal."""
        last_points = self.get_last_n_points(n)
        print(f"\n{'='*80}")
        print(f"Joint: {self.joint_name}")
        print(f"Showing last {len(last_points)} of {len(self.data)} points")
        print(f"{'='*80}")
        if last_points:
            print(f"{'Index':<8} {'Angle':<15}")
            print('-' * 80)
            start_idx = len(self.data) - len(last_points)
            for i, angle in enumerate(last_points, start=start_idx):
                print(f"{i:<8} {angle:<15.2f}")
        else:
            print("No data available")
        print(f"{'='*80}\n")
    
    def clear_data(self):
        """Clear all data points."""
        self.data.clear()
    
    def get_data_count(self):
        """Return the number of data points stored."""
        return len(self.data)


# List of all joint names
JOINT_NAMES = [
    "Left Shoulder", "Left Elbow", "Left Wrist",
    "Right Shoulder", "Right Elbow", "Right Wrist",
    "Left Hip", "Left Knee", "Left Ankle",
    "Right Hip", "Right Knee", "Right Ankle",
    "Left Spine", "Right Spine", "Neck"
]

# Initialize separate data stores for trainer and user
trainer_joint_data_stores = {f"t_{joint_name}": JointDataStore(f"t_{joint_name}") for joint_name in JOINT_NAMES}
user_joint_data_stores = {f"u_{joint_name}": JointDataStore(f"u_{joint_name}") for joint_name in JOINT_NAMES}


class WorkoutAnalyzer:
    """Class for analyzing data across multiple joints."""
    
    def __init__(self, trainer_stores, user_stores):
        self.trainer_stores = trainer_stores
        self.user_stores = user_stores
    
    def find_primary_angles(self, use_trainer=False, seconds=None):
        """Find primary angles based on standard deviation analysis.
        
        Algorithm:
        1. Calculate std deviation for all joints (excluding wrists)
        2. Auto-include joints with std deviation > 80 (high movement)
        3. Sort remaining joints by std deviation in descending order
        4. Find max difference between consecutive joints
        5. All joints above the max difference are considered primary angles
        
        Args:
            use_trainer: If True, analyze trainer data. Otherwise, analyze user data.
            seconds: If provided, only consider data from the last N seconds.
                    If None, use all available data.
        
        Returns:
            List of joint names that are primary angles.
        """
        stores = self.trainer_stores if use_trainer else self.user_stores
        
        # Calculate std deviation for all joints (excluding wrists)
        joint_std_pairs = []
        high_movement_joints = []  # Joints with std > 80
        
        for joint_name, store in stores.items():
            std_dev = store.find_std_deviation(seconds=seconds)
            if std_dev is not None:
                # Exclude wrist angles from primary angle detection
                if "Wrist" in joint_name:
                    continue
                
                # Auto-include joints with very high movement (std > 80)
                if std_dev > 80:
                    high_movement_joints.append(joint_name)
                else:
                    joint_std_pairs.append((joint_name, std_dev))
        
        # If we only have high movement joints, return them
        if len(joint_std_pairs) < 2:
            return high_movement_joints
        
        # Sort by std deviation in descending order
        joint_std_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Find max difference between consecutive joints
        max_diff = 0
        max_diff_index = 0
        for i in range(len(joint_std_pairs) - 1):
            diff = joint_std_pairs[i][1] - joint_std_pairs[i + 1][1]
            if diff > max_diff:
                max_diff = diff
                max_diff_index = i
        
        # All joints above (and including) the max difference index are primary
        primary_angles = [joint_name for joint_name, _ in joint_std_pairs[:max_diff_index + 1]]
        
        # Combine high movement joints with detected primary angles
        return high_movement_joints + primary_angles
    
    def get_all_joint_std_deviations(self, use_trainer=False, seconds=None):
        """Get standard deviation for all joints sorted by value.
        
        Args:
            use_trainer: If True, analyze trainer data. Otherwise, analyze user data.
            seconds: If provided, only consider data from the last N seconds.
        
        Returns:
            List of tuples (joint_name, std_dev) sorted by std_dev descending.
        """
        stores = self.trainer_stores if use_trainer else self.user_stores
        
        joint_std_list = []
        for joint_name, store in stores.items():
            std_dev = store.find_std_deviation(seconds=seconds)
            if std_dev is not None:
                joint_std_list.append((joint_name, std_dev))
        
        # Sort by std deviation in descending order
        joint_std_list.sort(key=lambda x: x[1], reverse=True)
        
        return joint_std_list
    
    def verify_reps_across_primary_angles(self, use_trainer=False, seconds=None):
        """Verify repetition counts across primary angles.
        
        This method combines rep data from all primary angles to validate
        and provide a consensus rep count.
        
        Args:
            use_trainer: If True, analyze trainer data. Otherwise, analyze user data.
            seconds: If provided, only consider data from the last N seconds.
        
        Returns:
            Dictionary with verified rep count and per-joint breakdown.
        
        TODO: Implement cross-joint rep verification algorithm.
        """
        primary_angles = self.find_primary_angles(use_trainer=use_trainer, seconds=seconds)
        stores = self.trainer_stores if use_trainer else self.user_stores
        
        # Get rep counts from each primary angle
        rep_counts = {}
        for angle_name in primary_angles:
            # Add prefix back to get the store key
            prefix = 't_' if use_trainer else 'u_'
            store_key = f"{prefix}{angle_name}"
            if store_key in stores:
                rep_counts[angle_name] = stores[store_key].count_reps()
        
        # Placeholder implementation
        # TODO: Implement consensus algorithm (e.g., median, voting, temporal alignment)
        verified_count = 0
        if rep_counts:
            verified_count = max(rep_counts.values()) if rep_counts else 0
        
        return {
            "verified_count": verified_count,
            "primary_angles": primary_angles,
            "individual_counts": rep_counts
        }


# Initialize workout analyzer
workout_analyzer = WorkoutAnalyzer(trainer_joint_data_stores, user_joint_data_stores)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("✅ In-memory data storage initialized")
    yield
    

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store all connected WebSocket clients
active_connections: list[WebSocket] = []

@app.get("/")
def read_root():
    return RedirectResponse(url="/posture")

@app.get("/posture", response_class=HTMLResponse)
async def get_posture_page():
    # Clear all data for fresh workout session
    for store in list(trainer_joint_data_stores.values()) + list(user_joint_data_stores.values()):
        store.clear_data()
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Posture Correction - Live Angles (uPlot)</title>
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
            .status {
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
            }
            .connected {
                background: #d4edda;
                color: #155724;
            }
            .disconnected {
                background: #f8d7da;
                color: #721c24;
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
            .filter-info {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
                font-size: 14px;
            }
            .filter-info strong {
                color: #1976d2;
            }
            .filter-info code {
                background: #fff;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                color: #d32f2f;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Posture Correction - Live Joint Angles (uPlot - Fast)</h1>
            <div id="status" class="status disconnected">Connecting...</div>
            <div id="filter-info" class="filter-info"></div>
            
            <div class="charts-grid" id="charts-container"></div>
            
            <h2 style="margin-top: 40px; text-align: center; color: #333;">60 FPS Interpolated Data (Database Ready)</h2>
            <div class="info" style="margin-bottom: 20px;">
                <strong>Interpolation:</strong> Smoothed angles resampled to exactly 60 frames per second using linear interpolation<br>
                <strong>Purpose:</strong> Consistent frame rate for database storage and analysis<br>
                <strong>Source:</strong> Applied to filtered/smoothed data (not raw data)
            </div>
            <div class="charts-grid" id="interpolated-charts-container"></div>
        </div>

        <script>
            /*
            ============================================
            POSTURE CORRECTION - ADVANCED SMOOTHING SYSTEM
            ============================================
            
            This visualization system applies advanced signal processing filters to smooth
            noisy pose detection data while preserving important movement characteristics.
            
            AVAILABLE FILTERS:
            
            1. SAVITZKY-GOLAY (savgol)
               - Method: Polynomial fitting in sliding window
               - Pros: Preserves peaks, valleys, and derivatives
               - Cons: Requires sufficient data points
               - Best for: Exercise movements where peak angles matter
               - Window sizes: 5 (fast), 7 (balanced), 9 (smooth)
            
            2. EXPONENTIAL MOVING AVERAGE (ema)
               - Method: Weighted average favoring recent data
               - Pros: Fast response, minimal lag, simple
               - Cons: Less noise reduction than other methods
               - Best for: Real-time applications requiring quick response
               - Alpha: 0.1-0.3 (smooth), 0.4-0.5 (responsive)
            
            3. DOUBLE EXPONENTIAL SMOOTHING (double_ema)
               - Method: EMA with trend component (Holt's method)
               - Pros: Handles linear trends, predicts next value
               - Cons: More complex, can overshoot
               - Best for: Data with gradual trends (e.g., slow movements)
               - Alpha: level smoothing, Beta: trend smoothing
            
            4. GAUSSIAN SMOOTHING (gaussian)
               - Method: Weighted average with Gaussian kernel
               - Pros: Heavy noise reduction, smooth curves
               - Cons: More lag, blurs sharp features
               - Best for: Very noisy data requiring heavy smoothing
               - Sigma: controls spread (1.0-2.0 typical)
            
            5. KALMAN FILTER (kalman)
               - Method: Optimal estimation with prediction and update
               - Pros: Theoretically optimal, handles uncertainty
               - Cons: Requires tuning, assumes linear dynamics
               - Best for: Real-time tracking with sensor noise
               - Process noise: system uncertainty, Measurement noise: sensor uncertainty
            
            6. HYBRID FILTER (hybrid) - RECOMMENDED
               - Method: Multi-stage pipeline combining best features
               - Stage 1: Kalman filter (noise reduction)
               - Stage 2: Savitzky-Golay (shape preservation)
               - Stage 3: Light EMA (final smoothing)
               - Pros: Best overall quality, balanced performance
               - Cons: More computational cost
               - Best for: Production use
            
            HOW TO CHANGE FILTERS:
            1. Modify SMOOTHING_CONFIG.filterType below
            2. Adjust filter-specific parameters
            3. Reload the page to see changes
            
            ============================================
            */
            
            // ============================================
            // CONFIGURATION - Adjust these settings
            // ============================================
            const SMOOTHING_CONFIG = {
                // Choose filter type: 'savgol', 'ema', 'double_ema', 'gaussian', 'kalman', 'hybrid'
                filterType: 'guassian',
                
                // Savitzky-Golay settings
                savgol_windowSize: 7,  // 5, 7, or 9 (larger = smoother but more lag)
                
                // EMA settings
                ema_alpha: 0.3,  // 0.1-0.5 (lower = smoother, higher = more responsive)
                
                // Double EMA settings
                double_ema_alpha: 0.3,
                double_ema_beta: 0.1,
                
                // Gaussian settings
                gaussian_windowSize: 7,
                gaussian_sigma: 1.5,
                
                // Kalman settings
                kalman_processNoise: 0.01,
                kalman_measurementNoise: 0.1,
                
                // Smoothing frequency (ms)
                smoothingInterval: 500  // How often to apply smoothing (500ms = 2x per second)
            };
            
            const statusDiv = document.getElementById('status');
            const chartsContainer = document.getElementById('charts-container');
            const filterInfoDiv = document.getElementById('filter-info');
            const maxDataPoints = 300; // Keep last 300 data points (~10 seconds at 30fps)
            
            // Display active filter configuration
            function displayFilterInfo() {
                const filterDescriptions = {
                    'savgol': 'Preserves peaks and valleys, best for exercise movements',
                    'ema': 'Fast response with minimal lag, good for real-time',
                    'double_ema': 'Handles trends well, good for gradual movements',
                    'gaussian': 'Heavy noise reduction with smooth curves',
                    'kalman': 'Optimal for noisy sensor data with prediction',
                    'hybrid': 'Best overall quality - combines Kalman, Savitzky-Golay, and EMA'
                };
                
                filterInfoDiv.innerHTML = `
                    <strong>🔧 Active Smoothing Filter:</strong> <code>${SMOOTHING_CONFIG.filterType.toUpperCase()}</code><br>
                    <strong>Description:</strong> ${filterDescriptions[SMOOTHING_CONFIG.filterType]}<br>
                    <strong>Update Frequency:</strong> Every ${SMOOTHING_CONFIG.smoothingInterval}ms
                `;
            }
            
            displayFilterInfo();
            
            // ============================================
            // SMOOTHING FILTERS
            // ============================================
            
            // 1. Savitzky-Golay filter (preserves peaks and shape)
            // Best for: Exercise movements where peak angles matter
            function savitzkyGolayFilter(data, windowSize = 7) {
                if (!data || data.length < windowSize) return data;
                
                // Savitzky-Golay coefficients for different window sizes
                const coefficients = {
                    5: [-3, 12, 17, 12, -3].map(c => c / 35),
                    7: [-2, 3, 6, 7, 6, 3, -2].map(c => c / 21),
                    9: [-21, 14, 39, 54, 59, 54, 39, 14, -21].map(c => c / 231)
                };
                
                const sgCoeffs = coefficients[windowSize] || coefficients[7];
                const filtered = [];
                const halfWindow = Math.floor(windowSize / 2);
                
                // For points at the edges, use the point as-is
                for (let i = 0; i < halfWindow; i++) {
                    filtered.push(data[i] === null ? null : data[i]);
                }
                
                // Apply Savitzky-Golay filter to middle points
                for (let i = halfWindow; i < data.length - halfWindow; i++) {
                    let sum = 0;
                    let validPoints = 0;
                    
                    for (let j = 0; j < windowSize; j++) {
                        const idx = i - halfWindow + j;
                        if (data[idx] !== null) {
                            sum += data[idx] * sgCoeffs[j];
                            validPoints++;
                        }
                    }
                    
                    filtered.push(validPoints >= Math.ceil(windowSize * 0.6) ? sum : null);
                }
                
                // For points at the end, use the point as-is
                for (let i = data.length - halfWindow; i < data.length; i++) {
                    filtered.push(data[i] === null ? null : data[i]);
                }
                
                return filtered;
            }
            
            // 2. Exponential Moving Average (EMA) - Fast response with smoothing
            // Best for: Real-time applications with minimal lag
            function exponentialMovingAverage(data, alpha = 0.3) {
                if (!data || data.length === 0) return data;
                
                const filtered = [];
                let ema = null;
                
                for (let i = 0; i < data.length; i++) {
                    if (data[i] !== null) {
                        if (ema === null) {
                            ema = data[i]; // Initialize with first valid value
                        } else {
                            ema = alpha * data[i] + (1 - alpha) * ema;
                        }
                        filtered.push(ema);
                    } else {
                        filtered.push(null);
                    }
                }
                
                return filtered;
            }
            
            // 3. Double Exponential Smoothing (Holt's method) - Handles trends
            // Best for: Data with linear trends (e.g., gradual movement changes)
            function doubleExponentialSmoothing(data, alpha = 0.3, beta = 0.1) {
                if (!data || data.length === 0) return data;
                
                const filtered = [];
                let level = null;
                let trend = 0;
                
                for (let i = 0; i < data.length; i++) {
                    if (data[i] !== null) {
                        if (level === null) {
                            level = data[i];
                            filtered.push(level);
                        } else {
                            const prevLevel = level;
                            level = alpha * data[i] + (1 - alpha) * (level + trend);
                            trend = beta * (level - prevLevel) + (1 - beta) * trend;
                            filtered.push(level + trend);
                        }
                    } else {
                        filtered.push(null);
                    }
                }
                
                return filtered;
            }
            
            // 4. Gaussian Smoothing - Weighted average with Gaussian kernel
            // Best for: Heavy noise reduction with smooth curves
            function gaussianSmoothing(data, windowSize = 7, sigma = 1.5) {
                if (!data || data.length < windowSize) return data;
                
                // Generate Gaussian kernel
                const halfWindow = Math.floor(windowSize / 2);
                const kernel = [];
                let kernelSum = 0;
                
                for (let i = -halfWindow; i <= halfWindow; i++) {
                    const weight = Math.exp(-(i * i) / (2 * sigma * sigma));
                    kernel.push(weight);
                    kernelSum += weight;
                }
                
                // Normalize kernel
                const normalizedKernel = kernel.map(k => k / kernelSum);
                
                const filtered = [];
                
                // Apply Gaussian filter
                for (let i = 0; i < data.length; i++) {
                    if (i < halfWindow || i >= data.length - halfWindow) {
                        filtered.push(data[i]);
                    } else {
                        let sum = 0;
                        let validPoints = 0;
                        
                        for (let j = 0; j < windowSize; j++) {
                            const idx = i - halfWindow + j;
                            if (data[idx] !== null) {
                                sum += data[idx] * normalizedKernel[j];
                                validPoints++;
                            }
                        }
                        
                        filtered.push(validPoints >= Math.ceil(windowSize * 0.6) ? sum : null);
                    }
                }
                
                return filtered;
            }
            
            // 5. Kalman Filter (simplified 1D) - Optimal for noisy sensor data
            // Best for: Real-time tracking with prediction
            function kalmanFilter(data, processNoise = 0.01, measurementNoise = 0.1) {
                if (!data || data.length === 0) return data;
                
                const filtered = [];
                let estimate = null;
                let errorEstimate = 1.0;
                
                for (let i = 0; i < data.length; i++) {
                    if (data[i] !== null) {
                        if (estimate === null) {
                            estimate = data[i];
                            filtered.push(estimate);
                        } else {
                            // Prediction step
                            const predictedEstimate = estimate;
                            const predictedError = errorEstimate + processNoise;
                            
                            // Update step
                            const kalmanGain = predictedError / (predictedError + measurementNoise);
                            estimate = predictedEstimate + kalmanGain * (data[i] - predictedEstimate);
                            errorEstimate = (1 - kalmanGain) * predictedError;
                            
                            filtered.push(estimate);
                        }
                    } else {
                        filtered.push(null);
                    }
                }
                
                return filtered;
            }
            
            // 6. Hybrid Filter - Combines multiple filters for best results
            // Best for: Production use - balances all requirements
            function hybridFilter(data) {
                // Step 1: Apply Kalman filter for noise reduction
                let smoothed = kalmanFilter(data, 0.01, 0.1);
                
                // Step 2: Apply Savitzky-Golay to preserve shape
                smoothed = savitzkyGolayFilter(smoothed, 5);
                
                // Step 3: Apply light EMA for final smoothing
                smoothed = exponentialMovingAverage(smoothed, 0.5);
                
                return smoothed;
            }
            
            // ============================================
            // INTERPOLATION FOR 60 FPS
            // ============================================
            
            // Linear interpolation function (from main3.py)
            function linearInterpolate(xOrig, yOrig, targetCount) {
                if (xOrig.length < 2 || targetCount < 2) return yOrig;
                
                const result = [];
                const xMin = xOrig[0];
                const xMax = xOrig[xOrig.length - 1];
                
                for (let i = 0; i < targetCount; i++) {
                    const targetX = xMin + ((xMax - xMin) * i / (targetCount - 1));
                    
                    // Find the two surrounding points
                    let idx = 0;
                    for (let j = 0; j < xOrig.length - 1; j++) {
                        if (xOrig[j] <= targetX && targetX <= xOrig[j + 1]) {
                            idx = j;
                            break;
                        }
                        if (targetX > xOrig[j + 1]) {
                            idx = j + 1;
                        }
                    }
                    
                    // Handle edge cases
                    if (idx >= xOrig.length - 1) {
                        result.push(yOrig[yOrig.length - 1]);
                        continue;
                    }
                    
                    // Linear interpolation
                    const x0 = xOrig[idx];
                    const x1 = xOrig[idx + 1];
                    const y0 = yOrig[idx];
                    const y1 = yOrig[idx + 1];
                    
                    if (x1 - x0 === 0) {
                        result.push(y0);
                    } else {
                        const t = (targetX - x0) / (x1 - x0);
                        const interpolatedY = y0 + t * (y1 - y0);
                        result.push(interpolatedY);
                    }
                }
                
                return result;
            }
            
            // Resample to 60 FPS for a given time range
            function resampleTo60FPS(timestamps, smoothedData) {
                if (timestamps.length < 2) return { timestamps: [], values: [] };
                
                const fps = 60;
                const frameDuration = 1.0 / fps; // 0.01666... seconds per frame
                const firstTimestamp = timestamps[0];
                const lastTimestamp = timestamps[timestamps.length - 1];
                
                // Align start time to the nearest 60 FPS frame boundary at or before the first timestamp
                // Find the whole second, then add frame boundaries
                const wholeSecond = Math.floor(firstTimestamp);
                const offsetWithinSecond = firstTimestamp - wholeSecond;
                const frameIndexWithinSecond = Math.floor(offsetWithinSecond / frameDuration);
                const alignedStartTime = wholeSecond + (frameIndexWithinSecond * frameDuration);
                
                // Generate timestamps at exactly 60 FPS from the aligned start
                const interpolatedTimestamps = [];
                const interpolatedValues = [];
                
                let currentTime = alignedStartTime;
                while (currentTime <= lastTimestamp) {
                    interpolatedTimestamps.push(currentTime);
                    currentTime += frameDuration;
                }
                
                // Use linearInterpolate to get values at these exact timestamps
                const targetCount = interpolatedTimestamps.length;
                if (targetCount > 0) {
                    const values = linearInterpolate(timestamps, smoothedData, targetCount);
                    return {
                        timestamps: interpolatedTimestamps,
                        values: values
                    };
                }
                
                return {
                    timestamps: [],
                    values: []
                };
            }
            
            // Joint mapping
            const joints = {
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
            };
            
            const charts = {};
            const chartData = {};
            let lastSmoothingTime = Date.now();
            
            // Create a chart for each joint using uPlot
            Object.keys(joints).forEach(jointKey => {
                // Create container
                const container = document.createElement('div');
                container.className = 'chart-container';
                container.innerHTML = `<h3>${joints[jointKey]}</h3>`;
                chartsContainer.appendChild(container);
                
                // Initialize data storage
                chartData[jointKey] = {
                    timestamps: [],
                    userAngles: [],
                    trainerAngles: [],
                    userAngleSmoothed: [],
                    trainerAngleSmoothed: [],
                    userConf: [],
                    trainerConf: [],
                    lastValidUserAngle: null,
                    lastValidTrainerAngle: null,
                    interpolated60Timestamps: [],
                    interpolated60UserAngles: [],
                    interpolated60TrainerAngles: []
                };
                
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
                            label: "User Angle",
                            stroke: "#2196F3",
                            width: 2,
                            points: { show: false }
                        },
                        {
                            label: "Trainer Angle",
                            stroke: "#4CAF50",
                            width: 2,
                            points: { show: false }
                        },
                        {
                            label: "User Angle Smoothed",
                            stroke: "#FF6B6B",
                            width: 2.5,
                            points: { show: false },
                            dashArray: [5, 5]
                        },
                        {
                            label: "Trainer Angle Smoothed",
                            stroke: "#FFA500",
                            width: 2.5,
                            points: { show: false },
                            dashArray: [5, 5]
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
                
                // Create chart with initial empty data (5 series: timestamps, user, trainer, user smoothed, trainer smoothed)
                const data = [
                    [0],
                    [0],
                    [0],
                    [0],
                    [0]
                ];
                
                charts[jointKey] = new uPlot(opts, data, container);
            });
            
            // Create interpolated charts (60 FPS)
            const interpolatedCharts = {};
            const interpolatedChartsContainer = document.getElementById('interpolated-charts-container');
            
            Object.keys(joints).forEach(jointKey => {
                // Create container
                const container = document.createElement('div');
                container.className = 'chart-container';
                container.innerHTML = `<h3>${joints[jointKey]} (60 FPS)</h3>`;
                interpolatedChartsContainer.appendChild(container);
                
                // uPlot options for interpolated charts
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
                            label: "User 60 FPS",
                            stroke: "#9C27B0",
                            width: 2,
                            points: { show: false }
                        },
                        {
                            label: "Trainer 60 FPS",
                            stroke: "#FF9800",
                            width: 2,
                            points: { show: false }
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
                
                // Create chart with initial empty data (3 series: timestamps, user 60fps, trainer 60fps)
                const data = [
                    [0],
                    [0],
                    [0]
                ];
                
                interpolatedCharts[jointKey] = new uPlot(opts, data, container);
            });
            
            function updateCharts(data) {
                const timestamp = data.t / 1000; // Convert to seconds for uPlot
                
                // Update each joint
                Object.keys(joints).forEach(jointKey => {
                    const userKey = `u_${jointKey}`;
                    const trainerKey = `tr_${jointKey}`;
                    
                    // Add timestamp
                    chartData[jointKey].timestamps.push(timestamp);
                    
                    // User angle with fallback to previous valid value
                    if (data[userKey]) {
                        const userAngle = data[userKey][0];
                        const userConf = data[userKey][1];
                        
                        if (userConf >= 40) {
                            // Confidence is good, plot new value and save it
                            chartData[jointKey].userAngles.push(userAngle);
                            chartData[jointKey].lastValidUserAngle = userAngle;
                        } else {
                            // Confidence is low, use previous valid value
                            if (chartData[jointKey].lastValidUserAngle !== null) {
                                chartData[jointKey].userAngles.push(chartData[jointKey].lastValidUserAngle);
                            } else {
                                // No previous value yet, use current angle anyway
                                chartData[jointKey].userAngles.push(userAngle);
                                chartData[jointKey].lastValidUserAngle = userAngle;
                            }
                        }
                        chartData[jointKey].userConf.push(userConf);
                    } else {
                        chartData[jointKey].userAngles.push(null);
                        chartData[jointKey].userConf.push(0);
                    }
                    
                    // Trainer angle with fallback to previous valid value
                    if (data[trainerKey]) {
                        const trainerAngle = data[trainerKey][0];
                        const trainerConf = data[trainerKey][1];
                        
                        if (trainerConf >= 40) {
                            // Confidence is good, plot new value and save it
                            chartData[jointKey].trainerAngles.push(trainerAngle);
                            chartData[jointKey].lastValidTrainerAngle = trainerAngle;
                        } else {
                            // Confidence is low, use previous valid value
                            if (chartData[jointKey].lastValidTrainerAngle !== null) {
                                chartData[jointKey].trainerAngles.push(chartData[jointKey].lastValidTrainerAngle);
                            } else {
                                // No previous value yet, use current angle anyway
                                chartData[jointKey].trainerAngles.push(trainerAngle);
                                chartData[jointKey].lastValidTrainerAngle = trainerAngle;
                            }
                        }
                        chartData[jointKey].trainerConf.push(trainerConf);
                    } else {
                        chartData[jointKey].trainerAngles.push(null);
                        chartData[jointKey].trainerConf.push(0);
                    }
                    
                    // Keep only last maxDataPoints
                    if (chartData[jointKey].timestamps.length > maxDataPoints) {
                        chartData[jointKey].timestamps.shift();
                        chartData[jointKey].userAngles.shift();
                        chartData[jointKey].trainerAngles.shift();
                        chartData[jointKey].userAngleSmoothed.shift();
                        chartData[jointKey].trainerAngleSmoothed.shift();
                        chartData[jointKey].userConf.shift();
                        chartData[jointKey].trainerConf.shift();
                    }
                    
                    // Update uPlot with new data
                    charts[jointKey].setData([
                        chartData[jointKey].timestamps,
                        chartData[jointKey].userAngles,
                        chartData[jointKey].trainerAngles,
                        chartData[jointKey].userAngleSmoothed,
                        chartData[jointKey].trainerAngleSmoothed
                    ]);
                });
            }
            
            // WebSocket connection
            console.log('Attempting to connect to WebSocket...');
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            // Queue to store all incoming frames
            let frameQueue = [];
            let isProcessing = false;
            
            ws.onopen = () => {
                console.log('WebSocket connected successfully!');
                statusDiv.textContent = 'Connected - Receiving live data...';
                statusDiv.className = 'status connected';
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.t) {
                        // Add frame to queue
                        frameQueue.push(data);
                        
                        // Start processing if not already processing
                        if (!isProcessing) {
                            processFrameQueue();
                        }
                    }
                } catch (e) {
                    console.error('Failed to parse message:', e);
                }
            };
            
            // Process all frames in the queue
            function processFrameQueue() {
                if (frameQueue.length === 0) {
                    isProcessing = false;
                    return;
                }
                
                isProcessing = true;
                
                // Process all frames in the queue
                const framesToProcess = [...frameQueue];
                frameQueue = [];
                
                console.log(`📊 Processing ${framesToProcess.length} frames`);
                
                framesToProcess.forEach(data => {
                    updateCharts(data);
                });
                
                // Apply smoothing filter every 1 second
                applySmoothing();
                
                // Continue processing if more frames arrived
                if (frameQueue.length > 0) {
                    requestAnimationFrame(processFrameQueue);
                } else {
                    isProcessing = false;
                }
            }
            
            // Apply advanced smoothing filters based on configuration
            function applySmoothing() {
                const now = Date.now();
                
                // Apply smoothing based on configured interval
                if (now - lastSmoothingTime >= SMOOTHING_CONFIG.smoothingInterval) {
                    // Select filter based on configuration (outside the loop)
                    let filterFunction;
                    let filterName;
                    
                    switch(SMOOTHING_CONFIG.filterType) {
                        case 'savgol':
                            filterFunction = (data) => savitzkyGolayFilter(data, SMOOTHING_CONFIG.savgol_windowSize);
                            filterName = `Savitzky-Golay (window=${SMOOTHING_CONFIG.savgol_windowSize})`;
                            break;
                        case 'ema':
                            filterFunction = (data) => exponentialMovingAverage(data, SMOOTHING_CONFIG.ema_alpha);
                            filterName = `EMA (alpha=${SMOOTHING_CONFIG.ema_alpha})`;
                            break;
                        case 'double_ema':
                            filterFunction = (data) => doubleExponentialSmoothing(data, SMOOTHING_CONFIG.double_ema_alpha, SMOOTHING_CONFIG.double_ema_beta);
                            filterName = `Double EMA (alpha=${SMOOTHING_CONFIG.double_ema_alpha}, beta=${SMOOTHING_CONFIG.double_ema_beta})`;
                            break;
                        case 'gaussian':
                            filterFunction = (data) => gaussianSmoothing(data, SMOOTHING_CONFIG.gaussian_windowSize, SMOOTHING_CONFIG.gaussian_sigma);
                            filterName = `Gaussian (window=${SMOOTHING_CONFIG.gaussian_windowSize}, sigma=${SMOOTHING_CONFIG.gaussian_sigma})`;
                            break;
                        case 'kalman':
                            filterFunction = (data) => kalmanFilter(data, SMOOTHING_CONFIG.kalman_processNoise, SMOOTHING_CONFIG.kalman_measurementNoise);
                            filterName = `Kalman (process=${SMOOTHING_CONFIG.kalman_processNoise}, measurement=${SMOOTHING_CONFIG.kalman_measurementNoise})`;
                            break;
                        case 'hybrid':
                        default:
                            filterFunction = hybridFilter;
                            filterName = 'Hybrid (Kalman + SavGol + EMA)';
                            break;
                    }
                    
                    // Apply the selected filter to all joints and collect interpolated data
                    const interpolatedDataByTimestamp = {};
                    
                    Object.keys(joints).forEach(jointKey => {
                        // Apply selected filter to user angles
                        if (chartData[jointKey].userAngles.length > 0) {
                            chartData[jointKey].userAngleSmoothed = filterFunction(
                                chartData[jointKey].userAngles
                            );
                        }
                        
                        // Apply selected filter to trainer angles
                        if (chartData[jointKey].trainerAngles.length > 0) {
                            chartData[jointKey].trainerAngleSmoothed = filterFunction(
                                chartData[jointKey].trainerAngles
                            );
                        }
                        
                        // Update chart with both raw and smoothed data
                        charts[jointKey].setData([
                            chartData[jointKey].timestamps,
                            chartData[jointKey].userAngles,
                            chartData[jointKey].trainerAngles,
                            chartData[jointKey].userAngleSmoothed,
                            chartData[jointKey].trainerAngleSmoothed
                        ]);
                        
                        // Apply 60 FPS interpolation to smoothed data
                        if (chartData[jointKey].userAngleSmoothed.length > 1 && 
                            chartData[jointKey].trainerAngleSmoothed.length > 1) {
                            
                            // Resample user smoothed data to 60 FPS
                            const userResampled = resampleTo60FPS(
                                chartData[jointKey].timestamps,
                                chartData[jointKey].userAngleSmoothed
                            );
                            
                            // Resample trainer smoothed data to 60 FPS
                            const trainerResampled = resampleTo60FPS(
                                chartData[jointKey].timestamps,
                                chartData[jointKey].trainerAngleSmoothed
                            );
                            
                            // Store interpolated data for database export
                            chartData[jointKey].interpolated60Timestamps = userResampled.timestamps;
                            chartData[jointKey].interpolated60UserAngles = userResampled.values;
                            chartData[jointKey].interpolated60TrainerAngles = trainerResampled.values;
                            
                            // Update interpolated chart
                            interpolatedCharts[jointKey].setData([
                                chartData[jointKey].interpolated60Timestamps,
                                chartData[jointKey].interpolated60UserAngles,
                                chartData[jointKey].interpolated60TrainerAngles
                            ]);
                            
                            // Collect data organized by timestamp
                            for (let i = 0; i < userResampled.timestamps.length; i++) {
                                const timestamp = userResampled.timestamps[i];
                                if (!interpolatedDataByTimestamp[timestamp]) {
                                    interpolatedDataByTimestamp[timestamp] = [];
                                }
                                interpolatedDataByTimestamp[timestamp].push({
                                    joint: joints[jointKey],
                                    userAngle: userResampled.values[i],
                                    trainerAngle: trainerResampled.values[i]
                                });
                            }
                        }
                    });
                    
                    // Send data grouped by timestamp (all 15 joints for each timestamp together)
                    Object.keys(interpolatedDataByTimestamp).forEach(timestamp => {
                        const jointsData = interpolatedDataByTimestamp[timestamp];
                        // Send all joints data for this timestamp in one message
                        jointsData.forEach(jointData => {
                            ws.send(JSON.stringify({
                                type: 'interpolated_data',
                                joint: jointData.joint,
                                timestamp: parseFloat(timestamp),
                                userAngle: Math.round(jointData.userAngle),
                                trainerAngle: Math.round(jointData.trainerAngle)
                            }));
                        });
                    });
                    
                    // Send completion notification
                    ws.send(JSON.stringify({
                        type: 'interpolation_complete',
                        timestamp: Date.now() / 1000
                    }));
                    
                    lastSmoothingTime = now;
                    console.log(`✨ ${filterName} filter applied + 60 FPS interpolation + notification sent`);
                }
            }
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                statusDiv.textContent = 'Connection error - Check console';
                statusDiv.className = 'status disconnected';
            };
            
            ws.onclose = () => {
                console.log('WebSocket closed');
                statusDiv.textContent = 'Disconnected';
                statusDiv.className = 'status disconnected';
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/view-joint/{joint_name}")
async def view_joint_data(joint_name: str, role: str = "both", last_n: int = None):
    """
    View data for a specific joint.
    
    Args:
        joint_name: Base joint name (e.g., 'Left Shoulder')
        role: 'trainer', 'user', or 'both' (default)
        last_n: Optional limit on number of points to return
    
    Examples:
    - http://localhost:8000/view-joint/Left Shoulder
    - http://localhost:8000/view-joint/Left Knee?role=user&last_n=50
    """
    trainer_key = f"t_{joint_name}"
    user_key = f"u_{joint_name}"
    
    result = {"joint": joint_name}
    
    if role in ["trainer", "both"]:
        if trainer_key in trainer_joint_data_stores:
            store = trainer_joint_data_stores[trainer_key]
            data = store.get_last_n_points(last_n) if last_n else store.get_all_data()
            result["trainer"] = {
                "count": store.get_data_count(),
                "data": data
            }
        else:
            result["trainer"] = {"error": "Trainer joint not found"}
    
    if role in ["user", "both"]:
        if user_key in user_joint_data_stores:
            store = user_joint_data_stores[user_key]
            data = store.get_last_n_points(last_n) if last_n else store.get_all_data()
            result["user"] = {
                "count": store.get_data_count(),
                "data": data
            }
        else:
            result["user"] = {"error": "User joint not found"}
    
    if role not in ["trainer", "user", "both"]:
        return {
            "error": "Invalid role. Use 'trainer', 'user', or 'both'",
            "available_joints": JOINT_NAMES
        }
    
    return result


@app.get("/summary")
async def view_summary():
    """View summary of all joints with data counts."""
    trainer_summary = {joint: store.get_data_count() for joint, store in trainer_joint_data_stores.items()}
    user_summary = {joint: store.get_data_count() for joint, store in user_joint_data_stores.items()}
    
    print("\n" + "="*60)
    print("SUMMARY - All Joints Data Count")
    print("="*60)
    print("\nTRAINER:")
    for joint, count in trainer_summary.items():
        print(f"{joint:<25}: {count:>6} points")
    print("\nUSER:")
    for joint, count in user_summary.items():
        print(f"{joint:<25}: {count:>6} points")
    print("="*60 + "\n")
    
    return {
        "trainer": trainer_summary,
        "user": user_summary
    }


@app.get("/api/primary-angles-data")
async def get_primary_angles_data(role: str = "user", seconds: int = None):
    """
    API endpoint to get primary angles JSON data.
    
    Args:
        role: 'trainer' or 'user' (default: 'user')
        seconds: Optional. Analyze only the last N seconds of data.
    
    Returns:
        JSON with primary angles data including std deviations.
    """
    use_trainer = (role == "trainer")
    primary_angles = workout_analyzer.find_primary_angles(use_trainer=use_trainer, seconds=seconds)
    all_joint_std = workout_analyzer.get_all_joint_std_deviations(use_trainer=use_trainer, seconds=seconds)
    
    # Format all joints with std dev
    all_joints_detailed = [
        {
            "joint": joint_name, 
            "std_dev": round(std_dev, 0),
            "is_primary": joint_name in primary_angles
        } 
        for joint_name, std_dev in all_joint_std
    ]
    
    return {
        "role": role,
        "primary_angles": primary_angles,
        "count": len(primary_angles),
        "analyzed_seconds": seconds if seconds else "all",
        "all_joints": all_joints_detailed
    }


@app.get("/api/primary-angles", response_class=HTMLResponse)
async def get_primary_angles(role: str = "user", seconds: int = None):
    """
    Display primary angles in a simple HTML table.
    
    Args:
        role: 'trainer' or 'user' (default: 'user')
        seconds: Optional. Analyze only the last N seconds of data.
    
    Example:
    - http://localhost:8000/api/primary-angles
    - http://localhost:8000/api/primary-angles?role=trainer
    - http://localhost:8000/api/primary-angles?role=user&seconds=5
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Primary Angles</title>
    </head>
    <body>
        <h1>Primary Angles Analysis</h1>
        <p>Role: <strong id="role"></strong> | Primary Angles: <strong id="count"></strong> | Time Window: <strong id="seconds"></strong></p>
        <p>Last Updated: <span id="last-updated"></span></p>
        
        <h2>All Joints (Sorted by Standard Deviation)</h2>
        <table border="1" id="std-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Joint Name</th>
                    <th>Std Deviation</th>
                    <th>Primary Angle</th>
                </tr>
            </thead>
            <tbody>
                <tr><td colspan="4">Loading...</td></tr>
            </tbody>
        </table>
        
        <h2>JSON Response</h2>
        <pre id="json-display">Loading...</pre>

        <script>
            const role = "{role}";
            const seconds = {seconds if seconds else 'null'};
            
            // WebSocket connection for real-time updates
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {{
                console.log('✅ Connected to backend via WebSocket');
                fetchPrimaryAngles(); // Initial fetch
            }};
            
            ws.onmessage = (event) => {{
                try {{
                    const data = JSON.parse(event.data);
                    
                    // Only update when interpolation completes
                    if (data.type === 'primary_angles_updated') {{
                        console.log('🔔 Received update notification at:', data.timestamp);
                        fetchPrimaryAngles(); // Update immediately
                    }}
                }} catch (e) {{
                    console.error('Parse error:', e);
                }}
            }};
            
            ws.onerror = (error) => {{
                console.error('WebSocket error:', error);
            }};
            
            ws.onclose = () => {{
                console.log('❌ Disconnected from backend');
                document.getElementById('last-updated').textContent = 'Disconnected - Retrying...';
                // Retry connection after 2 seconds
                setTimeout(() => location.reload(), 2000);
            }};
            
            async function fetchPrimaryAngles() {{
                try {{
                    const url = `/api/primary-angles-data?role=${{role}}${{seconds !== null ? '&seconds=' + seconds : ''}}`;
                    const response = await fetch(url);
                    const data = await response.json();
                    
                    // Update info
                    document.getElementById('role').textContent = data.role.toUpperCase();
                    document.getElementById('count').textContent = data.count;
                    document.getElementById('seconds').textContent = data.analyzed_seconds === 'all' ? 'All Data' : data.analyzed_seconds + ' seconds';
                    
                    // Update table
                    const tbody = document.querySelector('#std-table tbody');
                    if (data.all_joints && data.all_joints.length > 0) {{
                        let rows = '';
                        data.all_joints.forEach((joint, index) => {{
                            rows += `<tr>
                                <td>${{index + 1}}</td>
                                <td>${{joint.joint}}</td>
                                <td>${{joint.std_dev.toFixed(2)}}</td>
                                <td>${{joint.is_primary ? 'YES' : 'NO'}}</td>
                            </tr>`;
                        }});
                        tbody.innerHTML = rows;
                    }} else {{
                        tbody.innerHTML = '<tr><td colspan="4">No data available</td></tr>';
                    }}
                    
                    // Update JSON
                    document.getElementById('json-display').textContent = JSON.stringify(data, null, 2);
                    
                    // Update timestamp
                    document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
                    
                    console.log('📊 Updated primary angles:', data.primary_angles);
                }} catch (error) {{
                    document.getElementById('json-display').textContent = 'Error: ' + error.message;
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/verify-reps")
async def verify_reps(role: str = "user", seconds: int = None):
    """
    Verify repetition counts across primary angles.
    
    Args:
        role: 'trainer' or 'user' (default: 'user')
        seconds: Optional. Analyze only the last N seconds of data.
    
    Returns:
        Dictionary with verified rep count and per-joint breakdown.
    
    Example:
    - http://localhost:8000/api/verify-reps
    - http://localhost:8000/api/verify-reps?role=trainer
    - http://localhost:8000/api/verify-reps?role=user&seconds=10
    """
    use_trainer = (role == "trainer")
    result = workout_analyzer.verify_reps_across_primary_angles(use_trainer=use_trainer, seconds=seconds)
    result["role"] = role
    
    return result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Accept connection with custom max_size for large messages (500 MB)
    await websocket.accept()
    
    active_connections.append(websocket)
    print(f"WebSocket client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Receive message from frontend (no timeout - indefinite connection)
            data = await websocket.receive_text()
            
            try:
                # Parse JSON to check if it's batched data
                parsed_data = json.loads(data)
                
                # Check if this is interpolated data for in-memory storage
                if parsed_data.get('type') == 'interpolated_data':
                    # Store in memory - separate trainer and user data
                    joint_name = parsed_data['joint']
                    trainer_angle = parsed_data.get('trainerAngle')
                    user_angle = parsed_data.get('userAngle')
                    
                    # Store trainer data
                    trainer_key = f"t_{joint_name}"
                    if trainer_key in trainer_joint_data_stores and trainer_angle is not None:
                        trainer_joint_data_stores[trainer_key].add_data_point(trainer_angle)
                    
                    # Store user data
                    user_key = f"u_{joint_name}"
                    if user_key in user_joint_data_stores and user_angle is not None:
                        user_joint_data_stores[user_key].add_data_point(user_angle)
                    
                    # Skip broadcasting this to other clients
                    continue
                
                # Handle interpolation completion notification
                if parsed_data.get('type') == 'interpolation_complete':
                    print(f"✅ Interpolation complete at {parsed_data.get('timestamp')}")
                    
                    # Broadcast to all connected clients (Primary Angles page)
                    notification = json.dumps({
                        "type": "primary_angles_updated",
                        "timestamp": parsed_data.get('timestamp')
                    })
                    
                    for connection in active_connections:
                        try:
                            await connection.send_text(notification)
                        except:
                            pass
                    
                    continue
                
                if 'frames' in parsed_data and 'frameCount' in parsed_data:
                    # Batched data format
                    frames = parsed_data['frames']
                    frame_count = parsed_data['frameCount']
                          
                    # Process and broadcast each frame individually for plotting
                    for frame in frames:
                        frame_json = json.dumps(frame)
                        
                        # Broadcast to all connected clients (visualization pages)
                        for connection in active_connections:
                            try:
                                await connection.send_text(frame_json)
                            except:
                                pass
                    
                else:
                    # Single frame format (backward compatible)
                    # print(f"📊 Single frame received: {data[:100]}...")
                    
                    # Broadcast to all connected clients
                    for connection in active_connections:
                        try:
                            await connection.send_text(data)
                        except:
                            pass
            
            except json.JSONDecodeError:
                # Not JSON, just broadcast as-is (backward compatible)
                # print(f"Data received (non-JSON): {data}")
                for connection in active_connections:
                    try:
                        await connection.send_text(data)
                    except:
                        pass
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"WebSocket client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

