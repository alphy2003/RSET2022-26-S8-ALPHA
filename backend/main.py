from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import json

app = FastAPI()

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

@app.get("/messages", response_class=HTMLResponse)
async def get_messages_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Messages Display</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
            }
            #messages {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                min-height: 400px;
            }
            .message {
                padding: 10px;
                margin: 10px 0;
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                border-radius: 4px;
            }
            .timestamp {
                color: #666;
                font-size: 12px;
                margin-top: 5px;
            }
            .status {
                padding: 10px;
                margin-bottom: 20px;
                border-radius: 4px;
            }
            .connected {
                background: #d4edda;
                color: #155724;
            }
            .disconnected {
                background: #f8d7da;
                color: #721c24;
            }
        </style>
    </head>
    <body>
        <h1>Real-time Messages Display</h1>
        <div id="status" class="status disconnected">Connecting...</div>
        <div id="messages"></div>

        <script>
            const messagesDiv = document.getElementById('messages');
            const statusDiv = document.getElementById('status');
            
            // Connect to WebSocket
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                statusDiv.textContent = 'Connected - Waiting for messages...';
                statusDiv.className = 'status connected';
            };
            
            ws.onmessage = (event) => {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';
                
                const now = new Date().toLocaleString();
                messageDiv.innerHTML = `
                    <div>${event.data}</div>
                    <div class="timestamp">${now}</div>
                `;
                
                messagesDiv.insertBefore(messageDiv, messagesDiv.firstChild);
            };
            
            ws.onerror = (error) => {
                statusDiv.textContent = 'Connection error';
                statusDiv.className = 'status disconnected';
            };
            
            ws.onclose = () => {
                statusDiv.textContent = 'Disconnected';
                statusDiv.className = 'status disconnected';
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/posture", response_class=HTMLResponse)
async def get_posture_page():
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
                filterType: 'kalman',
                
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
            // INTERPOLATION FOR 30 FPS
            // ============================================
            
            // Linear interpolation function
            function linearInterpolate(timestamps, values, targetTimestamps) {
                const interpolated = [];
                
                for (const targetTime of targetTimestamps) {
                    // Find surrounding points
                    let i = 0;
                    while (i < timestamps.length - 1 && timestamps[i + 1] < targetTime) {
                        i++;
                    }
                    
                    if (i >= timestamps.length - 1) {
                        // Beyond data range, use last value
                        interpolated.push(values[values.length - 1]);
                    } else if (targetTime <= timestamps[0]) {
                        // Before data range, use first value
                        interpolated.push(values[0]);
                    } else {
                        // Interpolate between points
                        const t0 = timestamps[i];
                        const t1 = timestamps[i + 1];
                        const v0 = values[i];
                        const v1 = values[i + 1];
                        
                        if (v0 === null || v1 === null) {
                            interpolated.push(null);
                        } else {
                            const ratio = (targetTime - t0) / (t1 - t0);
                            const interpolatedValue = v0 + ratio * (v1 - v0);
                            interpolated.push(interpolatedValue);
                        }
                    }
                }
                
                return interpolated;
            }
            
            // Resample to 30 FPS for a given time range
            function resampleTo30FPS(timestamps, smoothedData, startTime, endTime) {
                const fps = 30;
                const frameInterval = 1.0 / fps;
                const targetTimestamps = [];
                
                for (let t = startTime; t < endTime; t += frameInterval) {
                    targetTimestamps.push(t);
                }
                
                return {
                    timestamps: targetTimestamps,
                    values: linearInterpolate(timestamps, smoothedData, targetTimestamps)
                };
            }
            
            // Display interpolated 30 FPS data in terminal
            function displayInterpolated30FPS() {
                // Get the time range of current data
                const allTimestamps = chartData['ls'].timestamps; // Use any joint
                if (allTimestamps.length < 2) return;
                
                const startTime = Math.floor(allTimestamps[0]);
                const endTime = Math.floor(allTimestamps[allTimestamps.length - 1]);
                
                // Process each complete second
                for (let second = startTime; second < endTime; second++) {
                    console.log(`\n${'='.repeat(80)}`);
                    console.log(`SECOND ${second} - 30 FPS INTERPOLATED FRAMES`);
                    console.log('='.repeat(80));
                    
                    // Resample each joint for this second
                    const secondData = {};
                    Object.keys(joints).forEach(jointKey => {
                        const timestamps = chartData[jointKey].timestamps;
                        const smoothed = chartData[jointKey].userAngleSmoothed;
                        const trainerSmoothed = chartData[jointKey].trainerAngleSmoothed;
                        
                        if (smoothed.length > 0) {
                            const userResampled = resampleTo30FPS(timestamps, smoothed, second, second + 1);
                            const trainerResampled = resampleTo30FPS(timestamps, trainerSmoothed, second, second + 1);
                            
                            secondData[jointKey] = {
                                user: userResampled.values,
                                trainer: trainerResampled.values
                            };
                        }
                    });
                    
                    // Display all 30 frames for this second
                    for (let frame = 0; frame < 30; frame++) {
                        const frameTime = second + (frame / 30);
                        console.log(`\nFrame ${frame + 1}/30 (t=${frameTime.toFixed(3)}s):`);
                        
                        Object.keys(joints).forEach(jointKey => {
                            if (secondData[jointKey]) {
                                const userAngle = secondData[jointKey].user[frame];
                                const trainerAngle = secondData[jointKey].trainer[frame];
                                if (userAngle !== null && trainerAngle !== null) {
                                    console.log(`  ${joints[jointKey]}: User=${userAngle.toFixed(1)}° Trainer=${trainerAngle.toFixed(1)}°`);
                                }
                            }
                        });
                    }
                }
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
                    lastValidTrainerAngle: null
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
                            label: "Time"
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
                    
                    // Apply the selected filter to all joints
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
                    });
                    
                    // Display interpolated 30 FPS data in terminal
                    displayInterpolated30FPS();
                    
                    lastSmoothingTime = now;
                    console.log(`✨ ${filterName} filter applied`);
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"WebSocket client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            
            try:
                # Parse JSON to check if it's batched data
                parsed_data = json.loads(data)
                
                if 'frames' in parsed_data and 'frameCount' in parsed_data:
                    # Batched data format
                    frames = parsed_data['frames']
                    frame_count = parsed_data['frameCount']
                    
                    # Minimal logging - just batch info
                    print(f"Received batch: {frame_count} frames")
                    
                    # Process and broadcast each frame individually for plotting
                    for frame in frames:
                        frame_json = json.dumps(frame)
                        
                        # Broadcast to all connected clients (visualization pages)
                        for connection in active_connections:
                            try:
                                await connection.send_text(frame_json)
                            except:
                                pass
                    
                    print(f"Broadcasted {frame_count} frames to {len(active_connections)} clients")
                
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

