from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
    return {"Hello": "World"}

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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Posture Correction - Live Joint Angles (uPlot - Fast)</h1>
            <div id="status" class="status disconnected">Connecting...</div>
            
            <div class="charts-grid" id="charts-container"></div>
        </div>

        <script>
            const statusDiv = document.getElementById('status');
            const chartsContainer = document.getElementById('charts-container');
            const maxDataPoints = 300; // Keep last 300 data points (~10 seconds at 30fps)
            
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
                    userConf: [],
                    trainerConf: []
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
                
                // Create chart with initial empty data
                const data = [
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
                    
                    // Add user angle
                    if (data[userKey]) {
                        chartData[jointKey].userAngles.push(data[userKey][0]);
                        chartData[jointKey].userConf.push(data[userKey][1]);
                    } else {
                        chartData[jointKey].userAngles.push(null);
                        chartData[jointKey].userConf.push(0);
                    }
                    
                    // Add trainer angle
                    if (data[trainerKey]) {
                        chartData[jointKey].trainerAngles.push(data[trainerKey][0]);
                        chartData[jointKey].trainerConf.push(data[trainerKey][1]);
                    } else {
                        chartData[jointKey].trainerAngles.push(null);
                        chartData[jointKey].trainerConf.push(0);
                    }
                    
                    // Keep only last maxDataPoints
                    if (chartData[jointKey].timestamps.length > maxDataPoints) {
                        chartData[jointKey].timestamps.shift();
                        chartData[jointKey].userAngles.shift();
                        chartData[jointKey].trainerAngles.shift();
                        chartData[jointKey].userConf.shift();
                        chartData[jointKey].trainerConf.shift();
                    }
                    
                    // Update uPlot with new data
                    charts[jointKey].setData([
                        chartData[jointKey].timestamps,
                        chartData[jointKey].userAngles,
                        chartData[jointKey].trainerAngles
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
                
                // Continue processing if more frames arrived
                if (frameQueue.length > 0) {
                    requestAnimationFrame(processFrameQueue);
                } else {
                    isProcessing = false;
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
                    batch_timestamp = parsed_data.get('batchTimestamp', 'N/A')
                    
                    print(f"\n{'='*80}")
                    print(f"📦 BATCH RECEIVED")
                    print(f"{'='*80}")
                    print(f"Frame Count: {frame_count}")
                    print(f"Batch Timestamp: {batch_timestamp}")
                    print(f"Connected Clients: {len(active_connections)}")
                    # print(f"\n🔍 All {frame_count} frames in batch:")
                    # for idx, frame in enumerate(frames):
                    #     print(f"  Frame {idx}: t={frame.get('t', 'N/A')}, keys={list(frame.keys())}")
                    print(f"{'='*80}\n")
                    
                    # Process and broadcast each frame individually for plotting
                    for idx, frame in enumerate(frames):
                        frame_json = json.dumps(frame)
                        
                        # Broadcast to all connected clients (visualization pages)
                        for connection in active_connections:
                            try:
                                await connection.send_text(frame_json)
                            except:
                                pass
                    
                    print(f"✅ Broadcasted {frame_count} frames to {len(active_connections)} clients")
                
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

