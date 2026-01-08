from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import random
import json
import os

app = FastAPI()

# Mount static files directory (create a 'static' folder in your project)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def plot_fps_data():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time FPS Interpolation</title>
        <link rel="stylesheet" href="/static/uPlot.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .chart-container {
                background: white;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            h2 {
                color: #666;
                margin-top: 0;
            }
            .info {
                background: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .stats {
                background: #fff3e0;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Real-time FPS Data Interpolation</h1>
            
            <div class="info">
                <strong>Raw Data:</strong> Random FPS (5-60) generated every second<br>
                <strong>Interpolated Data:</strong> Smoothed to 60 FPS using linear interpolation<br>
                <strong>Display Window:</strong> Last 3 seconds of data<br>
                <strong>Max Data Points:</strong> 300 points maintained
            </div>
            
            <div class="stats" id="stats">
                Time: 0.0s | Raw Points: 0 | Interpolated Points: 0 | Current FPS: 0
            </div>
            
            <div class="chart-container">
                <h2>Raw Data (Random FPS per Second)</h2>
                <div id="chart1"></div>
            </div>
            
            <div class="chart-container">
                <h2>Interpolated Data (60 FPS)</h2>
                <div id="chart2"></div>
            </div>
        </div>
        
        <script src="/static/uPlot.iife.min.js"></script>
        <script>
            let rawTimes = [];
            let rawValues = [];
            let interpolatedTimes = [];
            let interpolatedValues = [];
            let currentTime = 0;
            let chart1, chart2;
            const MAX_POINTS = 300;
            const DISPLAY_WINDOW = 3; // seconds
            const TARGET_FPS = 60;
            
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
            
            function generateDataForSecond() {
                const fps = Math.floor(Math.random() * 56) + 5; // 5-60 FPS
                const pointsThisSecond = fps;
                
                // Generate raw data points for this second
                for (let i = 0; i < pointsThisSecond; i++) {
                    rawTimes.push(currentTime + (i / fps));
                    rawValues.push(Math.random() * 100);
                }
                
                currentTime += 1;
                
                // Keep only last 300 points
                if (rawTimes.length > MAX_POINTS) {
                    const excess = rawTimes.length - MAX_POINTS;
                    rawTimes = rawTimes.slice(excess);
                    rawValues = rawValues.slice(excess);
                }
                
                // Generate interpolated data
                if (rawTimes.length >= 2) {
                    const startTime = rawTimes[0];
                    const endTime = rawTimes[rawTimes.length - 1];
                    const timeSpan = endTime - startTime;
                    const numInterpolated = Math.min(MAX_POINTS, Math.floor(timeSpan * TARGET_FPS) + 1);
                    
                    interpolatedTimes = [];
                    for (let i = 0; i < numInterpolated; i++) {
                        interpolatedTimes.push(startTime + (i * timeSpan / (numInterpolated - 1)));
                    }
                    
                    interpolatedValues = linearInterpolate(rawTimes, rawValues, numInterpolated);
                }
                
                // Filter to show only last 3 seconds
                const cutoffTime = currentTime - DISPLAY_WINDOW;
                
                const rawStartIdx = rawTimes.findIndex(t => t >= cutoffTime);
                const displayRawTimes = rawStartIdx >= 0 ? rawTimes.slice(rawStartIdx) : rawTimes;
                const displayRawValues = rawStartIdx >= 0 ? rawValues.slice(rawStartIdx) : rawValues;
                
                const interpStartIdx = interpolatedTimes.findIndex(t => t >= cutoffTime);
                const displayInterpTimes = interpStartIdx >= 0 ? interpolatedTimes.slice(interpStartIdx) : interpolatedTimes;
                const displayInterpValues = interpStartIdx >= 0 ? interpolatedValues.slice(interpStartIdx) : interpolatedValues;
                
                // Update charts
                if (chart1 && chart2) {
                    chart1.setData([displayRawTimes, displayRawValues]);
                    
                    // Set x-axis range for raw data chart
                    chart1.setScale('x', {
                        min: cutoffTime,
                        max: currentTime
                    });
                    
                    chart2.setData([displayInterpTimes, displayInterpValues]);
                    
                    // Set x-axis range for interpolated chart
                    chart2.setScale('x', {
                        min: cutoffTime,
                        max: currentTime
                    });
                }
                
                // Update stats
                document.getElementById('stats').textContent = 
                    `Time: ${currentTime.toFixed(1)}s | Raw Points: ${rawTimes.length} | Interpolated Points: ${interpolatedTimes.length} | Current FPS: ${fps}`;
            }
            
            function initCharts() {
                if (typeof uPlot === 'undefined') {
                    console.error('uPlot failed to load');
                    return;
                }
                
                const commonOpts = {
                    width: 1000,
                    height: 300,
                    scales: {
                        x: {
                            time: false
                        },
                        y: {
                            auto: true,
                            range: [0, 100]
                        }
                    },
                    axes: [
                        {
                            label: "Time (seconds)",
                            labelSize: 30,
                            labelFont: "bold 14px Arial",
                        },
                        {
                            label: "Value",
                            labelSize: 30,
                            labelFont: "bold 14px Arial",
                            scale: "y",
                        }
                    ]
                };
                
                const opts1 = {
                    ...commonOpts,
                    title: "Raw Data with Random FPS",
                    series: [
                        {},
                        {
                            label: "Value",
                            stroke: "red",
                            width: 2,
                            points: {
                                show: true,
                                size: 6,
                                fill: "red"
                            }
                        }
                    ]
                };
                
                const opts2 = {
                    ...commonOpts,
                    title: "Interpolated to 60 FPS",
                    series: [
                        {},
                        {
                            label: "Value",
                            stroke: "blue",
                            width: 2,
                            points: {
                                show: true,
                                size: 3,
                                fill: "blue"
                            }
                        }
                    ]
                };
                
                chart1 = new uPlot(opts1, [[], []], document.getElementById("chart1"));
                chart2 = new uPlot(opts2, [[], []], document.getElementById("chart2"));
                
                // Start generating data every second
                setInterval(generateDataForSecond, 1000);
                
                // Generate initial data
                generateDataForSecond();
            }
            
            setTimeout(initCharts, 100);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)