import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def generate_random_fps_data(duration_seconds=5, min_fps=30, max_fps=90):
    """
    Generate data with random FPS for each second.
    
    Parameters:
    - duration_seconds: total duration in seconds
    - min_fps: minimum frames per second
    - max_fps: maximum frames per second
    
    Returns:
    - timestamps: array of timestamps
    - values: array of values (0-180)
    """
    timestamps = []
    values = []
    
    current_time = 0.0
    current_value = np.random.uniform(0, 180)
    
    for second in range(duration_seconds):
        # Random FPS for this second
        fps_this_second = np.random.randint(min_fps, max_fps + 1)
        frame_interval = 1.0 / fps_this_second
        
        print(f"Second {second}: {fps_this_second} FPS (interval: {frame_interval*1000:.2f}ms)")
        
        # Generate frames for this second
        for frame in range(fps_this_second):
            timestamps.append(current_time)
            values.append(current_value)
            
            current_time += frame_interval
            # Smooth transition to next value
            current_value += np.random.uniform(-15, 15)
            current_value = np.clip(current_value, 0, 180)
    
    # Add final point
    timestamps.append(duration_seconds)
    values.append(current_value)
    
    return np.array(timestamps), np.array(values)

def interpolate_to_60fps(timestamps, values):
    """
    Perform linear interpolation to achieve exactly 60 FPS.
    
    Parameters:
    - timestamps: array of timestamps in seconds
    - values: array of values (0-180)
    
    Returns:
    - interpolated_timestamps: timestamps at 60 FPS intervals
    - interpolated_values: interpolated values
    """
    # Get time range
    min_time = timestamps[0]
    max_time = timestamps[-1]
    
    # 60 FPS means 1/60 second between frames
    frame_interval = 1 / 60
    
    # Generate timestamps at exactly 60 FPS intervals
    num_frames = int((max_time - min_time) / frame_interval) + 1
    interpolated_timestamps = np.linspace(min_time, max_time, num_frames)
    
    # Perform linear interpolation
    interpolator = interpolate.interp1d(timestamps, values, kind='linear')
    interpolated_values = interpolator(interpolated_timestamps)
    
    return interpolated_timestamps, interpolated_values

def plot_data(timestamps, values, interpolated_timestamps, interpolated_values):
    """
    Plot both original and interpolated data.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Original data with varying FPS
    ax1.plot(timestamps, values, '-', linewidth=1.5, color='#667eea', alpha=0.6)
    ax1.plot(timestamps, values, 'o', markersize=5, color='#764ba2', 
             label='Original Points (Variable FPS)', alpha=0.8)
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Original Data (Random 30-90 FPS per Second)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-10, 190)
    ax1.legend(loc='upper right')
    
    # Add statistics for original data
    duration = timestamps[-1] - timestamps[0]
    num_points = len(timestamps)
    avg_interval = duration / (num_points - 1)
    fps_original = 1 / avg_interval
    
    stats_text = (f'Total Points: {num_points} | Duration: {duration:.3f}s | '
                 f'Avg Interval: {avg_interval*1000:.2f}ms | Avg FPS: {fps_original:.2f}')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add vertical lines for each second
    for sec in range(int(duration) + 1):
        ax1.axvline(x=sec, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Plot 2: Interpolated data at 60 FPS with lines connecting dots
    ax2.plot(interpolated_timestamps, interpolated_values, '-', 
             linewidth=1.5, color='#00CC66', alpha=0.6, label='60 FPS Interpolated')
    ax2.plot(interpolated_timestamps, interpolated_values, 'o', 
             markersize=4, color='#00AA55', alpha=0.8)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('60 FPS Interpolated Data (Equal Time Spacing)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(-10, 190)
    ax2.legend(loc='upper right')
    
    # Add statistics for interpolated data
    duration_interp = interpolated_timestamps[-1] - interpolated_timestamps[0]
    num_points_interp = len(interpolated_timestamps)
    avg_interval_interp = duration_interp / (num_points_interp - 1)
    fps_interp = 1 / avg_interval_interp
    
    stats_text_interp = (f'Total Points: {num_points_interp} | Duration: {duration_interp:.3f}s | '
                        f'Interval: {avg_interval_interp*1000:.3f}ms | FPS: {fps_interp:.2f}')
    ax2.text(0.02, 0.98, stats_text_interp, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add vertical lines for each second
    for sec in range(int(duration_interp) + 1):
        ax2.axvline(x=sec, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.show()

def analyze_fps_per_second(timestamps, duration_seconds):
    """
    Analyze FPS for each second of the data.
    """
    print("\nFPS Analysis per Second:")
    print("-" * 60)
    
    for second in range(duration_seconds):
        # Count frames in this second
        frames_in_second = np.sum((timestamps >= second) & (timestamps < second + 1))
        print(f"Second {second}: {frames_in_second} frames")
    
    print("-" * 60)

def main():
    # Configuration
    duration_seconds = 5
    min_fps = 30
    max_fps = 90
    
    print("=" * 60)
    print("Generating Random FPS Data")
    print("=" * 60)
    
    # Generate data with random FPS per second
    timestamps, values = generate_random_fps_data(duration_seconds, min_fps, max_fps)
    
    print(f"\nOriginal Data Summary:")
    print(f"Total number of points: {len(timestamps)}")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.3f} seconds")
    print(f"Value range: [{values.min():.2f}, {values.max():.2f}]")
    
    # Analyze FPS per second
    analyze_fps_per_second(timestamps, duration_seconds)
    
    # Perform 60 FPS interpolation
    print("\n" + "=" * 60)
    print("Performing 60 FPS Interpolation")
    print("=" * 60)
    
    interpolated_timestamps, interpolated_values = interpolate_to_60fps(timestamps, values)
    
    print(f"\nInterpolated Data Summary:")
    print(f"Total number of points: {len(interpolated_timestamps)}")
    print(f"Duration: {interpolated_timestamps[-1] - interpolated_timestamps[0]:.3f} seconds")
    print(f"Frame interval: {1/60:.6f} seconds ({1000/60:.3f} ms)")
    
    # Verify equal spacing
    intervals = np.diff(interpolated_timestamps)
    print(f"\nTime Interval Verification:")
    print(f"Min interval: {intervals.min():.6f}s")
    print(f"Max interval: {intervals.max():.6f}s")
    print(f"Mean interval: {intervals.mean():.6f}s")
    print(f"All intervals equal: {np.allclose(intervals, intervals[0])}")
    
    actual_fps = 1 / intervals.mean()
    print(f"Actual FPS: {actual_fps:.2f}")
    
    # Show sample of interpolated data
    print(f"\nFirst 10 interpolated points:")
    print(f"{'Index':<8} {'Time (s)':<12} {'Value':<10}")
    print("-" * 32)
    for i in range(min(10, len(interpolated_timestamps))):
        print(f"{i:<8} {interpolated_timestamps[i]:<12.6f} {interpolated_values[i]:<10.2f}")
    
    # Plot both graphs
    print("\nGenerating plots...")
    plot_data(timestamps, values, interpolated_timestamps, interpolated_values)

if __name__ == "__main__":
    main()