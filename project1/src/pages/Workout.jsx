import { useState, useRef, useCallback, useEffect } from 'react';
import VideoUploadPose from '../components/VideoUploadPose';
import WebcamLiveFeed from '../components/WebcamLiveFeed';
import { createWorkoutDataSnapshot } from '../utils/poseCalculations';
import '../css/Workout.css';

function Workout() {
  const [videoFile, setVideoFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoUploadRef = useRef(null);
  const videoFileRef = useRef(null);
  
  // Store latest angles and gesture data
  const [trainerAngles, setTrainerAngles] = useState({});
  const [userAngles, setUserAngles] = useState({});
  const [currentGesture, setCurrentGesture] = useState(null);
  const [gestureConfidence, setGestureConfidence] = useState(0);
  const [combinedData, setCombinedData] = useState(null);

  // Update video file and ref together
  const handleVideoFileChange = (url) => {
    setVideoFile(url);
    videoFileRef.current = url;
  };

  // Handle trainer video angles update
  const handleTrainerAnglesUpdate = useCallback((angles) => {
    setTrainerAngles(angles);
  }, []);

  // Handle user webcam angles update
  const handleUserAnglesUpdate = useCallback((angles) => {
    setUserAngles(angles);
  }, []);

  // Handle gesture update from webcam
  const handleGestureUpdate = useCallback((gesture, confidence) => {
    setCurrentGesture(gesture);
    setGestureConfidence(confidence);
  }, []);

  // Combine all data whenever any part updates
  useEffect(() => {
    const snapshot = createWorkoutDataSnapshot(
      trainerAngles,
      userAngles,
      currentGesture,
      gestureConfidence
    );
    setCombinedData(snapshot);
    
    // Log combined data (you can also save to database, export, etc.)
    console.log('Combined Workout Data:', JSON.stringify(snapshot, null, 2));
  }, [trainerAngles, userAngles, currentGesture, gestureConfidence]);

  // Handle gesture control for video playback
  const handleGestureControl = useCallback((gesture) => {
    // Check if video file exists using ref (to avoid stale closure)
    if (!videoFileRef.current) {
      console.log('No video file uploaded yet');
      return;
    }
    
    // Get video element from ref
    const videoElement = videoUploadRef.current?.videoElement;
    if (!videoElement) {
      console.log('Video element not ready');
      return;
    }
    
    if (gesture === 'Pointing_Up' && videoElement.paused) {
      videoElement.play().then(() => {
        console.log('▶️ Playing video via gesture');
      }).catch(err => console.error('Play error:', err));
    } else if (gesture === 'Victory' && !videoElement.paused) {
      videoElement.pause();
      console.log('⏸️ Pausing video via gesture');
    }
  }, []);

  return (
    <div className="workout-container">
      <VideoUploadPose 
        ref={videoUploadRef}
        videoFile={videoFile}
        onVideoFileChange={handleVideoFileChange}
        onPlayStateChange={setIsPlaying}
        onAnglesUpdate={handleTrainerAnglesUpdate}
      />
      
      <WebcamLiveFeed 
        onGestureDetected={handleGestureControl}
        onAnglesUpdate={handleUserAnglesUpdate}
        onGestureUpdate={handleGestureUpdate}
      />

      {/* Display combined JSON data */}
      {combinedData && (
        <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          background: 'rgba(0,0,0,0.9)',
          color: '#10b981',
          padding: '15px',
          borderRadius: '8px',
          maxWidth: '400px',
          maxHeight: '300px',
          overflow: 'auto',
          fontSize: '10px',
          fontFamily: 'monospace',
          zIndex: 1000
        }}>
          <div style={{ marginBottom: '10px', fontWeight: 'bold', fontSize: '12px' }}>Combined Data:</div>
          <pre style={{ margin: 0 }}>{JSON.stringify(combinedData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default Workout;
