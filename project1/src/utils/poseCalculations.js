// Calculate angle between three points with weighted confidence
export const calculateAngle = (a, b, c) => {
  // Check if visibility property exists (some landmarks might not have it)
  const aVis = a.visibility ?? 1.0;
  const bVis = b.visibility ?? 1.0;
  const cVis = c.visibility ?? 1.0;
  
  // Weighted confidence: middle point (joint) is most important
  const angleConfidence = (
    aVis * 0.25 +
    bVis * 0.50 +  // Joint itself gets 50% weight
    cVis * 0.25
  );
  
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) {
    angle = 360 - angle;
  }
  
  return {
    angle: parseFloat(angle.toFixed(2)),
    confidence: parseFloat((angleConfidence * 100).toFixed(1))
  };
};

// Calculate all joint angles
export const calculateJointAngles = (landmarks) => {
  if (!landmarks || landmarks.length < 33) return {};

  return {
    leftShoulder: calculateAngle(landmarks[23], landmarks[11], landmarks[13]),
    leftElbow: calculateAngle(landmarks[11], landmarks[13], landmarks[15]),
    leftWrist: calculateAngle(landmarks[13], landmarks[15], landmarks[19]),
    rightShoulder: calculateAngle(landmarks[24], landmarks[12], landmarks[14]),
    rightElbow: calculateAngle(landmarks[12], landmarks[14], landmarks[16]),
    rightWrist: calculateAngle(landmarks[14], landmarks[16], landmarks[20]),
    leftHip: calculateAngle(landmarks[11], landmarks[23], landmarks[25]),
    leftKnee: calculateAngle(landmarks[23], landmarks[25], landmarks[27]),
    leftAnkle: calculateAngle(landmarks[25], landmarks[27], landmarks[31]),
    rightHip: calculateAngle(landmarks[12], landmarks[24], landmarks[26]),
    rightKnee: calculateAngle(landmarks[24], landmarks[26], landmarks[28]),
    rightAnkle: calculateAngle(landmarks[26], landmarks[28], landmarks[32]),
    leftSpine: calculateAngle(landmarks[11], landmarks[23], landmarks[25]),
    rightSpine: calculateAngle(landmarks[12], landmarks[24], landmarks[26]),
    neck: calculateAngle(landmarks[11], landmarks[0], landmarks[12])
  };
};

// Create combined workout data snapshot
export const createWorkoutDataSnapshot = (trainerAngles, userAngles, gesture, gestureConfidence, timestamp) => {
  return {
    timestamp: timestamp || Date.now(),
    trainer: {
      pose: trainerAngles
    },
    user: {
      pose: userAngles,
      gesture: {
        name: gesture || null,
        confidence: gestureConfidence ? parseFloat((gestureConfidence * 100).toFixed(1)) : 0
      }
    }
  };
};
