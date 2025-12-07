// UserDashboard.jsx - Main dashboard page

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import DashboardHeader from '../components/DashboardHeader';
import WorkoutCard from '../components/WorkoutCard';
import WeeklyGoalCard from '../components/WeeklyGoalCard';
import QuickStatsCard from '../components/QuickStatsCard';
import { useFirebase } from '../context/firebase';
import RecentSessionsCard from '../components/RecentSessionsCard';
import '../css/DashboardContainer.css';
import '../css/StatsGrid.css';
import '../css/ErrorStates.css';

function UserDashboard() {
  const navigate = useNavigate();
  const firebase = useFirebase();
  
  const [userData, setUserData] = useState(null);
  const [dataLoading, setDataLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch user data from Firestore
  useEffect(() => {
    const fetchUserData = async () => {
      if (!firebase.currentUser) {
        setDataLoading(false);
        return;
      }

      try {
        setDataLoading(true);
        const data = await firebase.getUserData(firebase.currentUser.uid);
        setUserData(data);
        setError(null);
      } catch (err) {
        setError(err.message);
        console.error("Error fetching user data:", err);
      } finally {
        setDataLoading(false);
      }
    };

    fetchUserData();
  }, [firebase.currentUser]);

  // Handle logout when error occurs
  const handleErrorLogout = async () => {
    try {
      await firebase.signout();
      navigate('/login');
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

  // Sample data for recent sessions (keeping for now)
  const recentSessions = [
    {
      type: 'Cardio',
      date: 'Tue, Jun 28',
      duration: '35m'
    },
    {
      type: 'Cardio',
      date: 'Sun, Jun 26',
      duration: '35m'
    },
    {
      type: 'Lower Body',
      date: 'Fri, Jun 24',
      duration: '50m'
    },
    {
      type: 'Full Body',
      date: 'Wed, Jun 22',
      duration: '60m'
    }
  ];

  const handleStartWorkout = () => {
    alert('Starting workout session!');
    // Add your start workout logic here
  };

  const handleViewDetails = (session) => {
    alert(`Viewing details for ${session.type} session`);
    // Add your view details logic here
  };

  // Show loading state
  if (dataLoading) {
    return (
      <div className="loading-state-container">
        Loading dashboard...
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="error-state-container">
        <div className="error-message">
          Error: {error}
        </div>
        <button onClick={handleErrorLogout} className="error-logout-button">
          Logout
        </button>
      </div>
    );
  }

  // Show error if no user data
  if (!userData) {
    return (
      <div className="error-state-container">
        <div className="no-data-message">
          No user data found
        </div>
        <button onClick={handleErrorLogout} className="error-logout-button">
          Logout
        </button>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-content">
        {/* Header */}
        <DashboardHeader 
          username={userData.name}
        />

        {/* Workout Card */}
        <WorkoutCard onStartWorkout={handleStartWorkout} />

        {/* Stats Grid */}
        <div className="stats-grid">
          <WeeklyGoalCard 
            current={0}
            target={userData.weeklyWorkoutsTarget} 
          />
          <QuickStatsCard 
            dayStreak={userData.dayStreak}
            totalWorkouts={userData.totalWorkouts}
            avgDuration={userData.avgDurationMinutes}
          />
        </div>

        {/* Recent Sessions */}
        <RecentSessionsCard 
          sessions={recentSessions}
          onViewDetails={handleViewDetails}
        />
      </div>
    </div>
  );
}

export default UserDashboard;
