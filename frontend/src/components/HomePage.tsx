import React, { useState, useEffect, useRef } from 'react';
import '../styles/HomePage.css';

const BACKEND_URL = 'http://localhost:8000';
// Fixed output video path
const OUTPUT_VIDEO_PATH = '/output_videos/output_video.mp4';

const HomePage = () => {
  const [teamName, setTeamName] = useState('');
  const [opponentTeamName, setOpponentTeamName] = useState('');
  const [parameters, setParameters] = useState({
    speed: false,
    distanceCovered: false,
    passAccuracy: false
  });
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [step, setStep] = useState(1);
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'completed' | 'error'>('idle');
  const [outputVideoUrl, setOutputVideoUrl] = useState<string | null>(null);
  const [recommendation, setRecommendation] = useState<any | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [videoAvailable, setVideoAvailable] = useState(false);
  const [videoFormat, setVideoFormat] = useState<'mp4' | 'avi'>('mp4');
  const [altVideoUrl, setAltVideoUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Function to check video availability
  const checkVideoAvailability = () => {
    if (outputVideoUrl) {
      setVideoAvailable(false); // Reset to loading state
      
      // Try a fetch with range request
      fetch(outputVideoUrl, { 
        method: 'GET',
        headers: {
          'Range': 'bytes=0-1000' // Request just the first 1000 bytes to check
        }
      })
        .then(response => {
          if (response.ok || response.status === 206) {
            console.log('Video is available with status:', response.status);
            setVideoAvailable(true);
            
            // Try to reload the video element
            if (videoRef.current) {
              videoRef.current.load();
            }
          } else {
            console.error('Output video not available', response.status);
            setVideoAvailable(false);
          }
        })
        .catch(error => {
          console.error('Error checking video availability:', error);
          setVideoAvailable(false);
        });
    }
  };

  // Check if video is available after processing
  useEffect(() => {
    if (processingStatus === 'completed' && outputVideoUrl) {
      let retryCount = 0;
      const maxRetries = 10;  // Increase retry count
      const retryInterval = 1500; // 1.5 seconds
      
      const attemptVideoCheck = () => {
        // Try to fetch the video to check if it's available
        fetch(outputVideoUrl, { 
          method: 'GET',
          headers: {
            'Range': 'bytes=0-1000' // Request just first part to confirm availability
          }
        })
          .then(response => {
            if (response.ok || response.status === 206) {
              console.log('Video is available with status:', response.status);
              setVideoAvailable(true);
              
              // Determine format from URL
              if (outputVideoUrl.endsWith('.avi')) {
                setVideoFormat('avi');
              } else {
                setVideoFormat('mp4');
              }
              
              // Try to reload the video element
              if (videoRef.current) {
                videoRef.current.load();
              }
            } else if (retryCount < maxRetries) {
              retryCount++;
              console.warn(`Output video not available yet. Status: ${response.status}. Retry ${retryCount}/${maxRetries} in ${retryInterval/1000}s`);
              setTimeout(attemptVideoCheck, retryInterval);
            } else {
              console.error('Output video not available after maximum retries');
              setVideoAvailable(false);
            }
          })
          .catch(error => {
            if (retryCount < maxRetries) {
              retryCount++;
              console.warn(`Error checking video. Retry ${retryCount}/${maxRetries} in ${retryInterval/1000}s`, error);
              setTimeout(attemptVideoCheck, retryInterval);
            } else {
              console.error('Error checking video availability after maximum retries:', error);
              setVideoAvailable(false);
            }
          });
      };
      
      attemptVideoCheck();
    }
  }, [processingStatus, outputVideoUrl]);

  // Handle video error event
  const handleVideoError = (e: React.SyntheticEvent<HTMLElement, Event>) => {
    console.error('Video error event:', e);
    setVideoAvailable(false);
  };

  const handleParameterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = e.target;
    setParameters({
      ...parameters,
      [name]: checked
    });
  };

  const validateTeamName = (name: string) => {
    // Check if team name contains any numbers
    return !(/\d/.test(name)) && name.trim().length > 0;
  };

  const handleTeamNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setTeamName(value);
  };

  const handleOpponentTeamNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setOpponentTeamName(value);
  };

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setVideoFile(e.target.files[0]);
    }
  };

  // Convert selected parameters to the substitution metric expected by backend
  const getSubstitutionMetric = () => {
    if (parameters.speed && parameters.distanceCovered && parameters.passAccuracy) {
      return "7"; // All metrics combined
    } else if (parameters.speed && parameters.distanceCovered) {
      return "4"; // Speed + Distance
    } else if (parameters.speed && parameters.passAccuracy) {
      return "5"; // Speed + Pass Accuracy
    } else if (parameters.distanceCovered && parameters.passAccuracy) {
      return "6"; // Distance + Pass Accuracy
    } else if (parameters.speed) {
      return "1"; // Speed only
    } else if (parameters.distanceCovered) {
      return "2"; // Distance only
    } else if (parameters.passAccuracy) {
      return "3"; // Pass Accuracy only
    }
    return "1"; // Default to Speed if nothing selected
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setProcessingStatus('processing');
    setErrorMessage(null);
    
    try {
      const formData = new FormData();
      
      if (videoFile) {
        formData.append('file', videoFile);
      }
      
      formData.append('team1', teamName);
      formData.append('team2', opponentTeamName);
      formData.append('substitution_metric', getSubstitutionMetric());
      formData.append('selected_team', teamName); // User's team is selected for substitution
      formData.append('use_stubs', 'true');
      
      // Send request to backend
      const response = await fetch(`${BACKEND_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process video');
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Use the provided video URL from the backend
        const videoUrl = `${BACKEND_URL}${result.output_video_url}?t=${new Date().getTime()}`;
        setOutputVideoUrl(videoUrl);
        
        // Also store alternative video URL if available
        if (result.alt_video_url) {
          const altUrl = `${BACKEND_URL}${result.alt_video_url}?t=${new Date().getTime()}`;
          setAltVideoUrl(altUrl);
        } else {
          setAltVideoUrl(null);
        }
        
        // Set format based on URL
        if (result.output_video_url.endsWith('.avi')) {
          setVideoFormat('avi');
        } else {
          setVideoFormat('mp4');
        }
        
        setRecommendation(result.recommendation);
        setProcessingStatus('completed');
        setStep(3); // Move to success step
        
        // Don't assume video is available immediately
        setVideoAvailable(false);
      } else {
        throw new Error(result.error || 'Failed to process video');
      }
    } catch (error) {
      console.error('Error submitting video analysis:', error);
      setProcessingStatus('error');
      setErrorMessage(error instanceof Error ? error.message : 'An unknown error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderStepOne = () => (
    <div className="setup-container">
      <h2>Welcome, Coach!</h2>
      <p>Let's set up your team's substitution analytics system.</p>
      
      <div className="form-group">
        <label htmlFor="teamName">Your Team Name</label>
        <input 
          type="text" 
          id="teamName" 
          value={teamName} 
          onChange={handleTeamNameChange}
          placeholder="Enter your team name"
          required
        />
        {teamName.trim() && !validateTeamName(teamName) && (
          <p className="error-message">Team name should not contain numbers</p>
        )}
      </div>
      
      <div className="form-group">
        <label htmlFor="opponentTeamName">Opponent Team Name</label>
        <input 
          type="text" 
          id="opponentTeamName" 
          value={opponentTeamName} 
          onChange={handleOpponentTeamNameChange}
          placeholder="Enter opponent team name"
          required
        />
        {opponentTeamName.trim() && !validateTeamName(opponentTeamName) && (
          <p className="error-message">Team name should not contain numbers</p>
        )}
      </div>
      
      <button 
        className="primary-button" 
        onClick={() => setStep(2)} 
        disabled={!validateTeamName(teamName) || !validateTeamName(opponentTeamName)}
      >
        Continue
      </button>
    </div>
  );

  const renderStepTwo = () => (
    <form onSubmit={handleSubmit} className="setup-container">
      <h2>Select Parameters for Analysis</h2>
      <p>Choose which metrics SmartSub should consider when making substitution recommendations</p>
      
      <div className="parameters-grid">
        <div className="parameter-card">
          <input 
            type="checkbox" 
            id="speed" 
            name="speed"
            checked={parameters.speed}
            onChange={handleParameterChange}
          />
          <label htmlFor="speed">
            <div className="parameter-icon">🏃</div>
            <h3>Speed</h3>
            <p>Track player sprint speed and acceleration during the match</p>
          </label>
        </div>
        
        <div className="parameter-card">
          <input 
            type="checkbox" 
            id="distanceCovered" 
            name="distanceCovered"
            checked={parameters.distanceCovered}
            onChange={handleParameterChange}
          />
          <label htmlFor="distanceCovered">
            <div className="parameter-icon">📏</div>
            <h3>Distance Covered</h3>
            <p>Track total distance run by each player</p>
          </label>
        </div>
        
        <div className="parameter-card">
          <input 
            type="checkbox" 
            id="passAccuracy" 
            name="passAccuracy"
            checked={parameters.passAccuracy}
            onChange={handleParameterChange}
          />
          <label htmlFor="passAccuracy">
            <div className="parameter-icon">🎯</div>
            <h3>Pass Accuracy</h3>
            <p>Analyze successful passes and pass completion rate</p>
          </label>
        </div>
      </div>
      
      <div className="video-upload-section">
        <h3>Match Video Analysis</h3>
        <p>Upload a video of your match for enhanced analysis</p>
        
        <div className="upload-container">
          <label htmlFor="videoUpload" className="upload-label">
            {videoFile ? videoFile.name : 'Select Video File'}
            <input 
              type="file" 
              id="videoUpload" 
              accept="video/*"
              onChange={handleVideoChange}
              className="file-input"
              required
            />
          </label>
          {videoFile && (
            <button 
              type="button" 
              className="remove-file-btn"
              onClick={() => setVideoFile(null)}
            >
              Remove
            </button>
          )}
        </div>
      </div>
      
      <div className="action-buttons">
        <button 
          type="button" 
          className="secondary-button"
          onClick={() => setStep(1)}
        >
          Back
        </button>
        <button 
          type="submit" 
          className="primary-button"
          disabled={isSubmitting || !Object.values(parameters).some(value => value === true) || !videoFile}
        >
          {isSubmitting ? 'Processing...' : 'Start Analysis'}
        </button>
      </div>
    </form>
  );
  
  const renderProcessingStep = () => (
    <div className="processing-container">
      <div className="loading-spinner"></div>
      <h2>Processing Video</h2>
      <p>Your video is being analyzed. This may take several minutes depending on the length of your video.</p>
      <p>We're tracking player movements, analyzing team formations, and calculating performance metrics.</p>
    </div>
  );
  
  const renderVideo = () => {
    if (!outputVideoUrl) return null;
    
    // Create separate download URLs
    const downloadUrl = outputVideoUrl.replace('/output_videos/', '/download_video/');
    const altDownloadUrl = altVideoUrl ? altVideoUrl.replace('/output_videos/', '/download_video/') : null;
    
    return (
      <div className="video-container">
        <h3>Processed Video</h3>
        
        {/* Video Element - Try to use video tag directly again with proper attributes */}
        <div className="video-player-container">
          <video 
            ref={videoRef}
            controls 
            autoPlay={false}
            playsInline
            preload="auto"
            width="100%" 
            height="auto"
            className="video-player"
          >
            <source src={outputVideoUrl} type={videoFormat === 'mp4' ? 'video/mp4' : 'video/x-msvideo'} />
            Your browser does not support the video tag.
          </video>
        </div>
        
        {/* Fallback iframe if video tag fails */}
        {!videoAvailable && (
          <div className="video-frame-container" style={{ marginTop: '20px' }}>
            <iframe 
              src={outputVideoUrl}
              className="video-frame"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              onError={handleVideoError as React.ReactEventHandler<HTMLIFrameElement>}
            ></iframe>
          </div>
        )}
        
        {/* Video Download Options */}
        <div className="video-options">
          <div className="video-actions center-actions">
            <a 
              href={downloadUrl}
              className="download-link primary-download"
              target="_blank"
              rel="noopener noreferrer"
            >
              Download Video ({videoFormat.toUpperCase()})
            </a>
            
            {altVideoUrl && altDownloadUrl && (
              <a 
                href={altDownloadUrl}
                className="download-link alt-format"
                target="_blank"
                rel="noopener noreferrer"
              >
                Alternative Format ({videoFormat === 'mp4' ? 'AVI' : 'MP4'})
              </a>
            )}
            
            <button 
              className="secondary-button"
              onClick={() => {
                // Try to reinitialize the video player
                if (videoRef.current) {
                  videoRef.current.load();
                  setTimeout(() => {
                    if (videoRef.current) {
                      const playPromise = videoRef.current.play();
                      if (playPromise !== undefined) {
                        playPromise.then(_ => {
                          console.log("Video playback started successfully");
                          setVideoAvailable(true);
                        })
                        .catch(e => {
                          console.error("Video playback failed:", e);
                          setVideoAvailable(false);
                        });
                      }
                    }
                  }, 1000);
                }
              }}
            >
              Reload Video
            </button>
          </div>
          
          <p className="small-text">
            If the video doesn't play in your browser, you can download it to view on your device's media player.
          </p>
        </div>
      </div>
    );
  };
  
  const renderStepThree = () => (
    <div className="setup-complete">
      <div className="success-icon">✓</div>
      <h2>Analysis Complete!</h2>
      <p>Your team "{teamName}" vs "{opponentTeamName}" has been analyzed successfully.</p>
      <p className="parameters-summary">
        Selected parameters: {Object.entries(parameters)
          .filter(([, value]) => value)
          .map(([key]) => {
            const formatted = key.replace(/([A-Z])/g, ' $1');
            return formatted.charAt(0).toUpperCase() + formatted.slice(1);
          })
          .join(', ')}
      </p>
      
      {renderVideo()}
      
      {recommendation && (
        <div className="recommendation-container">
          <h3>Substitution Recommendations</h3>
          <div className="recommendation-cards">
            <div className="recommendation-card underperformer">
              <div className="card-header">
                <h4>Player to Substitute</h4>
                <div className="team-badge">{recommendation.underperformer.team_name}</div>
              </div>
              <div className="player-info centered">
                <div className="player-number">#{recommendation.underperformer.track_id}</div>
                <div className="player-team">{recommendation.underperformer.team_name}</div>
              </div>
            </div>
            
            <div className="recommendation-arrow">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            
            <div className="recommendation-card replacement">
              <div className="card-header">
                <h4>Recommended Replacement</h4>
                <div className="team-badge">{recommendation.recommendation.team}</div>
              </div>
              <div className="player-info">
                <div className="player-name">{recommendation.recommendation.player_name}</div>
                <div className="player-id">ID: {recommendation.recommendation.player_id}</div>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-label">Avg. Speed</span>
                    <span className="stat-value">{typeof recommendation.recommendation.avg_speed === 'number' ? 
                      recommendation.recommendation.avg_speed.toFixed(2) + ' km/h' : 'N/A'}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Avg. Distance</span>
                    <span className="stat-value">{typeof recommendation.recommendation.avg_distance === 'number' ? 
                      recommendation.recommendation.avg_distance.toFixed(2) + ' km' : 'N/A'}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Pass Accuracy</span>
                    <span className="stat-value">{typeof recommendation.recommendation.passing_ratio === 'number' ? 
                      (recommendation.recommendation.passing_ratio * 100).toFixed(1) + '%' : 'N/A'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Performance Details */}
          <div className="performance-comparison">
            <h4>Replacement Player Details</h4>
            <div className="player-details">
              <div className="player-profile">
                <div className="player-avatar">
                  {recommendation.recommendation.player_name.charAt(0)}
                </div>
                <div className="player-identity">
                  <h5>{recommendation.recommendation.player_name}</h5>
                  <p>{recommendation.recommendation.team}</p>
                  <p className="player-id-small">ID: {recommendation.recommendation.player_id}</p>
                </div>
              </div>
              
              <div className="player-stats">
                {typeof recommendation.recommendation.avg_speed === 'number' && (
                  <div className="stat-detail">
                    <div className="stat-icon">🏃</div>
                    <div className="stat-info">
                      <div className="stat-title">Average Speed</div>
                      <div className="stat-value-large">{recommendation.recommendation.avg_speed.toFixed(1)} <span>km/h</span></div>
                    </div>
                  </div>
                )}
                
                {typeof recommendation.recommendation.avg_distance === 'number' && (
                  <div className="stat-detail">
                    <div className="stat-icon">📏</div>
                    <div className="stat-info">
                      <div className="stat-title">Average Distance</div>
                      <div className="stat-value-large">{recommendation.recommendation.avg_distance.toFixed(1)} <span>km</span></div>
                    </div>
                  </div>
                )}
                
                {typeof recommendation.recommendation.passing_ratio === 'number' && (
                  <div className="stat-detail">
                    <div className="stat-icon">🎯</div>
                    <div className="stat-info">
                      <div className="stat-title">Pass Accuracy</div>
                      <div className="stat-value-large">{(recommendation.recommendation.passing_ratio * 100).toFixed(1)}<span>%</span></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      
      <button 
        className="primary-button"
        onClick={() => {
          // Reset the state and go back to step 1
          setTeamName('');
          setOpponentTeamName('');
          setParameters({
            speed: false,
            distanceCovered: false,
            passAccuracy: false
          });
          setVideoFile(null);
          setOutputVideoUrl(null);
          setAltVideoUrl(null);
          setRecommendation(null);
          setProcessingStatus('idle');
          setVideoAvailable(false);
          setStep(1);
        }}
      >
        Start New Analysis
      </button>
    </div>
  );

  const renderErrorState = () => (
    <div className="error-container">
      <div className="error-icon">❌</div>
      <h2>Processing Error</h2>
      <p>There was an error processing your video:</p>
      <div className="error-message-box">
        {errorMessage || 'Unknown error occurred'}
      </div>
      <button 
        className="primary-button"
        onClick={() => {
          setProcessingStatus('idle');
          setStep(2); // Go back to upload step
        }}
      >
        Try Again
      </button>
    </div>
  );
  
  const renderCurrentStep = () => {
    if (processingStatus === 'processing') {
      return renderProcessingStep();
    }
    if (processingStatus === 'error') {
      return renderErrorState();
    }
    
    switch (step) {
      case 1:
        return renderStepOne();
      case 2:
        return renderStepTwo();
      case 3:
        return renderStepThree();
      default:
        return renderStepOne();
    }
  };
  
  return (
    <div className="home-page">
      <header className="dashboard-header">
        <div className="logo">
          <h1>SmartSub</h1>
        </div>
        <div className="user-info">
          <span className="user-greeting">Coach Dashboard</span>
        </div>
      </header>
      
      <main className="dashboard-content">
        <div className="setup-wizard">
          <div className="steps-indicator">
            <div className={`step ${step >= 1 ? 'active' : ''} ${step > 1 ? 'completed' : ''}`}>
              <div className="step-number">1</div>
              <div className="step-label">Team Setup</div>
            </div>
            <div className={`step ${step >= 2 ? 'active' : ''} ${step > 2 ? 'completed' : ''}`}>
              <div className="step-number">2</div>
              <div className="step-label">Parameters</div>
            </div>
            <div className={`step ${step >= 3 ? 'active' : ''}`}>
              <div className="step-number">3</div>
              <div className="step-label">Results</div>
            </div>
          </div>
          
          {renderCurrentStep()}
        </div>
      </main>
      
      <footer className="dashboard-footer">
        <p>&copy; 2023 SmartSub. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default HomePage; 