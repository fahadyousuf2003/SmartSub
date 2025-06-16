from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import cv2
import numpy as np
import os
import shutil
import uuid
import asyncio
from datetime import datetime
import json
import traceback
import pickle
import tempfile
import subprocess

# Import your existing modules
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistance_Estimator
from pass_tracker import PassTracker
from recommendation import RecommenderSystem

app = FastAPI(title="Soccer Analysis API - Streamlined")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "temp_uploads"
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_videos"
WEIGHTS_DIR = "weights"
STUBS_DIR = "stubs"
DB_URI = "mongodb+srv://ibadkhan06:iffatbadar@cluster0.kk71xnf.mongodb.net/FYP_DB?retryWrites=true&w=majority"
YOLO_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "player_and_ball_detection.pt")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STUBS_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

# Define team mappings for specific videos
TEAM_MAPPINGS = {
    "0.mp4": {1: "Arsenal", 2: "Liverpool"}
}

# Define API models
class AnalysisRequest(BaseModel):
    video_filename: str
    team1: str
    team2: str
    substitution_metric: str
    selected_team: Optional[str] = None
    use_stubs: bool = True

class AnalysisResponse(BaseModel):
    success: bool
    video_id: Optional[str] = None
    output_video_url: Optional[str] = None
    recommendation: Optional[Dict] = None
    error: Optional[str] = None


class VideoAnalysisProcessor:
    def __init__(self):
        self.tracker = None
        self.camera_estimator = None
        self.pass_tracker = None
        self.recommender = None
    
    def get_stub_paths(self, video_filename):
        """Get the paths for stub files based on video filename"""
        # Create a subfolder for this video if it doesn't exist
        video_stub_dir = os.path.join(STUBS_DIR, os.path.splitext(video_filename)[0])
        os.makedirs(video_stub_dir, exist_ok=True)
        
        # Define paths for the stub files
        tracks_stub_path = os.path.join(video_stub_dir, "stubs_tracks.pkl")
        camera_stub_path = os.path.join(video_stub_dir, "stubs_camera.pkl")
        
        return {
            "tracks": tracks_stub_path,
            "camera": camera_stub_path,
            "dir": video_stub_dir
        }
    
    def load_stub(self, stub_path, stub_type):
        """Load a stub file if it exists"""
        if os.path.exists(stub_path):
            try:
                with open(stub_path, 'rb') as f:
                    print(f"Loading {stub_type} stub from: {stub_path}")
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading {stub_type} stub from {stub_path}: {e}")
        return None
    
    def save_stub(self, data, stub_path, stub_type):
        """Save data to a stub file"""
        try:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir and not os.path.exists(stub_dir):
                os.makedirs(stub_dir)
            
            with open(stub_path, 'wb') as f:
                pickle.dump(data, f)
                print(f"Saved {stub_type} stub to: {stub_path}")
        except Exception as e:
            print(f"Error saving {stub_type} stub to {stub_path}: {e}")
    
    def initialize_components(self, first_frame):
        """Initialize all processing components"""
        print("Initializing components...")
        self.tracker = Tracker(YOLO_WEIGHTS_PATH)
        self.camera_estimator = CameraMovementEstimator(first_frame)
        self.pass_tracker = PassTracker()
        
        # Initialize recommender with DB connection
        try:
            self.recommender = RecommenderSystem(db_uri=DB_URI)
            if self.recommender.client:
                self.recommender.client.admin.command('ping')
                print("Recommender system initialized and connected to DB.")
            else:
                print("RecommenderSystem initialized, but MongoDB connection failed.")
                self.recommender = None
        except Exception as e:
            print(f"Failed to initialize RecommenderSystem: {e}")
            self.recommender = None
    
    def generate_tracks(self, cap, total_frames):
        """Generate tracking data for all frames"""
        all_tracks = {"players": [], "referees": [], "ball": []}
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Initialize tracker
        ret, first_frame = cap.read()
        if not ret:
            raise Exception("Could not read first frame for tracker initialization")
        self.tracker._initialize_class_names_if_needed(first_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process frames
        print(f"Processing {total_frames} frames for tracking...")
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Progress update
            if frame_idx % 100 == 0:
                print(f"Tracking frame {frame_idx}/{total_frames}")
            
            # Detect and track
            frame_tracks = self.tracker.detect_and_track_one_frame(frame)
            all_tracks["players"].append(frame_tracks.get("players", {}))
            all_tracks["referees"].append(frame_tracks.get("referees", {}))
            all_tracks["ball"].append(frame_tracks.get("ball", {}))
        
        return all_tracks
    
    def generate_camera_movement(self, cap, total_frames):
        """Generate camera movement data"""
        all_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print(f"Processing {total_frames} frames for camera movement...")
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        
        if not all_frames:
            return [[0.0, 0.0]] * total_frames
        
        camera_movement = self.camera_estimator.get_camera_movement(all_frames, read_from_stub=False)
        return camera_movement
    
    def process_tracks(self, tracks, camera_movement, team_mapping, cap):
        """Process tracks with all transformations and assignments"""
        print("Processing tracks with transformations...")
        
        # Add positions to tracks
        self.tracker.add_position_to_tracks(tracks)
        
        # Adjust for camera movement
        self.camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
        
        # Transform view
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)
        
        # Interpolate ball positions
        if "ball" in tracks and tracks["ball"]:
            tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
        
        # Add speed and distance
        speed_dist_estimator = SpeedAndDistance_Estimator()
        speed_dist_estimator.add_speed_and_distance_to_tracks(tracks)
        
        # Team assignment
        team_assigner = TeamAssigner()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if ret and tracks.get('players') and tracks['players'][0]:
            team_assigner.assign_team_color(first_frame, tracks['players'][0])
        
        # Ball assignment
        ball_assigner = PlayerBallAssigner()
        
        # Process each frame for team and ball assignments
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(len(tracks['players'])):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Assign teams
            if 'players' in tracks and frame_idx < len(tracks['players']):
                for player_id, player_info in tracks['players'][frame_idx].items():
                    if isinstance(player_info, dict) and 'bbox' in player_info:
                        team_id = team_assigner.get_player_team(frame, player_info['bbox'], player_id)
                        team_name = team_mapping.get(team_id, f"Team ID {team_id}")
                        tracks['players'][frame_idx][player_id]['team'] = team_id
                        tracks['players'][frame_idx][player_id]['team_name'] = team_name
                        if team_id in team_assigner.team_colors:
                            tracks['players'][frame_idx][player_id]['team_color'] = team_assigner.team_colors[team_id]
            
            # Assign ball possession
            players_data = tracks['players'][frame_idx] if frame_idx < len(tracks['players']) else {}
            ball_bbox = None
            if 'ball' in tracks and frame_idx < len(tracks['ball']) and \
               1 in tracks['ball'][frame_idx] and isinstance(tracks['ball'][frame_idx][1], dict):
                ball_bbox = tracks['ball'][frame_idx][1].get('bbox')
            
            if players_data and ball_bbox:
                assigned_player = ball_assigner.assign_ball_to_player(players_data, ball_bbox)
                if assigned_player != -1 and assigned_player in players_data:
                    players_data[assigned_player]['has_ball'] = True
        
        # Calculate passes
        self.pass_tracker.calculate_passes(tracks)
        print("Track processing completed")
    
    def generate_output_video(self, cap, all_tracks, output_path, frame_w, frame_h, fps, total_frames):
        """Generate the output video with annotations"""
        print("Generating output video...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Ensure tracker is initialized
        if self.tracker is None:
            print("Tracker not initialized, initializing it now...")
            ret, first_frame = cap.read()
            if not ret:
                raise Exception("Could not read first frame for tracker initialization")
            self.tracker = Tracker(YOLO_WEIGHTS_PATH)
            self.tracker._initialize_class_names_if_needed(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create intermediate AVI file first with XVID codec (most compatible with OpenCV)
        temp_avi_path = os.path.join(tempfile.gettempdir(), f"temp_output_{uuid.uuid4()}.avi")
        print(f"Generating intermediate AVI file at: {temp_avi_path}")
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(temp_avi_path, fourcc, fps, (frame_w, frame_h))
        
        if not out_video.isOpened():
            raise Exception("Could not open video writer")
        
        # Write frames with annotations
        team_ball_control = []
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Progress update
            if frame_idx % 100 == 0:
                print(f"Writing frame {frame_idx}/{total_frames}")
            
            # Get current ball possession
            current_team_control = 0
            if 'players' in all_tracks and frame_idx < len(all_tracks['players']):
                for player_info in all_tracks['players'][frame_idx].values():
                    if isinstance(player_info, dict) and player_info.get('has_ball', False):
                        current_team_control = player_info.get('team', 0)
                        break
            team_ball_control.append(current_team_control)
            
            # Draw annotations
            team_ball_control_np = np.array(team_ball_control, dtype=int)
            self.tracker.draw_annotations_on_single_frame(frame, frame_idx, all_tracks, team_ball_control_np)
            
            # Add speed/distance annotations
            speed_dist_estimator = SpeedAndDistance_Estimator()
            speed_dist_estimator.draw_speed_and_distance_on_single_frame(frame, frame_idx, all_tracks)
            
            # Add pass annotations
            self.pass_tracker.draw_passes_on_single_frame(frame, frame_idx, all_tracks)
            
            out_video.write(frame)
        
        out_video.release()
        print("Intermediate video generated successfully")
        
        # Now use FFmpeg to convert to a web-compatible MP4 format
        # If FFmpeg is not installed, we'll just copy the AVI file
        final_output_paths = {
            'mp4': output_path,
            'avi': os.path.splitext(output_path)[0] + '.avi'
        }
        
        # Make sure both output paths are accessible
        for path in final_output_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Try to use FFmpeg to convert AVI to MP4
        try:
            # Create a web-compatible MP4 with H.264 and AAC
            print(f"Converting to web-compatible MP4: {final_output_paths['mp4']}")
            # Set higher bitrate to ensure quality
            bitrate = "2M"
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", temp_avi_path, 
                "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart", # This is critical for web playback
                "-f", "mp4", # Force MP4 format
                final_output_paths['mp4']
            ]
            
            print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            ffmpeg_result = subprocess.run(
                ffmpeg_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            if ffmpeg_result.returncode != 0:
                print(f"FFmpeg error: {ffmpeg_result.stderr.decode('utf-8')}")
                raise Exception("FFmpeg conversion failed")
            
            print("FFmpeg conversion successful")
            
            # Also copy the original AVI as fallback
            shutil.copy(temp_avi_path, final_output_paths['avi'])
            
            # Return MP4 path as primary but keep AVI as backup
            main_output = final_output_paths['mp4']
            alt_output = final_output_paths['avi']
        
        except Exception as e:
            print(f"Error during FFmpeg conversion: {e}")
            print("Falling back to AVI format only")
            # If FFmpeg fails, just use the AVI file
            shutil.copy(temp_avi_path, final_output_paths['avi'])
            main_output = final_output_paths['avi']
            alt_output = None
        
        # Clean up temp file
        try:
            if os.path.exists(temp_avi_path):
                os.remove(temp_avi_path)
        except Exception as e:
            print(f"Failed to remove temporary file: {e}")
        
        print(f"Video processing complete. Primary output: {main_output}, Alternative: {alt_output}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset frame position
        
        return {
            'main': main_output,
            'alt': alt_output,
            'main_url': f"/output_videos/{os.path.basename(main_output)}",
            'alt_url': f"/output_videos/{os.path.basename(alt_output)}" if alt_output else None
        }

    async def process_video_with_analysis(self, video_id: str, video_path: str, output_path: str, 
                                        team1: str, team2: str, substitution_metric: str, 
                                        selected_team: Optional[str], use_stubs: bool = True, 
                                        video_filename: Optional[str] = None):
        """Process video with tracking, analysis, and recommendations"""
        print(f"Starting video processing for {video_id}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video {video_path}")
        
        try:
            # Get video properties
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video properties: {frame_w}x{frame_h}, {fps}fps, {total_frames} frames")
            
            # Read first frame for initialization
            ret, first_frame = cap.read()
            if not ret:
                raise Exception("Failed to read first frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Initialize components
            self.initialize_components(first_frame)
            
            # Get team mapping based on video or provided teams
            team_mapping = TEAM_MAPPINGS.get(video_filename, {1: team1, 2: team2})
            print(f"Using team mapping: {team_mapping}")
            
            # Get stub paths for this video
            stub_paths = self.get_stub_paths(video_filename) if video_filename else None
            
            # Step 1: Generate or load tracks
            all_tracks = None
            if use_stubs and stub_paths:
                all_tracks = self.load_stub(stub_paths["tracks"], "tracks")
            
            if all_tracks is None:
                print("No tracks stub found or stubs not used, generating tracks...")
                all_tracks = self.generate_tracks(cap, total_frames)
                if use_stubs and stub_paths:
                    self.save_stub(all_tracks, stub_paths["tracks"], "tracks")
            else:
                print("Loaded tracks from stub")
            
            # Step 2: Generate or load camera movement
            camera_movement = None
            if use_stubs and stub_paths:
                camera_movement = self.load_stub(stub_paths["camera"], "camera")
            
            if camera_movement is None:
                print("No camera movement stub found or stubs not used, generating...")
                camera_movement = self.generate_camera_movement(cap, total_frames)
                if use_stubs and stub_paths:
                    self.save_stub(camera_movement, stub_paths["camera"], "camera")
            else:
                print("Loaded camera movement from stub")
            
            # Adjust camera movement length if needed
            if len(camera_movement) != total_frames:
                print(f"Adjusting camera movement length from {len(camera_movement)} to {total_frames}")
                if len(camera_movement) > total_frames:
                    camera_movement = camera_movement[:total_frames]
                else:
                    camera_movement = camera_movement + [[0.0, 0.0]] * (total_frames - len(camera_movement))
            
            # Step 3: Process tracks (add metrics, team assignments, ball possession)
            print("Step 3: Processing tracks...")
            self.process_tracks(all_tracks, camera_movement, team_mapping, cap)
            
            # Step 4: Generate output video
            print("Step 4: Generating output video...")
            output_files = self.generate_output_video(cap, all_tracks, output_path, frame_w, frame_h, fps, total_frames)
            
            # Step 5: Generate recommendations
            print("Step 5: Generating recommendations...")
            recommendation = None
            if self.recommender:
                worst_performers, best_performers, player_metrics = self.recommender.evaluate_players(all_tracks)
                
                # Get substitution recommendation
                underperformer, db_recommendation = self.recommender.get_substitution_recommendation(
                    worst_performers, 
                    substitution_metric, 
                    selected_team
                )
                
                if underperformer and db_recommendation:
                    # Ensure all values are Python primitives (not NumPy types)
                    recommendation = {
                        "underperformer": {
                            "track_id": int(underperformer['track_id']) if isinstance(underperformer['track_id'], (np.integer, np.floating)) else underperformer['track_id'],
                            "team_name": str(underperformer.get('team_name', 'Unknown')),
                            "avg_speed": float(underperformer.get('avg_speed', 0)),
                            "total_distance": float(underperformer.get('total_distance', 0)),
                            "total_passes": int(underperformer.get('total_passes', 0)) if isinstance(underperformer.get('total_passes', 0), (np.integer, np.floating)) else underperformer.get('total_passes', 0),
                            "passing_ratio": float(underperformer.get('passing_ratio', 0))
                        },
                        "recommendation": {
                            "player_name": str(db_recommendation.get('player_name', 'N/A')),
                            "team": str(db_recommendation.get('team', 'Unknown')),
                            "player_id": str(db_recommendation.get('player_id', 'N/A')),
                            "avg_speed": float(db_recommendation.get('avg_speed_kmph', 0)) if isinstance(db_recommendation.get('avg_speed_kmph', 0), (np.integer, np.floating)) else db_recommendation.get('avg_speed_kmph', 'N/A'),
                            "avg_distance": float(db_recommendation.get('avg_distance_km', 0)) if isinstance(db_recommendation.get('avg_distance_km', 0), (np.integer, np.floating)) else db_recommendation.get('avg_distance_km', 'N/A'),
                            "passing_ratio": float(db_recommendation.get('passing_ratio', 0)) if isinstance(db_recommendation.get('passing_ratio', 0), (np.integer, np.floating)) else db_recommendation.get('passing_ratio', 'N/A')
                        }
                    }
            
            print("Processing completed successfully!")
            return {
                "success": True,
                "output_video_url": output_files['main_url'],
                "alt_video_url": output_files['alt_url'],
                "recommendation": recommendation
            }
            
        except Exception as e:
            print(f"Error processing video: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if cap.isOpened():
                cap.release()


# API Endpoints
@app.post("/api/analyze")
async def analyze_video(
    file: Optional[UploadFile] = File(None),
    video_filename: Optional[str] = Form(None),
    team1: str = Form(...),
    team2: str = Form(...),
    substitution_metric: str = Form(...),  # "1" through "7"
    selected_team: Optional[str] = Form(None),  # Optional team selection
    use_stubs: bool = Form(True)  # Whether to use stubs if available
):
    """
    Single endpoint to handle video analysis. Can accept either:
    1. A direct file upload via the 'file' parameter
    2. A reference to an existing file in the input_videos folder via 'video_filename'
    """
    video_id = str(uuid.uuid4())
    actual_filename = None
    
    try:
        # Clean up any existing output videos to prevent confusion
        for ext in ['.mp4', '.avi']:
            existing_file = os.path.join(OUTPUT_DIR, f"output_video{ext}")
            if os.path.exists(existing_file):
                try:
                    os.remove(existing_file)
                    print(f"Removed existing output file: {existing_file}")
                except Exception as e:
                    print(f"Failed to remove existing file {existing_file}: {e}")
        
        # Determine video source
        if file and file.filename:
            # Handle uploaded file
            actual_filename = file.filename
            input_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"Video uploaded to: {input_path}")
        elif video_filename:
            # Use existing file from input_videos folder
            actual_filename = video_filename
            input_path = os.path.join(INPUT_DIR, video_filename)
            if not os.path.exists(input_path):
                raise HTTPException(status_code=404, detail=f"Video file '{video_filename}' not found in input_videos folder")
            print(f"Using existing video: {input_path}")
        else:
            raise HTTPException(status_code=400, detail="Either file upload or video_filename must be provided")
        
        # Prepare output path - use fixed filename "output_video.mp4"
        output_filename = "output_video.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Process video
        processor = VideoAnalysisProcessor()
        result = await processor.process_video_with_analysis(
            video_id=video_id,
            video_path=input_path,
            output_path=output_path,
            team1=team1,
            team2=team2,
            substitution_metric=substitution_metric,
            selected_team=selected_team,
            use_stubs=use_stubs,
            video_filename=actual_filename
        )
        
        if result["success"]:
            return {
                "success": True,
                "video_id": video_id,
                "output_video_url": result["output_video_url"],
                "alt_video_url": result["alt_video_url"],
                "recommendation": result["recommendation"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        # Log the error
        print(f"Error during video analysis: {str(e)}")
        traceback.print_exc()
        
        # Clean up uploaded file on error
        if file and file.filename:
            try:
                if 'input_path' in locals() and os.path.exists(input_path):
                    os.remove(input_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos")
async def list_input_videos():
    """List all available videos in the input_videos folder"""
    try:
        videos = []
        for filename in os.listdir(INPUT_DIR):
            if os.path.isfile(os.path.join(INPUT_DIR, filename)) and any(
                filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']
            ):
                # Check if stubs exist for this video
                video_name = os.path.splitext(filename)[0]
                stub_dir = os.path.join(STUBS_DIR, video_name)
                has_tracks_stub = os.path.exists(os.path.join(stub_dir, "stubs_tracks.pkl"))
                has_camera_stub = os.path.exists(os.path.join(stub_dir, "stubs_camera.pkl"))
                
                videos.append({
                    "filename": filename,
                    "size_mb": round(os.path.getsize(os.path.join(INPUT_DIR, filename)) / (1024 * 1024), 2),
                    "has_stubs": has_tracks_stub and has_camera_stub,
                    "stub_info": {
                        "tracks": has_tracks_stub,
                        "camera": has_camera_stub
                    }
                })
        return {"videos": videos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/check-stubs/{video_filename}")
async def check_video_stubs(video_filename: str):
    """Check if stubs exist for the specified video"""
    try:
        video_path = os.path.join(INPUT_DIR, video_filename)
        # Check if video exists
        if not os.path.exists(video_path):
            return {
                "video_exists": False,
                "stubs_exist": False,
                "message": f"Video file '{video_filename}' not found in input_videos folder"
            }
            
        # Check if stubs exist
        video_name = os.path.splitext(video_filename)[0]
        stub_dir = os.path.join(STUBS_DIR, video_name)
        tracks_stub_path = os.path.join(stub_dir, "stubs_tracks.pkl")
        camera_stub_path = os.path.join(stub_dir, "stubs_camera.pkl")
        
        has_tracks_stub = os.path.exists(tracks_stub_path)
        has_camera_stub = os.path.exists(camera_stub_path)
        has_stubs = has_tracks_stub and has_camera_stub
        
        return {
            "video_exists": True,
            "stubs_exist": has_stubs,
            "stub_info": {
                "tracks": {
                    "exists": has_tracks_stub,
                    "path": tracks_stub_path if has_tracks_stub else None,
                    "size_mb": round(os.path.getsize(tracks_stub_path) / (1024 * 1024), 2) if has_tracks_stub else 0
                },
                "camera": {
                    "exists": has_camera_stub,
                    "path": camera_stub_path if has_camera_stub else None,
                    "size_mb": round(os.path.getsize(camera_stub_path) / (1024 * 1024), 2) if has_camera_stub else 0
                }
            },
            "message": "All stubs found" if has_stubs else 
                       "Partial stubs found" if (has_tracks_stub or has_camera_stub) else 
                       "No stubs available for this video"
        }
    except Exception as e:
        return {
            "video_exists": True,
            "stubs_exist": False,
            "error": str(e)
        }


@app.get("/api/download/{filename}")
async def download_video(filename: str):
    """Download processed video"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="video/avi",
        filename=filename
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/output_videos/{filename}")
async def get_video_file(filename: str, request: Request):
    """Serve any video file from the output_videos directory"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found")
    
    # Determine media type based on extension
    extension = os.path.splitext(filename)[1].lower()
    media_types = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska'
    }
    media_type = media_types.get(extension, 'application/octet-stream')
    
    # For direct browser rendering
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Accept-Ranges": "bytes",  # Always enable range requests
        "X-Content-Type-Options": "nosniff",  # Prevent MIME type sniffing
        "Content-Disposition": "inline",  # Force inline display, not download
        "Cache-Control": "no-cache, max-age=0"  # Prevent caching issues
    }
    
    # Handle range requests for better video seeking
    range_header = request.headers.get("Range", "")
    file_size = os.path.getsize(file_path)
    
    # If it's a range request, handle it properly
    if range_header:
        # Parse range header
        start, end = 0, file_size - 1
        if range_header.startswith("bytes="):
            ranges = range_header.replace("bytes=", "").split(",")[0].split("-")
            if ranges[0]:
                start = int(ranges[0])
            if ranges[1]:
                end = min(int(ranges[1]), file_size - 1)
        
        # Set correct headers for range request
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(end - start + 1)
        
        # For MP4 files, make sure we include content-type
        if extension == '.mp4':
            headers["Content-Type"] = "video/mp4"
        
        print(f"Serving file range {start}-{end}/{file_size}: {file_path}")
        
        # Use FastAPI's streaming response for range requests
        async def file_iterator():
            with open(file_path, 'rb') as f:
                f.seek(start)
                chunk_size = 8192  # 8KB chunks
                position = start
                while position <= end:
                    bytes_to_read = min(chunk_size, end - position + 1)
                    data = f.read(bytes_to_read)
                    if not data:
                        break
                    position += len(data)
                    yield data
        
        return StreamingResponse(
            file_iterator(),
            status_code=206,
            headers=headers,
            media_type=media_type
        )
    
    # For non-range requests, serve the entire file
    print(f"Serving entire file: {file_path} as {media_type}")
    return FileResponse(
        file_path,
        media_type=media_type,
        headers=headers
    )


@app.options("/output_videos/{filename}")
async def options_video_file(filename: str):
    """Handle OPTIONS requests for any video file endpoint"""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Max-Age": "86400",  # 24 hours
        "Accept-Ranges": "bytes"  # Explicitly enable range requests
    }
    return Response(status_code=200, headers=headers)


# Add a specific download endpoint that forces download
@app.get("/download_video/{filename}")
async def download_video_file(filename: str):
    """Force download of a video file"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Video file '{filename}' not found")
    
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Content-Disposition": f"attachment; filename={filename}"  # Force download
    }
    
    print(f"Serving file for download: {file_path}")
    return FileResponse(
        file_path,
        headers=headers
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)