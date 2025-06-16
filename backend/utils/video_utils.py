# utils/video_utils.py

import cv2

def read_video(video_path):
    """
    Reads all frames from a video file into a list.
    (Not used by the frame-by-frame main.py but kept for utility)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path, fps=24):
    """
    Saves a list of frames to a video file.
    (Not used by the frame-by-frame main.py but kept for utility)
    """
    if not output_video_frames:
        print("Warning: No frames provided to save_video.")
        return
        
    try:
        frame_height, frame_width = output_video_frames[0].shape[:2]
    except Exception as e:
        print(f"Error getting frame dimensions from output_video_frames: {e}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Defaulting to MJPG for .avi
    if output_video_path.lower().endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for path {output_video_path}")
        return

    for frame in output_video_frames:
        if frame is not None:
            out.write(frame)
    out.release()