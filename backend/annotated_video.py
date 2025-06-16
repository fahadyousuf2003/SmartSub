import os
import cv2

# --- Configuration ---
# Path to directory containing annotated frames (JPEGs)
ANNOTATED_FRAMES_INPUT_DIR = r'D:\Projects\Personal Projects\Football_Substitution_Planning\annotated_frames_dir\england_epl_2016-2017_2016-09-17 - 17-00 Hull City 1 - 4 Arsenal_1'
# Output video file path
OUTPUT_VIDEO_PATH = r'D:\Projects\Personal Projects\Football_Substitution_Planning\my_video_id_output.mp4'
# Frames per second for the output video
TARGET_FPS = 25.0


def make_video_from_frames(frames_dir: str, output_path: str, fps: float):
    # List and sort frame files (assumes .jpg extension)
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith('.jpg') or f.lower().endswith('.png')
    ])
    if not frame_files:
        raise ValueError(f"No image frames found in directory: {frames_dir}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    if first_frame is None:
        raise IOError(f"Cannot read first frame: {frame_files[0]}")
    height, width, _ = first_frame.shape
    frame_size = (width, height)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Write each frame
    for idx, fname in enumerate(frame_files, start=1):
        frame_path = os.path.join(frames_dir, fname)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: skipping unreadable frame: {fname}")
            continue
        video_writer.write(frame)
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(frame_files)} frames...")

    video_writer.release()
    print(f"Video saved to: {output_path} (FPS: {fps}, Size: {frame_size})")


if __name__ == '__main__':
    make_video_from_frames(ANNOTATED_FRAMES_INPUT_DIR, OUTPUT_VIDEO_PATH, TARGET_FPS)
