# camera_movement_estimator/camera_movement_estimator.py

import pickle
import cv2
import numpy as np
import os
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,first_frame_for_init):
        self.minimum_distance = 5 
        self.lk_params = dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
        if first_frame_for_init is None:
            raise ValueError("CameraMovementEstimator requires a valid frame for initialization.")
        try:
            first_frame_grayscale = cv2.cvtColor(first_frame_for_init,cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            raise ValueError(f"Could not convert first_frame_for_init to grayscale: {e}")
        mask_features = np.zeros_like(first_frame_grayscale)
        h, w = first_frame_grayscale.shape
        mask_features[:, 0:min(20, w//10)] = 1  
        mask_features[:, max(0, w - min(20, w//10)):w] = 1 
        self.features_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7, mask=mask_features)
    
    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        if not tracks or not camera_movement_per_frame: return
        for object_name, object_tracks_list in tracks.items():
            if not isinstance(object_tracks_list, list): continue
            for frame_num, track_dict_in_frame in enumerate(object_tracks_list):
                if not isinstance(track_dict_in_frame, dict) or frame_num >= len(camera_movement_per_frame): continue
                camera_movement = camera_movement_per_frame[frame_num]
                if not (isinstance(camera_movement, (list, np.ndarray)) and len(camera_movement) == 2): continue
                for track_id, track_info in track_dict_in_frame.items():
                    if isinstance(track_info, dict) and 'position' in track_info and track_info['position'] is not None:
                        position = track_info['position']
                        if isinstance(position, (list, tuple)) and len(position) == 2:
                            try:
                                pos_x_adj, pos_y_adj = position[0] - camera_movement[0], position[1] - camera_movement[1]
                                if isinstance(tracks.get(object_name), list) and \
                                   frame_num < len(tracks[object_name]) and \
                                   isinstance(tracks[object_name][frame_num], dict) and \
                                   track_id in tracks[object_name][frame_num] and \
                                   isinstance(tracks[object_name][frame_num][track_id], dict):
                                    tracks[object_name][frame_num][track_id]['position_adjusted'] = (pos_x_adj, pos_y_adj)
                            except (TypeError, IndexError): pass

    def get_camera_movement(self, frames_list_for_estimation, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path,'rb') as f: return pickle.load(f)
            except Exception as e: 
                print(f"Error loading CME stub {stub_path}: {e}")
                return [] if frames_list_for_estimation is None else [[0.0,0.0]]*len(frames_list_for_estimation)
        if frames_list_for_estimation is None or len(frames_list_for_estimation) < 2:
             return [] if frames_list_for_estimation is None else [[0.0,0.0]]*len(frames_list_for_estimation)

        num_frames = len(frames_list_for_estimation)
        camera_movement_data = [[0.0,0.0]] * num_frames
        try:
            old_gray = cv2.cvtColor(frames_list_for_estimation[0],cv2.COLOR_BGR2GRAY)
            old_features = cv2.goodFeaturesToTrack(old_gray, **self.features_params)
        except Exception as e:
            print(f"Error processing first frame in get_camera_movement: {e}")
            return camera_movement_data
        
        for frame_num in range(1, num_frames):
            current_frame_image = frames_list_for_estimation[frame_num]
            if current_frame_image is None: camera_movement_data[frame_num] = [0.0,0.0]; continue
            try:
                frame_gray = cv2.cvtColor(current_frame_image,cv2.COLOR_BGR2GRAY)
                if old_features is None or len(old_features) == 0:
                    if old_gray is not None: old_features = cv2.goodFeaturesToTrack(old_gray,**self.features_params)
                    if old_features is None or len(old_features) == 0: 
                        camera_movement_data[frame_num] = [0.0,0.0]; old_gray = frame_gray.copy(); continue 
                new_features, status_array, _ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)
                max_dist_val, cam_mv_x, cam_mv_y = 0.0, 0.0, 0.0
                if new_features is not None and status_array is not None:
                    good_new_features = new_features[status_array.ravel()==1] 
                    good_old_features = old_features[status_array.ravel()==1]
                    for new_pt_coords,old_pt_coords in zip(good_new_features,good_old_features):
                        new_fp, old_fp = new_pt_coords.ravel(), old_pt_coords.ravel()
                        distance = measure_distance(new_fp,old_fp)
                        if distance > max_dist_val:
                            max_dist_val = distance
                            cam_mv_x,cam_mv_y = measure_xy_distance(old_fp, new_fp) 
                if max_dist_val > self.minimum_distance:
                    camera_movement_data[frame_num] = [cam_mv_x,cam_mv_y]
                    old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features_params) 
                else: 
                    old_features = good_new_features.reshape(-1,1,2) if 'good_new_features' in locals() and good_new_features.size > 0 else None
                old_gray = frame_gray.copy()
            except Exception as e_gen:
                print(f"Error processing frame {frame_num} in get_camera_movement: {e_gen}")
                camera_movement_data[frame_num] = [0.0,0.0]
                if 'frame_gray' in locals() and frame_gray is not None: old_gray = frame_gray.copy()
                else: old_gray = None
                old_features = None
        if stub_path is not None and not read_from_stub: 
            try:
                with open(stub_path,'wb') as f: pickle.dump(camera_movement_data,f)
            except Exception as e: print(f"Error saving CME data to stub {stub_path}: {e}")
        return camera_movement_data