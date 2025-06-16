# trackers/tracker.py

from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack() # ByteTrack is stateful for iterative tracking
        self.cls_names_cache = None
        self.cls_names_inv_cache = None
        # Attempt to initialize class names if model populates them on load
        if hasattr(self.model, 'names') and self.model.names:
             self.cls_names_cache = self.model.names
             self.cls_names_inv_cache = {v: k for k, v in self.cls_names_cache.items()}


    def _initialize_class_names_if_needed(self, frame_for_init_if_no_cache=None):
        """
        Ensures class names are initialized. 
        If frame_for_init_if_no_cache is provided and names are not set, 
        it runs a dummy predict.
        """
        if self.cls_names_cache is not None:
            return True # Already initialized

        print("Attempting to initialize model class names...")
        try:
            # Check if model.names got populated after __init__ or by a previous call
            if hasattr(self.model, 'names') and self.model.names:
                self.cls_names_cache = self.model.names
                self.cls_names_inv_cache = {v: k for k, v in self.cls_names_cache.items()}
            elif frame_for_init_if_no_cache is not None:
                # A single predict call should populate model.names
                _ = self.model.predict(frame_for_init_if_no_cache, verbose=False, conf=0.1) 
                if hasattr(self.model, 'names') and self.model.names:
                    self.cls_names_cache = self.model.names
                    self.cls_names_inv_cache = {v: k for k, v in self.cls_names_cache.items()}
                else:
                    raise AttributeError("model.names not populated after prediction.")
            else: # No frame provided and names not available
                 raise AttributeError("model.names not available and no frame provided for initialization.")
            
            print("Model class names initialized.")
            return True
        except Exception as e:
            print(f"Error initializing class names: {e}. Using fallback.")
            self.cls_names_cache = {0: 'player', 1: 'referee', 2: 'ball', 3: 'goalkeeper'} # Example
            self.cls_names_inv_cache = {v: k for k, v in self.cls_names_cache.items()}
            # Decide if this is a critical failure or if fallback is acceptable
            # For now, allowing fallback, but this might lead to incorrect class mapping
            return False # Indicate initialization used fallback or failed partially


    def detect_and_track_one_frame(self, frame_image):
        if self.cls_names_cache is None: # Attempt initialization if not done
            if not self._initialize_class_names_if_needed(frame_image):
                 print("Critical Error: Class names could not be initialized. Skipping tracking for this frame.")
                 return {"players": {}, "referees": {}, "ball": {}}

        current_frame_tracks = {"players": {}, "referees": {}, "ball": {}}
        try:
            detections_ultralytics = self.model.predict(frame_image, verbose=False, conf=0.1)[0]
            detection_supervision = sv.Detections.from_ultralytics(detections_ultralytics)

            for i, class_id in enumerate(detection_supervision.class_id):
                if self.cls_names_cache.get(class_id) == "goalkeeper" and "player" in self.cls_names_inv_cache:
                    detection_supervision.class_id[i] = self.cls_names_inv_cache["player"]
            
            tracked_detections = self.tracker.update_with_detections(detection_supervision)

            for det_track_info in tracked_detections:
                bbox = det_track_info[0].tolist()
                class_id_tracked = det_track_info[3]
                tracker_id = det_track_info[4]
                object_class_name = self.cls_names_cache.get(class_id_tracked)
                if object_class_name == 'player': current_frame_tracks["players"][tracker_id] = {"bbox": bbox}
                elif object_class_name == 'referee': current_frame_tracks["referees"][tracker_id] = {"bbox": bbox}
            
            for i in range(len(detection_supervision.xyxy)):
                class_id_ball = detection_supervision.class_id[i]
                if self.cls_names_cache.get(class_id_ball) == 'ball':
                    bbox_ball = detection_supervision.xyxy[i].tolist()
                    current_frame_tracks["ball"][1] = {"bbox": bbox_ball}; break
        except Exception as e:
            print(f"Error during detect_and_track_one_frame: {e}")
        return current_frame_tracks

    def load_tracks_from_stub(self, stub_path):
        if stub_path is not None and os.path.exists(stub_path):
            try:
                with open(stub_path,'rb') as f: tracks = pickle.load(f)
                if not isinstance(tracks, dict) or not all(key in tracks for key in ["players", "referees", "ball"]):
                    print(f"Warning: Stub file {stub_path} invalid structure. None returned.")
                    return None
                print(f"Tracks successfully loaded from {stub_path}")
                return tracks
            except Exception as e:
                print(f"Error loading stub {stub_path}: {e}. None returned.")
                return None
        print(f"Track stub not found: {stub_path}. None returned.")
        return None

    def save_tracks_to_stub(self, tracks_data, stub_path):
        try:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir and not os.path.exists(stub_dir): os.makedirs(stub_dir)
            with open(stub_path, 'wb') as f: pickle.dump(tracks_data, f)
            print(f"Tracks successfully saved to {stub_path}")
        except Exception as e:
            print(f"Error saving tracks to stub {stub_path}: {e}")

    def add_position_to_tracks(self,tracks):
        for object_name, object_tracks_list in tracks.items():
            if not isinstance(object_tracks_list, list): continue
            for frame_num, track_dict_in_frame in enumerate(object_tracks_list):
                if not isinstance(track_dict_in_frame, dict): continue
                for track_id, track_info in track_dict_in_frame.items():
                    if isinstance(track_info, dict) and 'bbox' in track_info and \
                       track_info['bbox'] is not None and len(track_info['bbox']) == 4:
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox) if object_name != 'ball' else get_center_of_bbox(bbox)
                        if isinstance(tracks.get(object_name), list) and \
                           frame_num < len(tracks[object_name]) and \
                           isinstance(tracks[object_name][frame_num], dict) and \
                           track_id in tracks[object_name][frame_num] and \
                           isinstance(tracks[object_name][frame_num][track_id], dict):
                            tracks[object_name][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions_list_of_dicts):
        if not isinstance(ball_positions_list_of_dicts, list) or not ball_positions_list_of_dicts:
            return ball_positions_list_of_dicts
        bboxes = []
        for frame_data in ball_positions_list_of_dicts:
            if isinstance(frame_data, dict) and 1 in frame_data and isinstance(frame_data[1], dict) and 'bbox' in frame_data[1]:
                bboxes.append(frame_data[1]['bbox'])
            else: bboxes.append([np.nan]*4) 
        df_ball_positions = pd.DataFrame(bboxes,columns=['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate(method='linear').bfill().ffill()
        if df_ball_positions.isnull().values.any(): df_ball_positions.fillna(0, inplace=True)
        return [{1: {"bbox": row}} for row in df_ball_positions.to_numpy().tolist()]

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        if not bbox or len(bbox) != 4: return frame
        try: 
            y2 = int(bbox[3])
            x_center, _ = get_center_of_bbox(bbox)
            width = get_bbox_width(bbox)
            if width <= 0: return frame
            cv2.ellipse(frame, center=(x_center,y2), axes=(int(width), int(0.35*width)),
                        angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)
            if track_id is not None:
                rect_w, rect_h = 40, 20
                x1_r, y1_r = x_center - rect_w//2, (y2 - rect_h//2) + 15 
                x2_r, y2_r = x_center + rect_w//2, (y2 + rect_h//2) + 15
                cv2.rectangle(frame, (int(x1_r),int(y1_r)), (int(x2_r),int(y2_r)), color, cv2.FILLED)
                str_track_id = str(track_id)
                font_scale = 0.6 if len(str_track_id) <= 2 else 0.5 
                (text_w_val, text_h_val), _ = cv2.getTextSize(str_track_id, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_x, text_y = x1_r + (rect_w - text_w_val)//2, y1_r + (rect_h + text_h_val)//2 - 2 
                cv2.putText(frame, str_track_id, (int(text_x), int(text_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2)
        except Exception as e: print(f"Error drawing ellipse for bbox {bbox}, track_id {track_id}: {e}")
        return frame

    def draw_traingle(self,frame,bbox,color): # Tip-down triangle
        if not bbox or len(bbox) != 4: return frame
        try:
            y_top_bbox = int(bbox[1])
            x_center_bbox, _ = get_center_of_bbox(bbox)
            tip_y, base_y, base_half_width = y_top_bbox, y_top_bbox - 20, 10
            points = np.array([[x_center_bbox, tip_y], [x_center_bbox - base_half_width, base_y], [x_center_bbox + base_half_width, base_y]])
            cv2.drawContours(frame, [points.astype(np.int32)],0,color, cv2.FILLED)
            cv2.drawContours(frame, [points.astype(np.int32)],0,(0,0,0), 2)
        except Exception as e: print(f"Error drawing triangle for bbox {bbox}: {e}")
        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control_history_array):
        h_frame, w_frame = frame.shape[:2]
        box_w, box_h, margin = 350, 70, 10
        x1_o, y1_o = w_frame - box_w - margin, margin
        x2_o, y2_o = w_frame - margin, margin + box_h
        alpha, color_bg = 0.6, (255,255,255)
        x1_c, y1_c, x2_c, y2_c = max(0,x1_o), max(0,y1_o), min(w_frame,x2_o), min(h_frame,y2_o)
        if x1_c < x2_c and y1_c < y2_c:
            try:
                roi = frame[y1_c:y2_c, x1_c:x2_c]
                rect = np.full(roi.shape, color_bg, dtype=np.uint8)
                cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0, roi)
            except Exception as e: print(f"Error drawing TBC background: {e}")
        t1_f, t2_f = 0,0
        if isinstance(team_ball_control_history_array, np.ndarray) and team_ball_control_history_array.size > 0 :
            ctrl = team_ball_control_history_array[:frame_num + 1] 
            if ctrl.size > 0: t1_f, t2_f = np.sum(ctrl == 1), np.sum(ctrl == 2)
        den = t1_f + t2_f
        t1_p,t2_p = ((t1_f/den)*100 if den>0 else 0.0), ((t2_f/den)*100 if den>0 else 0.0)
        f_scale, thick, color_txt = 0.7, 2, (0,0,0)
        spacing = int(box_h / 2.8); margin_l = x1_c + 10
        txt1_y = min(y1_c + spacing, y2_c - spacing - 5) if y2_c > y1_c + spacing else y1_c + int(box_h/2) -5
        txt2_y = min(y1_c + 2*spacing, y2_c - 5) if y2_c > y1_c + 2*spacing else y1_c + int(box_h*0.75) -5
        try:
            cv2.putText(frame, f"Team 1: {t1_p:.1f}%", (margin_l, txt1_y), cv2.FONT_HERSHEY_SIMPLEX, f_scale, color_txt, thick)
            cv2.putText(frame, f"Team 2: {t2_p:.1f}%", (margin_l, txt2_y), cv2.FONT_HERSHEY_SIMPLEX, f_scale, color_txt, thick)
        except Exception as e: print(f"Error drawing TBC text: {e}")
        return frame

    def draw_annotations_on_single_frame(self, frame, frame_num, all_tracks_data, team_ball_control_history_array):
        players_data = all_tracks_data.get("players"); balls_data = all_tracks_data.get("ball"); referees_data = all_tracks_data.get("referees")
        player_dict = players_data[frame_num] if isinstance(players_data,list) and frame_num < len(players_data) and isinstance(players_data[frame_num],dict) else {}
        ball_dict = balls_data[frame_num] if isinstance(balls_data,list) and frame_num < len(balls_data) and isinstance(balls_data[frame_num],dict) else {}
        referee_dict = referees_data[frame_num] if isinstance(referees_data,list) and frame_num < len(referees_data) and isinstance(referees_data[frame_num],dict) else {}
        for track_id, p_info in player_dict.items():
            if isinstance(p_info, dict) and 'bbox' in p_info:
                color = p_info.get("team_color", (0,0,255)); self.draw_ellipse(frame, p_info["bbox"], color, track_id)
                if p_info.get('has_ball', False): self.draw_traingle(frame, p_info["bbox"], (0,0,255))
        for _, r_info in referee_dict.items():
            if isinstance(r_info, dict) and 'bbox' in r_info: self.draw_ellipse(frame, r_info["bbox"], (0,255,255))
        ball_info = ball_dict.get(1) 
        if isinstance(ball_info, dict) and 'bbox' in ball_info: self.draw_traingle(frame, ball_info["bbox"], (0,255,0))
        self.draw_team_ball_control(frame, frame_num, team_ball_control_history_array)
        return frame