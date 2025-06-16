# view_transformer/view_transformer.py

import numpy as np 
import cv2
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

class ViewTransformer():
    def __init__(self):
        court_width, court_length = 68.0, 23.32
        self.pixel_vertices = np.array([
            [110, 1035], [265, 275], [910, 260], [1640, 915]
        ], dtype=np.float32)
        self.target_vertices = np.array([
            [0, court_width], [0, 0], [court_length, 0], [court_length, court_width]
        ], dtype=np.float32)
        try:
            self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
        except cv2.error as e:
            print(f"Error initializing perspective transform: {e}. ViewTransformer may not work correctly.")
            self.perspective_transformer = None


    def transform_point(self,point_xy_tuple_or_list):
        if self.perspective_transformer is None: return None
        if point_xy_tuple_or_list is None or len(point_xy_tuple_or_list) != 2: return None
        try:
            point_for_test = (int(point_xy_tuple_or_list[0]), int(point_xy_tuple_or_list[1]))
        except (ValueError, TypeError): return None 

        if cv2.pointPolygonTest(self.pixel_vertices, point_for_test, False) < 0: return None 
        point_to_transform = np.array([[point_xy_tuple_or_list]], dtype=np.float32)
        try:
            transformed_point_array = cv2.perspectiveTransform(point_to_transform, self.perspective_transformer)
            if transformed_point_array is not None:
                return transformed_point_array[0][0].tolist()
        except cv2.error: pass # cv2.error can occur if matrix is singular or points are collinear
        except Exception: pass 
        return None

    def add_transformed_position_to_tracks(self,tracks_data):
        if self.perspective_transformer is None: 
            print("Warning: Perspective transformer not initialized in ViewTransformer. Cannot transform positions.")
            # Optionally set all 'position_transformed' to None
            for object_name, object_tracks_list in tracks_data.items():
                if not isinstance(object_tracks_list, list): continue
                for frame_num, track_dict_in_frame in enumerate(object_tracks_list):
                    if not isinstance(track_dict_in_frame, dict): continue
                    for track_id in track_dict_in_frame.keys():
                        if isinstance(tracks_data[object_name][frame_num].get(track_id), dict):
                             tracks_data[object_name][frame_num][track_id]['position_transformed'] = None
            return

        for object_name, object_tracks_list in tracks_data.items():
            if not isinstance(object_tracks_list, list): continue
            for frame_num, track_dict_in_frame in enumerate(object_tracks_list):
                if not isinstance(track_dict_in_frame, dict): continue
                for track_id, track_info in track_dict_in_frame.items():
                    transformed_pos = None 
                    if isinstance(track_info, dict): 
                        position_adjusted = track_info.get('position_adjusted') 
                        if position_adjusted is not None:
                            transformed_pos = self.transform_point(position_adjusted)
                    
                    if isinstance(tracks_data.get(object_name), list) and \
                       frame_num < len(tracks_data[object_name]) and \
                       isinstance(tracks_data[object_name][frame_num], dict) and \
                       track_id in tracks_data[object_name][frame_num] and \
                       isinstance(tracks_data[object_name][frame_num][track_id], dict):
                        tracks_data[object_name][frame_num][track_id]['position_transformed'] = transformed_pos