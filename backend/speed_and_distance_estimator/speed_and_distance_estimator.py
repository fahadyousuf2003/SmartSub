# speed_and_distance_estimator/speed_and_distance_estimator.py

import cv2
import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
    
    def add_speed_and_distance_to_tracks(self, tracks_data):
        if not isinstance(tracks_data, dict): return
        total_cumulative_distance = {} 
        for object_name, object_tracks_list in tracks_data.items():
            if object_name == "ball" or object_name == "referees" or not isinstance(object_tracks_list, list): continue
            number_of_frames = len(object_tracks_list)
            if number_of_frames == 0: continue
            if object_name not in total_cumulative_distance: total_cumulative_distance[object_name] = {}

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame_in_window = min(frame_num + self.frame_window, number_of_frames - 1)
                if last_frame_in_window <= frame_num: continue
                if not (frame_num < number_of_frames and isinstance(object_tracks_list[frame_num], dict)): continue 

                for track_id,_ in object_tracks_list[frame_num].items():
                    current_track_info_start = object_tracks_list[frame_num].get(track_id)
                    current_track_info_end = object_tracks_list[last_frame_in_window].get(track_id) if last_frame_in_window < number_of_frames and isinstance(object_tracks_list[last_frame_in_window], dict) else None

                    if not (isinstance(current_track_info_start, dict) and \
                            isinstance(current_track_info_end, dict) and \
                            'position_transformed' in current_track_info_start and \
                            current_track_info_start['position_transformed'] is not None and \
                            'position_transformed' in current_track_info_end and \
                            current_track_info_end['position_transformed'] is not None):
                        continue
                    
                    start_position = current_track_info_start['position_transformed']
                    end_position = current_track_info_end['position_transformed']
                    
                    distance_covered_in_window = measure_distance(start_position, end_position)
                    time_elapsed_in_window = (last_frame_in_window - frame_num) / self.frame_rate
                    
                    speed_meters_per_second = 0.0
                    if time_elapsed_in_window > 0:
                        speed_meters_per_second = distance_covered_in_window / time_elapsed_in_window
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    if track_id not in total_cumulative_distance[object_name]:
                        total_cumulative_distance[object_name][track_id] = 0.0
                    
                    # This cumulative distance is added for the segment, then the new total is applied
                    total_cumulative_distance[object_name][track_id] += distance_covered_in_window
                    
                    for frame_num_in_batch in range(frame_num, last_frame_in_window): # Apply to [start, end)
                        if frame_num_in_batch < number_of_frames and \
                           isinstance(object_tracks_list[frame_num_in_batch], dict) and \
                           track_id in object_tracks_list[frame_num_in_batch] and \
                           isinstance(object_tracks_list[frame_num_in_batch][track_id], dict):
                            tracks_data[object_name][frame_num_in_batch][track_id]['speed'] = speed_km_per_hour
                            tracks_data[object_name][frame_num_in_batch][track_id]['distance'] = total_cumulative_distance[object_name][track_id]
    
    def draw_speed_and_distance_on_single_frame(self, frame, frame_num, all_tracks_data):
        if not isinstance(all_tracks_data, dict): return frame
        for object_type, object_tracks_list in all_tracks_data.items():
            if object_type == "ball" or object_type == "referees": continue
            if not isinstance(object_tracks_list, list) or frame_num >= len(object_tracks_list): continue
            
            tracks_for_current_frame_object = object_tracks_list[frame_num]
            if not isinstance(tracks_for_current_frame_object, dict): continue

            for track_id, track_info in tracks_for_current_frame_object.items():
               if isinstance(track_info, dict) and "speed" in track_info and \
                  'bbox' in track_info and track_info['bbox'] is not None:
                   speed, distance = track_info.get('speed'), track_info.get('distance')
                   if speed is None or distance is None: continue
                   bbox = track_info['bbox']
                   if len(bbox) != 4: continue
                   try:
                       position = list(get_foot_position(bbox)); position[1]+=40 
                       text_pos_speed = tuple(map(int,position))
                       text_pos_distance = (text_pos_speed[0], text_pos_speed[1] + 15)
                       font_scale, thickness, text_color = 0.4, 1, (0,0,0)
                       cv2.putText(frame, f"{speed:.1f}km/h", text_pos_speed, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                       cv2.putText(frame, f"{distance:.1f}m", text_pos_distance, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                   except Exception as e:
                       print(f"Error drawing speed/distance for track {track_id}, bbox {bbox}: {e}")
        return frame