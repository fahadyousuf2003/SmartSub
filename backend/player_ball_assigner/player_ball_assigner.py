# player_ball_assigner/player_ball_assigner.py

import sys 
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players_in_frame,ball_bbox):
        if not ball_bbox or not isinstance(players_in_frame, dict) or not players_in_frame: 
            return -1
        try:    
            ball_pos = get_center_of_bbox(ball_bbox)
        except Exception: # If ball_bbox is malformed
            return -1

        min_dist_to_ball, assigned_player_id = float('inf'), -1
        for p_id, p_info in players_in_frame.items():
            if not isinstance(p_info, dict) or 'bbox' not in p_info or p_info['bbox'] is None: 
                continue 
            p_bbox = p_info['bbox']
            if len(p_bbox) != 4: continue 

            try:
                dist_l_foot = measure_distance((p_bbox[0],p_bbox[3]),ball_pos)
                dist_r_foot = measure_distance((p_bbox[2],p_bbox[3]),ball_pos)
                current_player_dist = min(dist_l_foot,dist_r_foot)
            except Exception: # If p_bbox is malformed for distance calculation
                continue

            if current_player_dist < self.max_player_ball_distance and current_player_dist < min_dist_to_ball:
                min_dist_to_ball = current_player_dist
                assigned_player_id = p_id
        return assigned_player_id