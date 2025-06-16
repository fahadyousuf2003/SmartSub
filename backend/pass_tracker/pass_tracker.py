import sys
# sys.path.append('../') # Adjust if necessary
import cv2
import numpy as np

class PassTracker:
    def __init__(self):
        """
        Initialize the PassTracker.
        Tracks ball possession changes. A "pass" is credited to the receiver.
        """
        self.player_passes = {}  # Stores cumulative passes (receptions) by each player_id
        self.last_player_with_ball = None  # Track which player_id last had the ball

    def calculate_passes(self, tracks):
        """
        Calculate passes based on ball touches by different players.
        The 'passes' field in tracks will store the number of receptions.
        """
        if 'players' not in tracks or not isinstance(tracks['players'], list) or not tracks['players']:
            print("Warning (PassTracker): 'players' data not found, not a list, or empty for pass calculation.")
            return

        all_player_ids_in_video = set()
        for frame_player_data in tracks['players']:
            if isinstance(frame_player_data, dict):
                for p_id in frame_player_data.keys():
                    all_player_ids_in_video.add(p_id)
        
        for p_id in all_player_ids_in_video:
            self.player_passes[p_id] = 0

        for frame_num in range(len(tracks['players'])):
            if isinstance(tracks['players'][frame_num], dict):
                for player_id in tracks['players'][frame_num].keys():
                    if isinstance(tracks['players'][frame_num][player_id], dict):
                        tracks['players'][frame_num][player_id]['passes'] = 0
                        if 'passing' in tracks['players'][frame_num][player_id]:
                            del tracks['players'][frame_num][player_id]['passing']
                        if 'receiving' in tracks['players'][frame_num][player_id]:
                            del tracks['players'][frame_num][player_id]['receiving']
        
        self.last_player_with_ball = None 

        for frame_num in range(len(tracks['players'])):
            if not isinstance(tracks['players'][frame_num], dict):
                continue

            current_player_with_ball_this_frame = None
            for p_id, p_info in tracks['players'][frame_num].items():
                if isinstance(p_info, dict) and p_info.get('has_ball', False):
                    current_player_with_ball_this_frame = p_id
                    break
            
            if current_player_with_ball_this_frame is not None:
                if self.last_player_with_ball is not None and \
                   current_player_with_ball_this_frame != self.last_player_with_ball:
                    
                    if current_player_with_ball_this_frame in self.player_passes:
                        self.player_passes[current_player_with_ball_this_frame] += 1
                    
                    if current_player_with_ball_this_frame in tracks['players'][frame_num] and \
                       isinstance(tracks['players'][frame_num][current_player_with_ball_this_frame], dict):
                        tracks['players'][frame_num][current_player_with_ball_this_frame]['receiving'] = True
                    
                    if self.last_player_with_ball in tracks['players'][frame_num] and \
                       isinstance(tracks['players'][frame_num][self.last_player_with_ball], dict):
                        tracks['players'][frame_num][self.last_player_with_ball]['passing'] = True
                
                self.last_player_with_ball = current_player_with_ball_this_frame

            if isinstance(tracks['players'][frame_num], dict):
                for p_id_update in tracks['players'][frame_num].keys():
                    if p_id_update in self.player_passes and \
                       isinstance(tracks['players'][frame_num][p_id_update], dict):
                        tracks['players'][frame_num][p_id_update]['passes'] = self.player_passes[p_id_update]

    def draw_passes_on_single_frame(self, frame, frame_num, tracks_data):
        """
        Add pass count display to a single video frame.
        """
        if 'players' not in tracks_data or not isinstance(tracks_data['players'], list) or \
           frame_num >= len(tracks_data['players']):
            return frame 
        
        current_frame_player_tracks = tracks_data['players'][frame_num]
        if not isinstance(current_frame_player_tracks, dict):
            return frame 

        for track_id, p_info in current_frame_player_tracks.items():
            if not isinstance(p_info, dict) or 'bbox' not in p_info:
                continue

            bbox = p_info['bbox']
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            
            pass_count = p_info.get('passes', 0) # This is the count of receptions

            try:
                foot_pos_x = int((bbox[0] + bbox[2]) / 2)
                text_y_offset = int(bbox[3] + 60)

                # --- TEXT CHANGE IS HERE ---
                text_to_draw = f"{pass_count} passes" 
                # --- END OF TEXT CHANGE ---

                font_scale = 0.4
                thickness = 1
                text_color_default = (50, 50, 50) 

                if p_info.get('passing', False):
                    # --- TEXT CHANGE IS HERE ---
                    text_to_draw = f"PASSING ({pass_count} passes)"
                    # --- END OF TEXT CHANGE ---
                    text_color_default = (0, 100, 0) 
                elif p_info.get('receiving', False):
                    # --- TEXT CHANGE IS HERE ---
                    text_to_draw = f"RECEIVED ({pass_count} passes)"
                    # --- END OF TEXT CHANGE ---
                    text_color_default = (100, 0, 0) 
                
                (text_w, text_h), _ = cv2.getTextSize(text_to_draw, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                rect_x1 = foot_pos_x - text_w // 2 - 3
                rect_y1 = text_y_offset - text_h - 3 
                rect_x2 = foot_pos_x + text_w // 2 + 3
                rect_y2 = text_y_offset + 3 
                
                h_frame, w_frame = frame.shape[:2]
                rect_x1_c = max(0, rect_x1); rect_y1_c = max(0, rect_y1)
                rect_x2_c = min(w_frame -1 , rect_x2); rect_y2_c = min(h_frame -1, rect_y2)

                if rect_x1_c < rect_x2_c and rect_y1_c < rect_y2_c: 
                    sub_img_roi = frame[rect_y1_c:rect_y2_c, rect_x1_c:rect_x2_c]
                    bg_color_rect = np.full(sub_img_roi.shape, (220, 220, 220), dtype=np.uint8) 
                    alpha_blend = 0.6
                    frame[rect_y1_c:rect_y2_c, rect_x1_c:rect_x2_c] = cv2.addWeighted(sub_img_roi, 1 - alpha_blend, bg_color_rect, alpha_blend, 0)

                text_pos_x_final = foot_pos_x - text_w // 2
                text_pos_y_final = text_y_offset 
                cv2.putText(frame, text_to_draw, (text_pos_x_final, text_pos_y_final),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color_default, thickness, cv2.LINE_AA)
            except Exception as e:
                print(f"Error drawing pass info for track_id {track_id} on frame {frame_num}: {e}")
        
        return frame
