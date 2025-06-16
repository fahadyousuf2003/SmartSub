import pymongo
import cv2
import numpy as np # For potential NaN handling if needed

class RecommenderSystem:
    def __init__(self, db_uri, db_name="FYP_DB", collection_name="players_data"):
        self.client = None
        self.db = None
        self.collection = None
        self.passing_threshold = 10  # Min passes for a meaningful ratio, adjust as needed
        self.default_evaluation_window = 600 # Default frames to consider for player stats

        try:
            self.client = pymongo.MongoClient(db_uri)
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster') # Check server status
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print(f"Successfully connected to MongoDB. DB: {db_name}, Collection: {collection_name}")
        except pymongo.errors.ConnectionFailure as e:
            print(f"MongoDB Connection Failure: {e}. Recommender system will not query database.")
            self.client = None # Ensure these are None if connection fails
            self.db = None
            self.collection = None
        except Exception as e_auth: # Catch other potential errors like auth or configuration
            print(f"Error connecting to MongoDB (check URI, credentials, network access): {e_auth}")
            self.client = None
            self.db = None
            self.collection = None


    def evaluate_players(self, tracks, evaluation_window_frames=None):
        player_metrics = []
        if evaluation_window_frames is None:
            evaluation_window_frames = self.default_evaluation_window

        if not tracks or 'players' not in tracks or not isinstance(tracks['players'], list) or not tracks['players']:
            print("Warning (RecommenderSystem): 'players' data not found, not a list, or empty for evaluation.")
            return {}, {}, []

        all_present_track_ids = set()
        for frame_data in tracks['players']:
            if isinstance(frame_data, dict):
                all_present_track_ids.update(frame_data.keys())
        
        if not all_present_track_ids:
            print("Warning (RecommenderSystem): No player track IDs found in any frame.")
            return {}, {}, []

        for track_id in all_present_track_ids:
            speeds = []
            last_known_distance = 0
            last_known_passes = 0
            current_team_id = None
            # Use the team_name directly from tracks if populated by main.py
            current_team_name = tracks['players'][0].get(track_id, {}).get('team_name', f"Unknown Team {track_id}")


            frames_processed_for_player = 0
            for frame_num in range(len(tracks['players'])):
                if frames_processed_for_player >= evaluation_window_frames:
                    break 

                if frame_num < len(tracks['players']) and \
                   isinstance(tracks['players'][frame_num], dict) and \
                   track_id in tracks['players'][frame_num] and \
                   isinstance(tracks['players'][frame_num][track_id], dict):
                    
                    info = tracks['players'][frame_num][track_id]
                    
                    if info.get("speed") is not None:
                        speeds.append(info.get("speed"))
                    
                    if info.get("distance") is not None:
                        last_known_distance = info.get("distance")
                    
                    if info.get("passes") is not None:
                        last_known_passes = info.get("passes")
                    
                    # Update team_name if it changes or gets defined later for the player
                    # (main.py should consistently populate this)
                    if info.get("team_name") is not None:
                        current_team_name = info.get("team_name")
                    if info.get("team") is not None: # also keep track of id if needed
                        current_team_id = info.get("team")

                    frames_processed_for_player += 1

            if not speeds and last_known_distance == 0 and last_known_passes == 0:
                continue

            avg_speed = sum(speeds) / len(speeds) if speeds else 0
            passing_ratio = (last_known_passes / self.passing_threshold) if self.passing_threshold > 0 and last_known_passes > 0 else 0.0

            player_metrics.append({
                "track_id": track_id,
                "avg_speed": avg_speed,
                "total_distance": last_known_distance,
                "total_passes": last_known_passes,
                "passing_ratio": passing_ratio,
                "team_id": current_team_id,
                "team_name": current_team_name
            })

        if not player_metrics:
            print("Warning (RecommenderSystem): No player metrics could be calculated.")
            return {}, {}, []
            
        num_players_to_show = min(3, len(player_metrics))

        worst_performers = {
            "speed": sorted(player_metrics, key=lambda x: x["avg_speed"])[:num_players_to_show],
            "distance": sorted(player_metrics, key=lambda x: x["total_distance"])[:num_players_to_show],
            "passing": sorted(player_metrics, key=lambda x: x["passing_ratio"])[:num_players_to_show]
        }
        best_performers = {
            "speed": sorted(player_metrics, key=lambda x: x["avg_speed"], reverse=True)[:num_players_to_show],
            "distance": sorted(player_metrics, key=lambda x: x["total_distance"], reverse=True)[:num_players_to_show],
            "passing": sorted(player_metrics, key=lambda x: x["passing_ratio"], reverse=True)[:num_players_to_show]
        }
        worst_performers["speed_distance"] = sorted(player_metrics, key=lambda x: (x["avg_speed"], x["total_distance"]))[:num_players_to_show]
        worst_performers["distance_passing"] = sorted(player_metrics, key=lambda x: (x["total_distance"], x["passing_ratio"]))[:num_players_to_show]
        worst_performers["speed_passing"] = sorted(player_metrics, key=lambda x: (x["avg_speed"], x["passing_ratio"]))[:num_players_to_show]
        worst_performers["all"] = sorted(player_metrics, key=lambda x: (x["avg_speed"], x["total_distance"], x["passing_ratio"]))[:num_players_to_show]

        return worst_performers, best_performers, player_metrics

    def find_better_player_by_metrics(self, metrics_to_improve, current_player_stat_values, target_team_name):
        # ******** CORE FIX FOR NotImplementedError ********
        if self.collection is None: # Changed from 'if not self.collection:'
            print("MongoDB collection not available. Cannot find better player.")
            return None

        metric_to_db_field_map = {
            "speed": "avg_speed_kmph",
            "distance": "avg_distance_km",
            "passing": "passing_ratio"
        }
        query_conditions = {}
        
        if target_team_name:
            query_conditions["team"] = target_team_name
        else:
            print("Warning: No target team name specified for finding replacement. Querying all teams if any metrics defined.")
            # If no team is specified, the query might be too broad or not meaningful.
            # Depending on desired behavior, might return None or query without team filter.

        has_at_least_one_metric_condition = False
        for metric_key in metrics_to_improve:
            db_field_name = metric_to_db_field_map.get(metric_key)
            if db_field_name and metric_key in current_player_stat_values:
                value_to_exceed = current_player_stat_values[metric_key]
                if isinstance(value_to_exceed, (int, float)):
                    if metric_key == "distance": 
                        value_to_exceed = value_to_exceed / 1000 
                    query_conditions[db_field_name] = {"$gt": float(value_to_exceed)}
                    has_at_least_one_metric_condition = True
                else:
                    print(f"Warning: Value for metric '{metric_key}' is not a number: {value_to_exceed}")
        
        if not has_at_least_one_metric_condition and target_team_name:
            print(f"Warning: No valid metric improvement criteria. Finding best available player in team '{target_team_name}' based on sort order if any.")
        elif not has_at_least_one_metric_condition and not target_team_name:
             print(f"Warning: No valid metric improvement criteria and no team specified. Query is too broad. Aborting find.")
             return None


        projection_fields = {
            "_id": 0, "player_name": 1, "avg_speed_kmph": 1,
            "avg_distance_km": 1, "passing_ratio": 1, "player_id": 1, "team": 1
        }
        
        sort_order_criteria = []
        for metric_key in metrics_to_improve:
            db_field_name = metric_to_db_field_map.get(metric_key)
            if db_field_name:
                sort_order_criteria.append((db_field_name, pymongo.DESCENDING))
        
        if not sort_order_criteria: 
            # Default sort if no improvement metrics could be translated (e.g. player_name or a general performance score)
            # For now, let's keep it potentially empty if no criteria.
            # Or add a sensible default like: sort_order_criteria.append(("avg_speed_kmph", pymongo.DESCENDING))
             print("No specific sort order based on improvement metrics, results may not be optimal.")


        try:
            # Only execute query if there's something meaningful to query for
            if not query_conditions: # If query is empty (e.g. no team, no metrics)
                print("Query conditions are empty. Aborting database search for recommendation.")
                return None

            print(f"Database query for recommendation: {query_conditions}, Sort: {sort_order_criteria}")
            result_cursor = self.collection.find(query_conditions, projection_fields)
            if sort_order_criteria: # Apply sort only if criteria exist
                result_cursor = result_cursor.sort(sort_order_criteria)
            
            recommended_player = next(result_cursor.limit(1), None)

            if recommended_player:
                print(f"Found recommendation from DB: {recommended_player.get('player_name', 'N/A')}")
            else:
                print(f"No player found in DB meeting the criteria: {query_conditions}")
            return recommended_player
        except pymongo.errors.PyMongoError as e:
            print(f"MongoDB query error in find_better_player_by_metrics: {e}")
            return None

    def get_substitution_recommendation(self, worst_performers_map, metric_choice_key, team_choice_name=None):
        metric_key_map = {
            "1": ("speed", ["speed"]), "2": ("distance", ["distance"]), "3": ("passing", ["passing"]),
            "4": ("speed_distance", ["speed", "distance"]), "5": ("distance_passing", ["distance", "passing"]),
            "6": ("speed_passing", ["speed", "passing"]), "7": ("all", ["speed", "distance", "passing"])
        }
        if metric_choice_key not in metric_key_map:
            print(f"Invalid metric choice key: {metric_choice_key}"); return None, None
        
        category_key, metrics_to_focus_on = metric_key_map[metric_choice_key]

        if category_key not in worst_performers_map or not worst_performers_map[category_key]:
            print(f"No worst performers data for category: '{category_key}'"); return None, None

        candidate_underperformers = worst_performers_map[category_key]
        underperforming_player_to_sub = None

        if team_choice_name:
            for player in candidate_underperformers:
                if player.get("team_name") == team_choice_name:
                    underperforming_player_to_sub = player; break
            if not underperforming_player_to_sub:
                print(f"No underperforming player from team '{team_choice_name}' in category '{category_key}'."); return None, None
        elif candidate_underperformers:
            underperforming_player_to_sub = candidate_underperformers[0]
        
        if not underperforming_player_to_sub:
            print(f"Could not identify underperformer for category '{category_key}'."); return None, None

        print(f"Identified underperformer: Player TrackID #{underperforming_player_to_sub['track_id']} (Team: {underperforming_player_to_sub.get('team_name', 'N/A')}) for category '{category_key}'")
        current_player_stats = {
            "speed": underperforming_player_to_sub.get("avg_speed", 0),
            "distance": underperforming_player_to_sub.get("total_distance", 0),
            "passing": underperforming_player_to_sub.get("passing_ratio", 0)
        }
        team_of_underperformer = underperforming_player_to_sub.get("team_name")
        
        # If user selected "All Teams", team_choice_name is None.
        # We should use the underperformer's actual team to find a replacement from that same team.
        # If team_choice_name was specified, team_of_underperformer should ideally be the same.
        effective_target_team = team_of_underperformer # Always try to replace with player from same team

        if not effective_target_team:
            print(f"Warning: Underperforming player has no team name. Replacement search might be less targeted or broader.")
            # If underperformer has no team, and user picked "All teams", then target_team_name for query becomes None (broad search)
            # If underperformer has no team, and user picked a specific team, this state shouldn't happen if filtering above is correct.

        recommended_substitute = self.find_better_player_by_metrics(
            metrics_to_focus_on, current_player_stats, effective_target_team
        )
        return underperforming_player_to_sub, recommended_substitute

    def display_player_analysis(self, worst_performers, best_performers):
        print("\n=== PLAYER ANALYSIS ===")
        metrics_to_display = ["speed", "distance", "passing"]
        for kind, data_map in [("WORST", worst_performers), ("BEST", best_performers)]:
            print(f"\n--- {kind} PERFORMERS (Individual Metrics) ---")
            for metric in metrics_to_display:
                if metric in data_map and data_map[metric]:
                    players = data_map[metric]
                    print(f"\n{metric.upper()} ({kind} {len(players)}):")
                    for i, player in enumerate(players, 1):
                        team_info = f" ({player.get('team_name', 'Unknown')})"
                        pass_val = player.get('total_passes',0)
                        pass_ratio_val = player.get('passing_ratio', 0.0)
                        if metric == "speed": print(f"{i}. Player TrackID #{player['track_id']}{team_info}: {player['avg_speed']:.2f} km/h")
                        elif metric == "distance": print(f"{i}. Player TrackID #{player['track_id']}{team_info}: {player['total_distance']:.2f} m")
                        elif metric == "passing": print(f"{i}. Player TrackID #{player['track_id']}{team_info}: {pass_val} passes (ratio: {pass_ratio_val:.2f})")
                else:
                    print(f"\nNo data for {kind.lower()} performers in {metric}.")

    def add_recommendations_to_video(self, frames, underperformer_data, recommendation_data, display_start_frame=0):
        if not recommendation_data or not underperformer_data: return frames
        output_frames = [frame.copy() for frame in frames]
        msg_lines = ["SUBSTITUTION RECOMMENDATION:"]
        u_team = f" ({underperformer_data.get('team_name', 'N/A')})"
        u_stats = f"Speed:{underperformer_data.get('avg_speed',0):.1f}km/h, Dist:{underperformer_data.get('total_distance',0):.1f}m, Pass:{underperformer_data.get('total_passes',0)}({underperformer_data.get('passing_ratio',0):.1f})"
        msg_lines.append(f"Sub Player TrkID#{underperformer_data.get('track_id','N/A')}{u_team}")
        msg_lines.append(f"  Stats: {u_stats}")
        r_team = f" ({recommendation_data.get('team', 'N/A')})"
        r_stats_parts = []
        if recommendation_data.get('avg_speed_kmph') is not None: r_stats_parts.append(f"Speed:{recommendation_data.get('avg_speed_kmph'):.1f}km/h")
        if recommendation_data.get('avg_distance_km') is not None: r_stats_parts.append(f"Dist:{recommendation_data.get('avg_distance_km'):.1f}km")
        if recommendation_data.get('passing_ratio') is not None: r_stats_parts.append(f"PassRatio:{recommendation_data.get('passing_ratio'):.1f}")
        r_stats_str = ", ".join(r_stats_parts)
        msg_lines.append(f"With Player: {recommendation_data.get('player_name', 'N/A')}{r_team} (DB ID:{recommendation_data.get('player_id','N/A')})")
        if r_stats_str: msg_lines.append(f"  DB Stats: {r_stats_str}")

        font, font_scale, font_thickness, text_color, bg_color = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1, (0,0,0), (230,230,230)
        line_height, padding = 20, 5
        for i in range(display_start_frame, len(output_frames)):
            frame = output_frames[i]; frame_h, frame_w = frame.shape[:2]
            max_text_w = max((cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in msg_lines), default=0)
            box_h, box_w = len(msg_lines) * line_height + 2*padding, max_text_w + 2*padding
            box_x1, box_y1 = 10, frame_h - box_h - 10
            box_x2, box_y2 = min(frame_w-10, box_x1 + box_w), min(frame_h-10, box_y1 + box_h)
            if box_x1 < box_x2 and box_y1 < box_y2:
                sub_img = frame[box_y1:box_y2, box_x1:box_x2]
                rect_fill = np.full(sub_img.shape, bg_color, dtype=np.uint8)
                cv2.addWeighted(rect_fill, 0.7, sub_img, 0.3, 0, sub_img)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0,0,0), 1)
            for j, text_line in enumerate(msg_lines):
                cv2.putText(frame, text_line, (box_x1+padding, box_y1+padding+(j*line_height)+(line_height-5)),
                            font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return output_frames

    def get_user_substitution_choice(self):
        print("\n=== SUBSTITUTION CRITERIA ===")
        print("1. Speed only\n2. Distance only\n3. Passing only\n4. Speed + Distance\n5. Distance + Passing\n6. Speed + Passing\n7. All parameters")
        while True:
            choice = input("Enter your choice (1-7): ").strip()
            if choice in map(str, range(1,8)): return choice
            print("Invalid choice. Please enter a number between 1 and 7.")
    
    def get_team_choice(self, available_team_names):
        print("\n=== TEAM SELECTION FOR RECOMMENDATION ===")
        if not available_team_names:
            print("No specific teams identified. Recommendation may be broader."); return None 
        print("Choose team for substitution recommendation:")
        for i, team_name_item in enumerate(available_team_names, 1): print(f"{i}. {team_name_item}")
        print(f"{len(available_team_names) + 1}. All Teams / Overall (uses underperformer's team)")
        while True:
            choice_str = input(f"Enter choice (1-{len(available_team_names) + 1}): ").strip()
            try:
                choice_int = int(choice_str)
                if 1 <= choice_int <= len(available_team_names): return available_team_names[choice_int - 1]
                if choice_int == len(available_team_names) + 1: return None
                print(f"Invalid choice. Select 1 to {len(available_team_names) + 1}.")
            except ValueError: print("Invalid input. Enter a number.")