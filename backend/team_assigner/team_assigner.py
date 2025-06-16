# team_assigner/team_assigner.py

from sklearn.cluster import KMeans
import numpy as np
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

class TeamAssigner:
    def __init__(self):
        self.team_colors = {1: np.array([255,0,0]), 2: np.array([0,0,255])} 
        self.player_team_dict = {} 
        self.kmeans = None

    def _get_clustering_model(self,image_data_for_kmeans):
        if image_data_for_kmeans.shape[0] * image_data_for_kmeans.shape[1] < 2 : return None
        image_2d = image_data_for_kmeans.reshape(-1,3)
        if image_2d.shape[0] < 2: return None
        try:
            kmeans_model = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
            kmeans_model.fit(image_2d)
            return kmeans_model
        except ValueError: # Catch cases where K-Means cannot run (e.g. not enough unique samples)
            # print("ValueError during K-Means fitting, possibly not enough unique samples.")
            return None
        except Exception as e:
            # print(f"Error during K-Means fitting: {e}")
            return None

    def get_player_color(self,frame_image,bbox):
        if bbox is None or len(bbox) != 4: return np.array([0,0,0])
        x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame_image.shape[:2]
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(w_frame, x2), min(h_frame, y2)
        if x1_c >= x2_c or y1_c >= y2_c: return np.array([0,0,0]) 
        player_image_roi = frame_image[y1_c:y2_c, x1_c:x2_c]
        if player_image_roi.size == 0: return np.array([0,0,0])
        top_half_height = int(player_image_roi.shape[0] / 2)
        top_half_image_roi = player_image_roi[0:top_half_height,:]
        image_to_cluster = top_half_image_roi if top_half_image_roi.size > 0 else player_image_roi
        if image_to_cluster.size == 0 : return np.array([0,0,0])
        kmeans_instance = self._get_clustering_model(image_to_cluster)
        if kmeans_instance is None: return np.array([0,0,0]) 
        pixel_labels = kmeans_instance.labels_
        if image_to_cluster.shape[0] == 0 or image_to_cluster.shape[1] == 0: return np.array([0,0,0])
        clustered_img_labels = pixel_labels.reshape(image_to_cluster.shape[0], image_to_cluster.shape[1])
        corner_cluster_labels = []
        if clustered_img_labels.shape[0]>0 and clustered_img_labels.shape[1]>0: corner_cluster_labels.append(clustered_img_labels[0,0])
        if clustered_img_labels.shape[0]>0 and clustered_img_labels.shape[1]>1: corner_cluster_labels.append(clustered_img_labels[0,-1])
        if clustered_img_labels.shape[0]>1 and clustered_img_labels.shape[1]>0: corner_cluster_labels.append(clustered_img_labels[-1,0])
        if clustered_img_labels.shape[0]>1 and clustered_img_labels.shape[1]>1: corner_cluster_labels.append(clustered_img_labels[-1,-1])
        non_player_cluster_label = 0
        if not corner_cluster_labels:
            if pixel_labels.size > 0:
                unique_lbls, counts = np.unique(pixel_labels, return_counts=True)
                if len(counts)>0: non_player_cluster_label = unique_lbls[np.argmax(counts)]
        else: non_player_cluster_label = max(set(corner_cluster_labels),key=corner_cluster_labels.count)
        player_cluster_label = 1 - non_player_cluster_label
        if not (0 <= player_cluster_label < len(kmeans_instance.cluster_centers_)): return np.array([0,0,0])
        return kmeans_instance.cluster_centers_[player_cluster_label]

    def assign_team_color(self,first_frame_image, player_detections_first_frame):
        player_colors_list = []
        if not isinstance(player_detections_first_frame, dict): player_detections_first_frame = {}
        for _, player_detection_info in player_detections_first_frame.items():
            if isinstance(player_detection_info, dict) and 'bbox' in player_detection_info:
                player_colors_list.append(self.get_player_color(first_frame_image,player_detection_info["bbox"]))
        
        valid_player_colors = [color for color in player_colors_list if not np.array_equal(color, np.array([0,0,0]))]

        if not valid_player_colors or len(np.unique(np.array(valid_player_colors), axis=0)) < 2:
            print("Warning: Not enough distinct player colors for K-Means. Using default team colors.")
            self.kmeans = None; return
        try:
            self.kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10, random_state=0)
            self.kmeans.fit(valid_player_colors)
            self.team_colors[1] = self.kmeans.cluster_centers_[0]
            self.team_colors[2] = self.kmeans.cluster_centers_[1]
        except Exception as e:
            print(f"Error during K-Means for team colors: {e}. Using default team colors.")
            self.kmeans = None

    def get_player_team(self,current_frame_image,player_bbox,player_id):
        if player_id in self.player_team_dict: return self.player_team_dict[player_id]
        player_color = self.get_player_color(current_frame_image,player_bbox)
        assigned_team_id = 1 
        if self.kmeans is None:
            dist1 = np.linalg.norm(player_color - self.team_colors.get(1, np.array([255,0,0])))
            dist2 = np.linalg.norm(player_color - self.team_colors.get(2, np.array([0,0,255])))
            assigned_team_id = 1 if dist1 <= dist2 else 2 # Assign to closest
        else:
            try:
                assigned_team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] + 1
            except Exception: 
                dist1 = np.linalg.norm(player_color - self.team_colors.get(1, np.array([255,0,0])))
                dist2 = np.linalg.norm(player_color - self.team_colors.get(2, np.array([0,0,255])))
                assigned_team_id = 1 if dist1 <= dist2 else 2
        self.player_team_dict[player_id] = assigned_team_id
        return assigned_team_id