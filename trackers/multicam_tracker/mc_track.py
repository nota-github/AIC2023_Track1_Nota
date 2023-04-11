from . import matching
from itertools import combinations, chain
import numpy as np


class ID_Distributor:
    def __init__(self, init_id=0):
        self.cur_id = init_id

    def assign_id(self):
        self.cur_id += 1
        return self.cur_id

class MCTracker:
    def __init__(self, appearance_thresh=0.04, match_thresh=0.8, map_size=None):
        self.appearance_thresh = appearance_thresh
        self.match_thresh = match_thresh
        if map_size:
            # self.euc_thresh = (map_size[0] + map_size[1]) / 2.0 / 10.0
            self.euc_thresh = np.sqrt(map_size[0]**2 + map_size[1]**2) / 10.0
            print('euc_thresh: ', self.euc_thresh)
    
    def update(self, trackers):
        matched_ids = []
        tracker_pairs = list(combinations(trackers, 2))

        for a_tracker, b_tracker in tracker_pairs:
            a_features = [t.smooth_feat for t in a_tracker.tracked_stracks]
            b_features = [t.smooth_feat for t in b_tracker.tracked_stracks]

            a_locations = [t.location[0] for t in a_tracker.tracked_stracks]
            b_locations = [t.location[0] for t in b_tracker.tracked_stracks]

            euc_dists = matching.euclidean_distance(a_locations, b_locations)
            emb_dists = matching.embedding_distance(a_features, b_features) / 2.0

            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[euc_dists > self.euc_thresh] = 1.0

            matches, u_afeats, u_bfeats = matching.linear_assignment(emb_dists, thresh=self.match_thresh)

            for id_atrack, id_btrack in matches:
                a_global_id = a_tracker.tracked_stracks[id_atrack].global_id
                b_global_id = b_tracker.tracked_stracks[id_btrack].global_id
                matched_id = a_global_id if a_global_id <= b_global_id else b_global_id
                total_global_ids = [track.global_id for track in a_tracker.tracked_stracks] + [track.global_id for track in b_tracker.tracked_stracks]
                total_global_ids_lost = [track.global_id for track in a_tracker.lost_stracks] + [track.global_id for track in b_tracker.lost_stracks]
                if total_global_ids.count(matched_id) > 1 or total_global_ids_lost.count(matched_id):
                    continue
                a_tracker.tracked_stracks[id_atrack].global_id = b_tracker.tracked_stracks[id_btrack].global_id = matched_id
    
    def postprocess(self, result_lists):
        """ToDo (Offline Method)"""

        return result_lists
