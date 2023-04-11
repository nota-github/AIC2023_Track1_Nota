from . import matching
from itertools import combinations
import numpy as np
import pdb


class ID_Distributor:
    def __init__(self, init_id=0):
        self.cur_id = init_id

    def assign_id(self):
        self.cur_id += 1
        return self.cur_id

class Clustering:
    def __init__(self, appearance_thresh=0.25, euc_thresh=0.2, match_thresh=0.8, map_size=None):
        self.appearance_thresh = appearance_thresh
        self.match_thresh = match_thresh
        if map_size:
            self.euc_thresh = euc_thresh
            self.max_len = np.sqrt(map_size[0]**2 + map_size[1]**2)
            print('euc_thresh: ', self.euc_thresh)

    def update(self, trackers, cur_frame, scene):
        matched_ids = []
        tracker_pairs = list(combinations(trackers, 2))
        for a_tracker, b_tracker in tracker_pairs:
            a_features, b_features, a_locations, b_locations, a_pose, b_pose = [], [], [], [], [], []
            a_match_to_id = {}
            num = 0
            for i, track in enumerate(a_tracker.tracked_stracks):
                a_features.append(track.curr_feat)  # smooth_feat로 교체 고려
                a_locations.append(track.location[0])
                if track.pose is None:
                    t, l, w, h = track.tlwh.tolist()
                    a_pose.append([False] * 5)
                else:
                    a_pose.append(self.pose_check(track.pose))
                a_match_to_id[num] = i
                num += 1
            b_match_to_id = {}
            num = 0
            for i, track in enumerate(b_tracker.tracked_stracks):
                b_features.append(track.curr_feat)  # smooth_feat로 교체 고려
                b_locations.append(track.location[0])
                if track.pose is None:
                    # t, l, w, h = track.tlwh.tolist()
                    b_pose.append([False] * 5)
                else:
                    b_pose.append(self.pose_check(track.pose))
                b_match_to_id[num] = i
                num += 1

            euc_dists = matching.euclidean_distance(a_locations, b_locations) / self.max_len
            emb_dists = matching.embedding_distance(a_features, b_features) / 2

            if 0 not in emb_dists.shape:
                norm_emb_dists = (emb_dists - np.min(emb_dists)) / (np.max(emb_dists) - np.min(emb_dists))
                norm_euc_dists = (euc_dists - np.min(euc_dists)) / (np.max(euc_dists) - np.min(euc_dists))

                dists = np.zeros_like(euc_dists)
                for i in range(len(dists)):
                    for j in range(len(dists[0])):
                        ratio = sum((np.array(a_pose[i]) * np.array(b_pose[j]))) / 10

                        dists[i][j] += (1 - ratio) * norm_euc_dists[i][j] + ratio * norm_emb_dists[i][j]
                        # if ratio >= 0.3:
                        #     if emb_dists[i][j] > self.appearance_thresh:
                        #         dists[i][j] = 1

                        # if ratio < 0.2:
                        #     dists[i][j] = norm_euc_dists[i][j]
                        # else:
                        #     dists[i][j] += 0.5 * norm_euc_dists[i][j] + 0.5 * norm_emb_dists[i][j]
                        
                        # dists[i][j] += 0.5 * norm_euc_dists[i][j] + 0.5 * norm_emb_dists[i][j]
            
            else:
                dists = 0.5 * euc_dists + 0.5 * emb_dists
            # dists[emb_dists > 0.8] = 1.0
            # dists[emb_dists > 0.4] = 1.0
            # dists[emb_dists > 0.3] = 1.0
            # dists[emb_dists > 0.25] = 1.0
            # dists[euc_dists > 0.15] = 1.0
            if scene=='S003' or scene=='S005':
                if cur_frame < 100:
                    dists[emb_dists > 0.8] = 1.0
                    dists[euc_dists > 0.175] = 1.0
                else:
                    dists[emb_dists > 0.4] = 1.0
                    # dists[euc_dists > 0.15] = 1.0
                    # dists[euc_dists > 0.25] = 1.0 
                    dists[euc_dists > 0.10] = 1.0
            elif scene=='S009':
                dists[emb_dists > 0.30] = 1.0
                dists[euc_dists > 0.15] = 1.0
            elif scene=='S014':
                dists[emb_dists > 0.275] = 1.0
                dists[euc_dists > 0.15] = 1.0
            elif scene=='S018':
                dists[emb_dists > 0.275] = 1.0
                dists[euc_dists > 0.15] = 1.0
            elif scene=='S021':
                dists[emb_dists > 0.275] = 1.0
                dists[euc_dists > 0.15] = 1.0
            elif scene=='S022':
                dists[emb_dists > 0.25] = 1.0
                dists[euc_dists > 0.15] = 1.0
            else:
                raise

            matches, u_afeats, u_bfeats = matching.linear_assignment(dists, thresh=0.999)

            for id_atrack, id_btrack in matches:
                a_global_id = a_tracker.tracked_stracks[a_match_to_id[id_atrack]].t_global_id
                b_global_id = b_tracker.tracked_stracks[b_match_to_id[id_btrack]].t_global_id
                matched_id = a_global_id if a_global_id <= b_global_id else b_global_id
                total_global_ids = [track.t_global_id for track in a_tracker.tracked_stracks] + [track.t_global_id for track in b_tracker.tracked_stracks]
                total_global_ids_lost = [track.t_global_id for track in a_tracker.lost_stracks] + [track.t_global_id for track in b_tracker.lost_stracks]
                if total_global_ids.count(matched_id) > 1 or total_global_ids_lost.count(matched_id):
                    continue

                if a_tracker.tracked_stracks[a_match_to_id[id_atrack]].matched_dist and a_global_id > b_global_id:
                    cur_dist = euc_dists[id_atrack][id_btrack] * emb_dists[id_atrack][id_btrack]
                    # cur_dist = dists[id_atrack][id_btrack]
                    old_dist = a_tracker.tracked_stracks[a_match_to_id[id_atrack]].matched_dist
                    if cur_dist > old_dist:
                        print(f'(old dist is smaller) cur: {cur_dist}, old: {old_dist}')
                        continue
                elif b_tracker.tracked_stracks[b_match_to_id[id_btrack]].matched_dist and a_global_id < b_global_id:
                    cur_dist = euc_dists[id_atrack][id_btrack] * emb_dists[id_atrack][id_btrack]
                    # cur_dist = dists[id_atrack][id_btrack]
                    old_dist = b_tracker.tracked_stracks[b_match_to_id[id_btrack]].matched_dist
                    if cur_dist > old_dist:
                        print(f'(old dist is smaller) cur: {cur_dist}, old: {old_dist}')
                        continue
                
                a_tracker.tracked_stracks[a_match_to_id[id_atrack]].t_global_id = b_tracker.tracked_stracks[b_match_to_id[id_btrack]].t_global_id = matched_id

                dist = euc_dists[id_atrack][id_btrack] * emb_dists[id_atrack][id_btrack]
                a_tracker.tracked_stracks[a_match_to_id[id_atrack]].matched_dist = dist
                b_tracker.tracked_stracks[b_match_to_id[id_btrack]].matched_dist = dist
                
        for tracker in trackers:
            for track in tracker.tracked_stracks:
                track.matched_dist = None
                
        total_objects = [[track.t_global_id, track.curr_feat, track.location[0], track.pose] for tracker in trackers for track in tracker.tracked_stracks if track.global_id != -1]
        total_ids = sorted(set([obj[0] for obj in total_objects]))
        groups = []
        for global_id in total_ids:
            num = 0
            features = []
            centroid = 0.0
            poses = []
            for obj in total_objects:
                if obj[0] == global_id:
                    num += 1
                    features.append(obj[1])
                    centroid += obj[2]
                    poses.append(obj[3])
            centroid /= num
            groups.append([global_id, features, centroid, poses])
        
        return np.array(groups, dtype=object)

    def pose_check(self, pose_):
        pose = pose_['keypoints']
        shoulder = pose[:2][:, 2].mean()
        hip = pose[[6, 7]][:, 2].mean()
        knee = pose[[8, 9]][:, 2].mean()
        head = pose[12][2]
        neck = pose[13][2]
        return np.array([head, neck, shoulder, hip, knee]) > 0.3

