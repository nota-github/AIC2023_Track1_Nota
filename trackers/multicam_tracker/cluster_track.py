import numpy as np
from collections import deque
from scipy.spatial.distance import cdist

from . import matching
from .basetrack import BaseTrack, TrackState


class MTrack(BaseTrack):
    def __init__(self, global_id, centroid, feat, pose, min_hits=30, scene=None, feat_history=50):
        self.is_activated = False
        self.centroid = centroid
        self.global_id = global_id

        self.smooth_feat = None
        self.curr_feat = None
        self.curr_pose = None
        # self.features = deque([], maxlen=feat_history)
        if scene == 'S001':
            self.features = deque([], maxlen=50)
            self.pose_thresh = 14
        else:
            self.features = deque([], maxlen=25)
            self.pose_thresh = 10
        if feat is not None:
            self.update_features(feat, pose)
        self.alpha = 0.9
        self.min_hits = min_hits
    
    def update_features(self, features, poses):
        # self.curr_feat = features
        # if len(self.features) == self.features.maxlen:
        #     return
        # self.features.extend(features)

        self.curr_feat = features
        self.curr_pose = poses

        for feat, pose in zip(features, poses):
            if pose is None: continue
            if len(self.features) == self.features.maxlen: return
            num_point = sum(pose['keypoints'][:, 2] > 0.5)
            if num_point >= self.pose_thresh: self.features.append(feat)
        
        # if len(self.features) == 0:
        #     self.features.extend(features)
        
        if len(self.features) == 0:
            max_num, max_feat = 0, None
            for feat, pose in zip(features, poses):
                if pose is None: continue
                if sum(pose['keypoints'][:, 2] > 0.5) > max_num:
                    max_num = sum(pose['keypoints'][:, 2] > 0.5)
                    max_feat = feat
            
            if max_num > 0:
                self.features.append(max_feat)
            else:
                self.features.extend(features)
                
    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.centroid = new_track.centroid
        self.global_id = new_track.global_id

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.curr_pose)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.centroid = new_track.centroid
        self.global_id = new_track.global_id

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.curr_pose)

        self.state = TrackState.Tracked
        if self.tracklet_len > self.min_hits:
            self.is_activated = True


class MCTracker:
    def __init__(self, appearance_thresh=0.8, euc_thresh=0.5, match_thresh=0.8, map_size=None, max_time_lost=18000, min_hits=90):
        self.tracked_mtracks = []  # type: list[MTrack]
        self.lost_mtracks = []  # type: list[MTrack]
        self.removed_mtracks = []  # type: list[MTrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits

        self.appearance_thresh = appearance_thresh
        self.match_thresh = match_thresh
        # self.match_thresh = 0.99
        if map_size:
            # self.euc_thresh = (map_size[0] + map_size[1]) / 2.0 / 10.0
            # self.euc_thresh = np.sqrt(map_size[0]**2 + map_size[1]**2) / 10.0
            self.euc_thresh = euc_thresh
            self.max_len = np.sqrt(map_size[0]**2 + map_size[1]**2)
            print('max_len: ', self.max_len)
    
    def update(self, trackers, groups, scene=None):
        self.frame_id += 1
        activated_mtracks = []
        refind_mtracks = []
        lost_mtracks = []
        removed_mtracks = []

        if len(groups):  # group type: array[[t_groud_id, features, centroid(location)], ...]
            global_ids = groups[:, 0]
            # print('input length of global_ids: ', len(global_ids))
            features = groups[:, 1]
            centroids = groups[:, 2]
            poses = groups[:, 3]
        else:
            centroids = []
            features = []
            poses = []
        
        if len(centroids) > 0:
            new_groups = [MTrack(g, c, f, p, self.min_hits, scene) for (g, c, f, p) in zip(global_ids, centroids, features, poses)]
        else:
            new_groups = []

        ''' Step 1: Add newly detected groups to tracked_mtracks '''
        unconfirmed = []
        tracked_mtracks = []  # type: list[MTrack]
        for track in self.tracked_mtracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_mtracks.append(track)
        
        ''' Step 2: First Association with tracked mtracks '''
        mtrack_pool = joint_mtracks(tracked_mtracks, self.lost_mtracks)
        # exist_features = [feat for m in mtrack_pool for feat in list(m.features)]
        exist_features = [feat for m in tracked_mtracks for feat in list(m.features)]
        # lengths_exists = [len(m.features) for m in mtrack_pool]
        lengths_exists = [len(m.features) for m in tracked_mtracks]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]
        # print('exist: ', len(mtrack_pool))
        # print('new: ', len(new_groups))
        exist_centroids = [m.centroid for m in tracked_mtracks]
        new_centroids = [g.centroid for g in new_groups]

        shape = (len(lengths_exists), len(lengths_new))
        if 0 in shape:
            dists = np.empty(shape)
        elif scene=='S001':
            if self.frame_id % 5 == 0:
                rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
                emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
                dists = emb_dists
            else:
                rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
                emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
                euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / self.max_len
                norm_emb_dists = (emb_dists - np.min(emb_dists)) / (np.max(emb_dists) - np.min(emb_dists))
                norm_euc_dists = (euc_dists - np.min(euc_dists)) / (np.max(euc_dists) - np.min(euc_dists))
                dists = 0.5 * norm_euc_dists + 0.5 * norm_emb_dists
                dists[euc_dists > 0.25] = 1.0
        else:
            # rerank_dists = re_ranking(np.array(new_features), np.array(exist_features), 20, 6, 0.3)
            # rerank_dists = rerank_dists.transpose()
            rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
            euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / self.max_len

            dists = emb_dists

        # matches, u_exist, u_new = matching.linear_assignment(dists, thresh=self.match_thresh)
        matches, u_exist, u_new = matching.linear_assignment(dists, thresh=0.999)
        # print('u_exist, u_new: ', u_exist, u_new)

        for iexist, inew in matches:
            exist = tracked_mtracks[iexist]
            new = new_groups[inew]
            if exist.state == TrackState.Tracked:
                exist.update(new, self.frame_id)
                activated_mtracks.append(exist)
            else:
                exist.re_activate(new, self.frame_id, new_id=False)
                refind_mtracks.append(exist)
        # print('refind_mtracks: ', [t.track_id for t in refind_mtracks])
        
        for it in u_exist:
            track = tracked_mtracks[it]
            if not track.state == TrackState.Lost and (not track.state == TrackState.Removed):
                track.mark_lost()
                lost_mtracks.append(track)

        ''' Step 3: Second association with lost mtracks '''
        new_groups = [new_groups[i] for i in u_new]

        lost_features = [feat for m in self.lost_mtracks for feat in list(m.features)]
        lengths_lost = [len(m.features) for m in self.lost_mtracks]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]

        shape = (len(lengths_lost), len(lengths_new))
        if 0 in shape:
            emb_dists = np.empty((len(lengths_lost), len(lengths_new)))
        else:
            # rerank_dists = re_ranking(np.array(new_features), np.array(lost_features), 20, 6, 0.3)
            # rerank_dists = rerank_dists.transpose()
            rerank_dists = matching.embedding_distance(lost_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_lost, lengths_new, shape, normalize=False)

        dists = emb_dists

        # dists[emb_dists > self.appearance_thresh] = 1.0  # drop above appearance thresh
        # dists[emb_dists > 0.8] = 1.0  # drop above appearance thresh

        matches, u_lost, u_new = matching.linear_assignment(dists, thresh=self.match_thresh)

        for ilost, inew in matches:
            lost = self.lost_mtracks[ilost]
            new = new_groups[inew]
            lost.re_activate(new, self.frame_id, new_id=False)
            refind_mtracks.append(lost)
        # print('refind_mtracks: ', [t.track_id for t in refind_mtracks])

        ''' Step 4: Deal with unconfirmed tracks, usually tracks with only one beginning frame '''
        new_groups = [new_groups[i] for i in u_new]
        # print('unmatched_new: ', new_groups)
        # print('unmatched_exist: ', unconfirmed)

        exist_centroids = [m.centroid for m in unconfirmed]
        new_centroids = [g.centroid for g in new_groups]

        exist_features = [feat for m in unconfirmed for feat in list(m.features)]
        lengths_exists = [len(m.features) for m in unconfirmed]
        new_features = [feat for g in new_groups for feat in list(g.features)]
        lengths_new = [len(m.features) for m in new_groups]

        # euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / self.max_len

        shape = (len(lengths_exists), len(lengths_new))
        if 0 in shape:
            dists = np.empty(shape)
        elif scene=='S001':
            rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
            emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
            euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / self.max_len
            dists = emb_dists * euc_dists
            dists[euc_dists > 0.2] = 1.0
        else:
            # rerank_dists = re_ranking(np.array(new_features), np.array(exist_features), 20, 6, 0.3)
            # rerank_dists = rerank_dists.transpose()
            rerank_dists = matching.embedding_distance(exist_features, new_features) / 2.0
            # emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, euc_dists.shape, normalize=True)
            emb_dists = grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=False)
            euc_dists = matching.euclidean_distance(exist_centroids, new_centroids) / self.max_len

            dists = emb_dists
            dists[euc_dists > 0.1] = 1.0

        # matches, u_unconfirmed, u_new = matching.linear_assignment(dists, thresh=0.999 if scene=='S001' else self.match_thresh)
        matches, u_unconfirmed, u_new = matching.linear_assignment(dists, thresh=0.999)
        # print('u_exist, u_new: ', u_exist, u_new)
        for iexist, inew in matches:
            unconfirmed[iexist].update(new_groups[inew], self.frame_id)
            activated_mtracks.append(unconfirmed[iexist])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_mtracks.append(track)

        """ Step 5: Init new mtracks """
        for inew in u_new:
            track = new_groups[inew]
            # if track.score < self.new_track_thresh:
            #     continue
            track.activate(self.frame_id)
            activated_mtracks.append(track)
        
        """ Step 6: Update state """
        for track in self.lost_mtracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_mtracks.append(track)
        
        if scene=='S001' and self.frame_id == 7800:
            self.max_time_lost = 40
            self.min_hits = 3600
        elif scene=='S001' and self.frame_id == 9000:
            self.max_time_lost = 36000

        """ Merge """
        self.tracked_mtracks = [t for t in self.tracked_mtracks if t.state == TrackState.Tracked]
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, activated_mtracks)
        self.tracked_mtracks = joint_mtracks(self.tracked_mtracks, refind_mtracks)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, self.tracked_mtracks)
        self.lost_mtracks.extend(lost_mtracks)
        self.lost_mtracks = sub_mtracks(self.lost_mtracks, self.removed_mtracks)
        self.removed_mtracks.extend(removed_mtracks)
        # self.tracked_mtracks, self.lost_mtracks = remove_duplicate_mtracks(self.tracked_mtracks, self.lost_mtracks)

        output_mtracks = [track for track in self.tracked_mtracks if track.is_activated]
        unconfirmed_mtracks = [track for track in self.tracked_mtracks if not track.is_activated]
        print(f'tracking ids: {[m.track_id for m in output_mtracks]}')
        print(f'unconfirmed_tracks ids: {[m.track_id for m in unconfirmed_mtracks]}')
        print(f'lost_tracks ids: {[m.track_id for m in self.lost_mtracks]}')


def re_ranking(probFea,galFea,k1,k2,lambda_value, MemorySave = False, Minibatch = 2000):

    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]    
    feat = np.append(probFea,galFea,axis = 0)
    feat = feat.astype(np.float16)
    # print('computing original distance')
    if MemorySave:
        original_dist = np.zeros(shape = [all_num,all_num],dtype = np.float16)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it,] = np.power(cdist(feat[i:it,],feat, 'cosine'),2).astype(np.float16)
            else:
                original_dist[i:,:] = np.power(cdist(feat[i:,],feat, 'cosine'),2).astype(np.float16)
                break
            i = it
    else:
        original_dist = cdist(feat,feat, 'cosine').astype(np.float16)  
        original_dist = np.power(original_dist,2).astype(np.float16)
    del feat    
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    # print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float16)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])
    
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float16)

    
    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float16)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)
    
    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

def grouping_rerank(rerank_dists, lengths_exists, lengths_new, shape, normalize=True):
    emb_dists = np.zeros(shape, dtype=np.float)
    total_sum = np.sum(rerank_dists)
    num = 0
    ratio = 0
    for i, len_e in enumerate(lengths_exists):
        for j, len_n in enumerate(lengths_new):
            start_x = sum(lengths_exists[:i])
            end_x = start_x + lengths_exists[i]
            start_y = sum(lengths_new[:j])
            end_y = start_y + lengths_new[j]
            # emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) / total_sum
            if shape == (1,1):
                emb_dists[i,j] = np.mean(rerank_dists[start_x:end_x, start_y:end_y]) 
            else:
                emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) / total_sum / (len_e*len_n)
            # emb_dists[i,j] = np.sum(rerank_dists[start_x:end_x, start_y:end_y]) 
    max_val = np.max(emb_dists)
    min_val = np.min(emb_dists)
    if shape != (1, 1) and normalize:
        emb_dists = (emb_dists - min_val) / (max_val - min_val)
    return emb_dists

def joint_mtracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_mtracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


# def remove_duplicate_mtracks(stracksa, stracksb):
#     pdist = matching.iou_distance(stracksa, stracksb)
#     pairs = np.where(pdist < 0.15)
#     dupa, dupb = list(), list()
#     for p, q in zip(*pairs):
#         timep = stracksa[p].frame_id - stracksa[p].start_frame
#         timeq = stracksb[q].frame_id - stracksb[q].start_frame
#         if timep > timeq:
#             dupb.append(q)
#         else:
#             dupa.append(p)
#     resa = [t for i, t in enumerate(stracksa) if not i in dupa]
#     resb = [t for i, t in enumerate(stracksb) if not i in dupb]
#     return resa, resb