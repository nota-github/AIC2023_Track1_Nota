import cv2
import numpy as np
from collections import deque
from mmpose.apis import inference_topdown

from . import matching
# from .gmc import GMC
from .fast_reid_interfece import FastReIDInterface

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
import pdb

class ID_Assigner:
    def __init__(self, init_id=0):
        self.cur_id = init_id

    def next_id(self):
        self.cur_id += 1
        return self.cur_id

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, pose=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        self.pose = pose
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.centroid = np.asarray(self._tlwh[:2] + self._tlwh[2:] / 2, dtype=np.float)
        self.t_global_id = 0
        self.global_id = 0

        self.matched_dist = None

    def update_features(self, feat):
        # feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, id_assigner=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if not id_assigner:
            self.track_id = self.next_id()
        else:
            self.track_id = id_assigner.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, id_assigner=None):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            if not id_assigner:
                self.track_id = self.next_id()
            else:
                self.track_id = id_assigner.next_id()
        self.score = new_track.score
        self.pose = new_track.pose

        self.centroid = self.tlwh_to_xywh(new_track.tlwh)[:2]

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

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.pose = new_track.pose

        self.centroid = self.tlwh_to_xywh(new_track.tlwh)[:2]

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    # def __init__(self, track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=30, 
    #             match_thresh=0.8, with_reid=True, proximity_thresh=0.5, appearance_thresh=0.4, euc_thresh=0.1, 
    #             fuse_score=True, frame_rate=30, max_batch_size=8, map_len=None, real_data=True):
    def __init__(self, track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=30, 
                match_thresh=0.8, with_reid=True, proximity_thresh=0.5, appearance_thresh=0.4, euc_thresh=0.1, 
                fuse_score=True, frame_rate=30,  map_len=None, real_data=True):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        # self.max_time_lost = self.buffer_size
        self.max_time_lost = 1
        self.kalman_filter = KalmanFilter()

        self.match_thresh = match_thresh
        self.fuse_score = fuse_score

        # ReID module
        self.with_reid = with_reid
        self.real_data = real_data
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.euc_thresh = euc_thresh
        # self.max_batch_size = max_batch_size

        self.max_len = map_len if map_len else np.sqrt(1920**2 + 1080**2)
        # print(f'max_len: {self.max_len}')

        self.id_assigner = ID_Assigner()
        # self.id_assigner = None
        if self.real_data:
            self.encoder = [
                FastReIDInterface('./configs/reid/Market1501/mgn_R50-ibn.yml', './pretrained/market_mgn_R50-ibn.pth', 'cuda'), 
                FastReIDInterface('./configs/reid/DukeMTMC/sbs_R101-ibn.yml', './pretrained/duke_sbs_R101-ibn.pth', 'cuda'), 
                FastReIDInterface('./configs/reid/MSMT17/AGW_S50.yml', './pretrained/msmt_agw_S50.pth', 'cuda'), 
            ]
        else:
            self.encoder = [FastReIDInterface('./configs/reid/AIC/bagtricks_R50.yml', './pretrained/market_aic_bot_R50_1.pth', 'cuda')]
        # self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update(self, output_results, img, pose):  # encoder, pose 추가
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            elif output_results.shape[1] == 6:  # 추가
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, 5].astype(np.uint8)
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]
            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            # Find high threshold detections
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            # pose_input = [{"bbox": det} for det in dets]
            pose_input = dets
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []
            pose_input = []

        if len(dets) > 0:
            '''Detections'''
            if self.with_reid:
                '''Extract embeddings '''
                ap = [encoder.inference(img, dets) for encoder in self.encoder]
                features_keep = np.mean(ap, axis=0)
                pose_result = inference_topdown(pose, img, pose_input, bbox_format='xyxy')
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f, {'keypoints': np.concatenate([p.pred_instances.keypoints[0], np.expand_dims(p.pred_instances.keypoint_scores[0], axis=1)], axis=1)}) for
                              (tlbr, s, f, p) in zip(dets, scores_keep, features_keep, pose_result)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        # ious_dists = matching.iou_distance(strack_pool, detections)
        # ious_dists_mask = (ious_dists > self.proximity_thresh)
        # ious_dists_mask = (ious_dists > 0.8)

        # if self.fuse_score:
            # ious_dists = matching.fuse_score(ious_dists, detections)
        
        centroid_dists = matching.centroid_distance(strack_pool, detections)
        centroid_dists /= self.max_len
        # centroid_dists_mask = (centroid_dists > self.proximity_thresh)

        if self.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            # raw_emb_dists = emb_dists.copy()
            # emb_dists[emb_dists > 0.4] = 1.0
            # emb_dists[ious_dists_mask] = 1.0
            # emb_dists[centroid_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)
            # dists = emb_dists
            # dists = ious_dists

            dists = 0.3 * centroid_dists + 0.7 * emb_dists 
            # dists = emb_dists
            dists[centroid_dists > self.euc_thresh] = 1.0
            dists[emb_dists > self.appearance_thresh] = 1.0

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, id_assigner=self.id_assigner)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, id_assigner=self.id_assigner)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        # ious_dists = matching.iou_distance(unconfirmed, detections)
        # ious_dists_mask = (ious_dists > self.proximity_thresh)
        # if self.fuse_score:
        #     ious_dists = matching.fuse_score(ious_dists, detections)
        centroid_dists = matching.centroid_distance(unconfirmed, detections)
        centroid_dists /= self.max_len

        if self.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
            # raw_emb_dists = emb_dists.copy()
            # emb_dists[emb_dists > self.appearance_thresh] = 1.0
            # emb_dists[ious_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)

            dists = 0.3 * centroid_dists + 0.7 * emb_dists 
            # dists = emb_dists
            dists[centroid_dists > self.euc_thresh] = 1.0
            dists[emb_dists > self.appearance_thresh] = 1.0
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id, id_assigner=self.id_assigner)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks


def joint_stracks(tlista, tlistb):
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


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
