# from detection_coco.models.model import NPNet
# from detection_v5.models.model import NPNet
from detection_eva.model import EVA
# from reids.fastreid.models.model import ReID
from reids.ensemble.model import ReIDEnsemble
from trackers.botsort.bot_sort import BoTSORT

from trackers.multicam_tracker.cluster_track import MCTracker
from trackers.multicam_tracker.clustering import Clustering, ID_Distributor

from mmpose.apis import init_pose_model

from perspective_transform.model import PerspectiveTransform
from perspective_transform.calibration import calibration_position
from tools.utils import (_COLORS, get_reader_writer, finalize_cams, write_vids, write_results_testset, 
                    visualize, update_result_lists_testset, sources, result_paths, map_infos, write_map, cam_ids)

import cv2
import os
import time
import numpy as np
import argparse


def run(args, conf_thres, iou_thres, sources, result_paths, perspective, cam_ids):
    # assert len(sources) == len(result_paths[0]), 'length of sources and result_paths is different'
    # detection model initilaize
    # NPNet.initialize()
    # detection = NPNet()
    # detection.conf_thres = conf_thres
    # detection.iou_thres = iou_thres
    # classes = detection.classes
    detection = EVA()
    detection.initialize('detection_eva/eva_s001_results.json')
    classes = detection.classes
    
    # reid model initilaize
    # ReID.initialize(max_batch_size=args['max_batch_size'])
    # reid = ReID()
    reid = ReIDEnsemble()
    reid.initialize(max_batch_size=args['max_batch_size'])

    # pose estimation initialize
    config_file = './configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/crowdpose/hrnet_w32_crowdpose_256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_crowdpose_256x192-960be101_20201227.pth'
    pose = init_pose_model(config_file, checkpoint_file, device='cuda:0')

    # trackers initialize
    # trackers = [BoTSORT(track_buffer=args['track_buffer'], max_batch_size=args['max_batch_size']) for i in range(len(sources))]
    trackers = []
    for i in range(len(sources)):
        trackers.append(BoTSORT(track_buffer=args['track_buffer'], max_batch_size=args['max_batch_size'], 
                            appearance_thresh=args['sct_appearance_thresh'], euc_thresh=args['sct_euclidean_thresh']))

    # perspective transform initialize
    calibrations = calibration_position[perspective]
    perspective_transforms = [PerspectiveTransform(c, map_infos[perspective]['size'], args['ransac_thresh']) for c in calibrations]

    # id_distributor and multi-camera tracker initialize
    clustering = Clustering(appearance_thresh=args['clt_appearance_thresh'], euc_thresh=args['clt_euclidean_thresh'],
                            match_thresh=0.8, map_size=map_infos[perspective]['size'])
    mc_tracker = MCTracker(appearance_thresh=args['mct_appearance_thresh'], match_thresh=0.8, map_size=map_infos[perspective]['size'],
                        max_time_lost=36000, min_hits=30)
    id_distributor = ID_Distributor()

    # get source imgs, video writers
    src_handlers = [get_reader_writer(s) for s in sources]
    results_lists = [[] for i in range(len(sources))]  # make empty lists to store tracker outputs in MOT Format
    map_img = cv2.imread(map_infos[perspective]['source'])
    map_writer = cv2.VideoWriter(map_infos[perspective]['savedir'], cv2.VideoWriter_fourcc(*'mp4v'), 30, map_infos[perspective]['size'])

    # total_frames = len(src_handlers[0][0])
    total_frames = max([len(s[0]) for s in src_handlers])
    total_cams = len(sources)
    cur_frame = 0
    stop = False
    final_results_lists = {}

    while True:
        if cur_frame == 25000:  # SKIP
            # final_results_lists = {_: results for _, results in enumerate(results_lists)}
            break
        imgs = []
        start = time.time()

        # first, run trackers each frame independently
        for i, ((img_paths, writer), tracker, perspective_transform, result_list) in enumerate(zip(src_handlers, trackers, perspective_transforms, results_lists)):
            if len(img_paths) == 0:
                if i not in final_results_lists:
                    final_results_lists[i] = result_list
                    trackers[i].tracked_stracks = []
                if len(final_results_lists) == total_cams:
                    stop = True
                    break
                imgs.append(np.zeros((1080, 1920, 3), np.uint8))
                continue
            # if  180 < cur_frame and cur_frame < 4700 or (5300 < cur_frame and cur_frame < 6000):  # SKIP
            # if  180 < cur_frame and cur_frame < 4700 or (5300 < cur_frame and cur_frame < 6000):  # SKIP
            if cur_frame < 20000:
                img_paths.pop(0)
                continue
            img_path = img_paths.pop(0)
            img = cv2.imread(img_path)
            # img = cv2.imread(img_paths.pop(0))
            dets = detection.run(img_path)  # run detection model
            # dets = detection.run(img)  # run detection model 
            online_targets = tracker.update(np.array(dets), img, reid, pose)  # run tracker
            perspective_transform.run(tracker)  # run perspective transform

            # assign global_id to each track for multi-camera tracking
            for t in tracker.tracked_stracks:
                t.t_global_id = id_distributor.assign_id()  # assign temporal global_id
            imgs.append(img)
        if stop: break
        
        cur_frame += 1
        # if 180 < cur_frame and cur_frame < 4700 or (5300 < cur_frame and cur_frame < 6000): continue  # SKIP

        # second, run multi-camera tracker using above trackers results
        groups = clustering.update(trackers, cur_frame)
        mc_tracker.update(trackers, groups)
        latency = time.time() - start

        # update result lists using updated trackers
        update_result_lists_testset(trackers, results_lists, cur_frame, cam_ids)

        if args['write_vid']:
            write_vids(trackers, imgs, src_handlers, latency, pose, _COLORS, mc_tracker, cur_frame)
            map_img = write_map(trackers, map_img, map_writer, _COLORS, mc_tracker, cur_frame)
        
        print(f"video frame ({cur_frame}/{total_frames}) ({latency:.6f} s)")
        # cur_frame += 1
    
    finalize_cams(src_handlers)
    map_writer.release()
    # third, postprocess on final results
    # mc_tracker.postprocess(results_lists)  # todo

    # save results txt
    results_lists = list(final_results_lists.values())
    write_results_testset(results_lists, result_paths)

    # NPNet.finalize()
    # ReID.finalize()
    reid.finalize()
    print('Done')


if __name__ == '__main__':
    args = {
        'max_batch_size' : 16,  # maximum input batch size of reid model
        'track_buffer' : 150,  # the frames for keep lost tracks
        'with_reid' : True,  # whether to use reid model's out feature map at first association

        'sct_appearance_thresh' : 0.4,  # threshold of appearance feature cosine distance when do single-cam tracking
        'sct_euclidean_thresh' : 0.1,  # threshold of euclidean distance when do single-cam tracking

        # 'clt_appearance_thresh' : 0.25,  # threshold of appearance feature cosine distance when do multi-cam clustering
        # 'clt_euclidean_thresh' : 0.2,  # threshold of euclidean distance when do multi-cam clustering
        'clt_appearance_thresh' : 0.07,  # threshold of appearance feature cosine distance when do multi-cam clustering
        # 'clt_euclidean_thresh' : 0.35,  # threshold of euclidean distance when do multi-cam clustering
        'clt_euclidean_thresh' : 0.05,  # threshold of euclidean distance when do multi-cam clustering

        'mct_appearance_thresh' : 0.4,  # threshold of appearance feature cosine distance when do cluster tracking (not important)

        'ransac_thresh' : 10,  # threshold of ransac when find homography matrix 
        'frame_rate' : 30,  # your video(camera)'s fps
        'write_vid' : True,  # write result to video
        }

    run(args=args, conf_thres=0.1, iou_thres=0.45, sources=sources['S001'], result_paths=result_paths['S001'], perspective='S001', cam_ids=cam_ids['S001'])