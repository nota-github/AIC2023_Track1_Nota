from pathlib import Path
from tqdm import tqdm
from scipy.stats import f_oneway, zscore
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import sys
sys.path.append('.')
from perspective_transform.model import PerspectiveTransform
from perspective_transform.calibration import calibration_position
from tools.utils import sources, map_infos, _COLORS
from trackers.multicam_tracker.matching import embedding_distance

def compute_reprojection_error(src_pts, dst_pts, H):
    """
    Reprojection error: The reprojection error is the difference between the projected points and the actual points. 
    A lower reprojection error means a better homography matrix.
    """
    projected_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H)
    error = np.sqrt(np.sum((projected_pts - dst_pts.reshape(-1, 1, 2)) ** 2, axis=2))
    return error.mean()

def compute_condition_number(H):
    """
    Condition number: The condition number measures the sensitivity of the homography matrix to small changes in the input points. 
    A lower condition number means a more stable homography matrix.
    """
    return np.linalg.cond(H)
    # u, s, v = np.linalg.svd(H)
    # return np.abs(s[0] / s[-1])

def compute_ransac_inliers(mask):
    """
    RANSAC inliers: RANSAC is an iterative algorithm that can estimate the homography matrix from noisy data. 
    The number of inliers found by RANSAC is a good indicator of the quality of the homography matrix.
    """
    return len(mask), mask.sum()

def get_dists(space_name, ransac_thresh):
    print(f"\nStart {space_name}\n")

    gt_txts = Path("./MOTFormat/singlecam/gt/").glob("**/*.txt")
    gt_txts = [str(p) for p in gt_txts if space_name in str(p)]
    gt_txts = sorted(gt_txts,  key=lambda x: int(str(x.split("_")[-1][1:4])))
    print("candidate sets: ", gt_txts)

    obj_ids = set()
    gt_labels = []
    for gt_txt in gt_txts:
        gt_label = {}
        with open(gt_txt, 'r') as f:
            labels = f.readlines()
        for l in labels:
            frame_id, obj_id, t, l, w, h, _, _, _, _ = l.split(",")
            frame_id, obj_id = int(frame_id), int(obj_id)
            t, l, w, h = float(t), float(l), float(w), float(h)
            info = [t + w/2, l + h]
            obj_ids.add(obj_id)

            if frame_id in gt_label:
                gt_label[frame_id][obj_id]=info
            else:
                gt_label[frame_id] = {obj_id: info}
        gt_labels.append(gt_label)
    print(f"\nTotal object ids: {obj_ids}")

    perspective = space_name
    calibrations = calibration_position[perspective]
    perspective_transforms = [PerspectiveTransform(c, map_infos[perspective]['size'], ransac_thresh) for c in calibrations]
    map_size = np.sqrt(map_infos[perspective]['size'][0]**2 + map_infos[perspective]['size'][1]**2)

    for i, p in enumerate(perspective_transforms):
        c_num = compute_condition_number(p.homography_matrix)
        total, i_num = compute_ransac_inliers(p.mask)
        print(f"\n{i}th homography matrix's condition number: {c_num}")
        print(f"{i}th homography matrix's inliers number: {i_num}/{total}")

    """
    gt_labels = [
        {"frame_id 1" : {obj_id: [x,y], obj_id: [x,y], ...}, "frame_id 2" : {obj_id: [x,y], obj_id: [x,y], ...}, ...  # cam1
        {"frame_id 1" : {obj_id: [x,y], obj_id: [x,y], ...}, "frame_id 2" : {obj_id: [x,y], obj_id: [x,y], ...}, ...  # cam2
        ...
    ]
    """
    distsets_per_cam = [[] for i in range(len(gt_labels))]

    start, end = min(list(gt_labels[0].keys())), max(list(gt_labels[0].keys()))
    print(f"\nstrat frame id: {start}, end frame id : {end}")
    for i in tqdm(range(start, end+1)):
        id_exist = {obj_id : False for obj_id in obj_ids}
        for cam_id, cam in enumerate(gt_labels):
            if i not in cam: continue
            objects = cam[i]
            for obj_id in objects:
                id_exist[obj_id] = True
                coord = objects[obj_id]
                coord = perspective_transforms[cam_id].run_for_check(coord)
                coord = [c[0] for c in coord]
                coords_sets = []
                for cam_id_2nd, cam_2nd in enumerate(gt_labels):
                    if (cam_id_2nd==cam_id) or (i not in cam_2nd) or (obj_id not in cam_2nd[i]): continue
                    coord_2nd = cam_2nd[i][obj_id]
                    coord_2nd = perspective_transforms[cam_id_2nd].run_for_check(coord_2nd)
                    coord_2nd = [c[0] for c in coord_2nd]
                    coords_sets.append(coord_2nd)
                if len(coords_sets)==0: continue
                mean_coord = np.mean(coords_sets, axis=0)
                dist = np.linalg.norm(coord - mean_coord)
                # dist /= map_size
                distsets_per_cam[cam_id].append(dist)
            # if False in id_exist.values():  #
            #     print(f"obj_id not exist at frame {i}, {id_exist}")

    plot_boxplot(distsets_per_cam, space_name)

    # get ANOVA table
    f_statistic, p_value = f_oneway(distsets_per_cam[0], distsets_per_cam[1], distsets_per_cam[2], distsets_per_cam[3], distsets_per_cam[4])  # modify to match the number of Cam

    print("F-statistic:", f_statistic)
    print("p-value:", p_value)

def draw_on_map(space_name, ransac_thresh):
    print(f"\nStart {space_name}\n")

    gt_txts = Path("./MOTFormat/singlecam/gt/").glob("**/*.txt")
    gt_txts = [str(p) for p in gt_txts if space_name in str(p)]
    gt_txts = sorted(gt_txts,  key=lambda x: int(str(x.split("_")[-1][1:4])))
    print("candidate sets: ", gt_txts)

    obj_ids = set()
    gt_labels = []
    for gt_txt in gt_txts:
        gt_label = {}
        with open(gt_txt, 'r') as f:
            labels = f.readlines()
        for l in labels:
            frame_id, obj_id, t, l, w, h, _, _, _, _ = l.split(",")
            frame_id, obj_id = int(frame_id), int(obj_id)
            t, l, w, h = float(t), float(l), float(w), float(h)
            info = [[t + w/2, l + h], [t, l + h]]
            obj_ids.add(obj_id)

            if frame_id in gt_label:
                gt_label[frame_id][obj_id]=info
            else:
                gt_label[frame_id] = {obj_id: info}
        gt_labels.append(gt_label)
    print(f"\nTotal object ids: {obj_ids}")

    perspective = space_name
    calibrations = calibration_position[perspective]
    perspective_transforms = [PerspectiveTransform(c, map_infos[perspective]['size'], ransac_thresh) for c in calibrations]

    distsets_per_cam = [[] for i in range(len(gt_labels))]
    map_img = cv2.imread(map_infos[perspective]['source'])
    map_path = f'output_videos/check_homography_{space_name}.mp4'
    map_writer = cv2.VideoWriter(map_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, map_infos[perspective]['size'])

    start, end = min(list(gt_labels[0].keys())), max(list(gt_labels[0].keys()))
    print(f"\nstrat frame id: {start}, end frame id : {end}")
    for i in tqdm(range(start, end+1)):
        # if i ==300: break
        img = map_img.copy()
        id_exist = {obj_id : False for obj_id in obj_ids}
        for cam_id, cam in enumerate(gt_labels):
            if i not in cam: continue
            objects = cam[i]
            for obj_id in objects:
                id_exist[obj_id] = True
                coord, coord_left = objects[obj_id]
                coord, confidence = perspective_transforms[cam_id].run_for_check(coord, coord_left)
                coord = [c[0] for c in coord]
                img = visualize_map(coord, img, cam_id, obj_id, confidence, _COLORS)
        map_writer.write(img)
        # if False in id_exist.values():  #
        #     print(f"obj_id not exist at frame {i}, {id_exist}")
    map_writer.release()
    print("Drawing map Ended")
    print(f"saved path: {map_path}")
                    
def visualize_map(loc, img, cam_id, track_id, confidence, colors):
    m = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f'{cam_id}/{track_id}/{confidence:3f}'
    txt_size = cv2.getTextSize(text, font, 0.4*m, 1*m)[0]
    color = (colors[track_id%80] * 255).astype(np.uint8).tolist()
    # color = (colors[cam_id%80] * 255).astype(np.uint8).tolist()
    cv2.line(img, loc, loc, color, m)
    cv2.putText(img, text, (loc[0], loc[1] + txt_size[1]), font, 0.4*m, color, thickness=1*m)
    return img

def plot_boxplot(distsets, space):
    fig, ax = plt.subplots()

    bp = ax.boxplot(distsets)

    cams = [cam['name'] for cam in calibration_position[space]]

    # set x-axis tick labels
    ax.set_xticklabels(cams)

    # set title and axis labels
    ax.set_title('Boxplot of Distance between Estimated Coordinates')
    ax.set_xlabel('Each Cameras')
    ax.set_ylabel('Distances')

    # save the figure as an image
    fig_path = f'tools/outputs/homography_boxplot_{space}.png'
    fig.savefig(fig_path)
    
    for cam, dist in zip(cams, distsets):
        print(f"{cam}'s number of elements: {len(dist)} / number of outliers: {count_outliers(dist)} / ratio of outliers: {count_outliers(dist)/len(dist):5f}")

    print(f'\nboxplot saved at {fig_path}')

def count_outliers(lst):
    q1, q3 = np.percentile(lst, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = [x for x in lst if x < lower_bound or x > upper_bound]
    return len(outliers)

if __name__ == "__main__":

    spaces = ["S005", "S008", "S013", "S017", "S020"]
    ransac_thresh = 10

    get_dists("S005", ransac_thresh)
    # draw_on_map("S005", ransac_thresh)

    # get_dists("S008", ransac_thresh)
    # draw_on_map("S008", ransac_thresh)

    # get_dists("S013", ransac_thresh)
    # draw_on_map("S013", ransac_thresh)

    # get_dists("S017", ransac_thresh)
    # draw_on_map("S017", ransac_thresh)

    # get_dists("S020", ransac_thresh)
    # draw_on_map("S020", ransac_thresh)
    