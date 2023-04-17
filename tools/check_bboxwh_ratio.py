
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

def get_dists(space_name, sets):
    print(f"\nStart {sets}/{space_name}")
    total_ratios = []
    ratios_sets = []
    cams = os.listdir(f'Frames/{sets}/{space_name}')
    cams = sorted([cam for cam in cams if 'c' in cam])
    for cam in cams:
        gt_txts = Path("./MOTFormat/singlecam/gt/").glob("**/*.txt")
        gt_txts = [str(p) for p in gt_txts if space_name in str(p) and cam in str(p)]

        ratios = []
        for gt_txt in gt_txts:
            with open(gt_txt, 'r') as f:
                labels = f.readlines()
            for l in labels:
                frame_id, obj_id, t, l, w, h, _, _, _, _ = l.split(",")
                frame_id, obj_id = int(frame_id), int(obj_id)
                t, l, w, h = float(t), float(l), float(w), float(h)
                ratios.append(h/w)
                total_ratios.append(h/w)
        print(f"Statistics of {sets}/{space_name}/{cam}")
        print(f"Median: {np.median(ratios)} / Mean: {np.mean(ratios)}")
        ratios_sets.append(ratios)

    print(f"\nStatistics of {sets}/{space_name}/total")
    print(f"Median: {np.median(total_ratios)} / Mean: {np.mean(total_ratios)}")
    ratios_sets.append(total_ratios)
    cams.append('total')
    plot_boxplot(ratios_sets, cams, space_name)


def plot_boxplot(ratios, names, space):
    fig, ax = plt.subplots()

    bp = ax.boxplot(ratios)

    # set x-axis tick labels
    ax.set_xticklabels(names)

    # set title and axis labels
    ax.set_title('Boxplot of Ratio (Height/Width)')
    ax.set_xlabel('Each Cameras')
    ax.set_ylabel('Ratios')

    # save the figure as an image
    fig_path = f'tools/outputs/wh_ratio_boxplot_{space}.png'
    fig.savefig(fig_path)
    print(f'boxplot saved at {fig_path}')

if __name__ == "__main__":

    spaces = ["S005", "S008", "S013", "S017", "S020"]

    for space in spaces:
        get_dists(space, "validation")

    # id_dists = []
    # for space in spaces:
    #     get_dists(same_id_dists, diff_id_dists, space, "validation", reid, sample_num)
    # plot(same_id_dists, diff_id_dists, "Total")
