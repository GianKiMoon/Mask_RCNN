from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import numpy as np
import os


def get_dense_pose_mask(polys):
    mask_gen = np.zeros([256, 256])
    for i in range(1, 15):
        if polys[i-1]:
            print(polys[i-1], "\n")
            current_mask = mask_util.decode(polys[i-1])
            mask_gen[current_mask > 0] = i
    return mask_gen


if __name__ == '__main__':
    print("Beginning playground execution")

    coco_folder = '../datasets/coco/'
    dp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_minival.json')

    # Get img id's for the minival dataset.
    im_ids = dp_coco.getImgIds()
    # Select a random image id.
    Selected_im = im_ids[57]  # Choose im no 57 to replicate
    # Load the image
    im = dp_coco.loadImgs(Selected_im)[0]
    # Load Anns for the selected image.
    ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
    anns = dp_coco.loadAnns(ann_ids)

    for ann in anns:
        if 'dp_masks' in ann.keys():
            c = ann['iscrowd']
            print("IsCrowd: ", c)
            print(ann['bbox'])
            dp = ann['dp_masks']
            # GetDensePoseMask(ann['dp_masks'])
            y = np.array(ann['dp_y'])
            y_scale = y/4.5
            y_scale_floor = y//4.5
            print(y)
            print(y_scale)
            print(y_scale_floor)
            break

