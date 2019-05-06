from mrcnn import model as mrcnn_model
from samples.coco import coco as cocoMask
from samples.densepose import densepose as dpMask
import synthpod.utils as dp_util
from mrcnn import model as modelMask
from mrcnn import config as configMask
import synthpod.model as dp_model
import os
import skimage.color
import skimage.io
import skimage.transform
import cv2
import numpy as np
import pickle
import synthpod.encode_results_for_competition as enc
import tqdm as tqdm
import matplotlib.pyplot as plt

mask_rcnn_weights = ''
densepose_branch_weights = ''

###Eval Flags###

use_gt = True
save_detection_images = True

def load_image(image_path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

import seaborn as sns

def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(1, n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return seg_img

def detect(model, image, rois, class_ids, scores, masks):

    box_images, box_rois, box_scores = get_box_images(image, rois, masks, scores)

    m_instances = []
    i_instances = []
    u_instances = []
    v_instances = []

    for idx in range(box_images.__len__()):
        y1 = int(box_rois[idx][0])
        x1 = int(box_rois[idx][1])
        y2 = int(box_rois[idx][2])
        x2 = int(box_rois[idx][3])

        width = x2 - x1
        height = y2 - y1

        m, i, u, v = model.predict([np.expand_dims(box_images[idx], 0)])

        m_box = cv2.resize(m[0, :, :, :], (width, height), interpolation=0)
        i_box = cv2.resize(i[0, :, :, :], (width, height), interpolation=0)
        u_box = cv2.resize(u[0, :, :, :], (width, height), interpolation=0)
        v_box = cv2.resize(v[0, :, :, :], (width, height), interpolation=0)

        i_img = dp_util.give_color_to_seg_img(np.argmax(i_box, 2), 25)
        # cv2.imshow('image', i_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        m_instances.append(m_box)
        i_instances.append(i_box)
        u_instances.append(u_box)
        v_instances.append(v_box)

    return m_instances, i_instances, u_instances, v_instances, box_rois, box_scores


def get_box_images(image, rois, masks, scores):

    masks = np.concatenate(masks, axis=2)

    box_images = []
    box_rois = []
    box_scores = []
    for i in range(rois.__len__()):
        if not scores:
            score = 1
        else:
            score = scores[i]

        if score >= 0.85:
            y1 = int(rois[i][0])
            x1 = int(rois[i][1])
            y2 = int(rois[i][2])
            x2 = int(rois[i][3])
            box_img = image[y1:y2, x1:x2, :]
            box_img = cv2.resize(box_img, (256, 256))

            box_mask = masks[:, :, i]
            box_mask = box_mask[y1:y2, x1:x2].astype(np.int8)
            box_mask = np.expand_dims(box_mask, 2)
            box_mask = cv2.resize(box_mask, (256, 256), interpolation=0)

            # Center images to interval [-1, 1]
            box_img = np.float32(box_img) / 127.5 - 1
            show_mask = np.zeros_like(box_mask).astype(np.uint16)
            show_mask[np.where(box_mask == 1)] = 255
            show_mask = cv2.cvtColor(show_mask, cv2.COLOR_GRAY2RGB)
            # cv2.imshow('image', np.array(show_mask, dtype=np.uint8))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            box_img = np.concatenate([box_img, np.expand_dims(box_mask, 2)], 2)

            box_images.append(box_img)
            box_rois.append(rois[i])
            box_scores.append(score)

    return box_images, box_rois, box_scores


def format_uv_detection(dp_i, dp_u, dp_v, width, height):

    output = np.zeros([3, int(height), int(width)], dtype=np.float32)
    indexUV = np.argmax(dp_i, 2)
    output[0] = indexUV

    for part_id in range(1, 25):
        CurrentU = dp_u[:, :, part_id]
        CurrentV = dp_v[:, :, part_id]
        output[1, indexUV == part_id] = CurrentU[indexUV == part_id] * 255
        output[2, indexUV == part_id] = CurrentV[indexUV == part_id] * 255

    return output


if __name__ == "__main__":
    # Load Models

    class InferenceConfig(dpMask.DenseposeConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0

    config = InferenceConfig()
    maskModel = modelMask.MaskRCNN(mode="inference", config=config,
                                  model_dir="../logs")#"../logs/coco20190205T0843/mask_rcnn_coco_0010.h5")
    maskModel.load_weights("../mask_rcnn_densepose_0003.h5", by_name=True)
                       #exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    model_densepose = dp_model.ResNet(nClasses=15,
                            input_height=256,
                            input_width=256, mode="inference")

    model_densepose.load_weights("dp20190128T0951/mask_rcnn_dp_0022.h5") ###BaselineDP
    # model_densepose.load_weights("dp20190215T1007/mask_rcnn_dp_0009.h5") ###TransferSynthDP
    #model_densepose.load_weights("dp20190216T0811/mask_rcnn_dp_0020.h5") ###PreSynthDP

    # Load Dataset

    subset = "minival"
    dataset_dir = "../datasets/coco"
    if subset == "minival" or subset == "valminusminival":
        subset = "val"
    image_dir = "{}/{}2014".format(dataset_dir, subset)
    coco = cocoMask.CocoDataset.load_coco(cocoMask.CocoDataset(),
                                          "../datasets/coco", "minival",
                                          return_coco=True)

    image_ids = list(coco.imgs.keys())
    results = []

    for i in tqdm.tqdm(image_ids):
        path = os.path.join(image_dir, coco.imgs[i]['file_name'])
        img = load_image(path)

        if use_gt:

            anns = coco.loadAnns(coco.getAnnIds(imgIds=[i], iscrowd=None))

            rois = []
            masks = []
            for idx, ann in enumerate(anns):
                if 'dp_masks' in ann and ann['dp_I'] and ann['dp_U'] and ann['dp_V']:
                    roi = np.array(ann['bbox'])
                    roi_t = roi[[1, 0, 3, 2]]
                    rois.append(np.array([roi_t[0], roi_t[1], roi_t[0] + roi_t[2], roi_t[1] + roi_t[3]]))
                    mask = coco.annToMask(ann)
                    mask = np.expand_dims(mask, 2)
                    masks.append(mask)

                    # res = maskModel.detect([img])[0]
                    # det_roi = res['rois']
                    # print()

            if not rois:
                continue

            dp_masks, dp_is, dp_us, dp_vs, rois, scores = detect(model_densepose, img,
                                                                         rois, [], [],
                                                                         masks)

        else:
            res = maskModel.detect([img])[0]
            dp_masks, dp_is, dp_us, dp_vs, rois, scores = detect(model_densepose, img,
                                                             res['rois'], res['class_ids'], res['scores'], res['masks'])

        IUV_fields = []
        for j in range(rois.__len__()):
            y1 = int(rois[j][0])
            x1 = int(rois[j][1])
            y2 = int(rois[j][2])
            x2 = int(rois[j][3])
            width = x2 - x1
            height = y2 - y1

            uv_det = format_uv_detection(dp_is[j], dp_us[j], dp_vs[j], width, height)
            IUV_fields.append(uv_det)
            if use_gt:
                score = 1.0
            else:
                score = scores[j].astype(np.float)

            results.extend([{'image_id': i, 'category_id': 1, 'uv': uv_det.astype(np.uint8),
                             'bbox': [x1, y1, width, height], 'score': score}])

        if save_detection_images:
            im_name = i
            All_Coords = np.zeros(img.shape)
            All_inds = np.zeros([img.shape[0], img.shape[1]])
            K = 26
            ##
            boxes = np.vstack(rois)
            ##

            for i in range(IUV_fields.__len__()):
                entry = boxes[i, :]
                height_percent = (entry[2] - entry[0]) / img.shape[0]
                if scores[i] > 0.90 and height_percent > 0.50:
                    entry = entry.astype(int)
                    ####
                    output = IUV_fields[i]
                    uv = output.transpose([1, 2, 0])[:, :, 1:3]
                    i = output.transpose([1, 2, 0])[:, :, 0] ### ?
                    All_Coords[entry[0]:entry[2], entry[1]:entry[3], 1:3] = uv
                    All_inds[entry[0]:entry[2], entry[1]:entry[3]] = np.squeeze(i)

            output_dir = "synthpod_detections_new_baseline/"
            IUV_SaveName = str(im_name) + '_IUV.png'
            INDS_SaveName = str(im_name) + '_INDS.png'
            cv2.imwrite(os.path.join(output_dir, '{}'.format(IUV_SaveName)), All_Coords)
            t = give_color_to_seg_img(All_inds, 25)
            cv2.imwrite(os.path.join(output_dir, '{}'.format(INDS_SaveName)), (t * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(output_dir, '{}'.format(str(im_name) + '_Original.png')), cv2.imread(path))


            def make_image(outputname, size=(12, 12), dpi=80):
                fig = plt.figure()
                fig.set_size_inches(size)


                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                #ax.imshow(data, aspect='equal')
                ax.imshow(plt.imread(path), aspect='auto')
                ax.contour(All_Coords[:, :, 1] / 256., 50, linewidths=1)
                ax.contour(All_Coords[:, :, 2] / 256., 50, linewidths=1)
                plt.savefig(outputname, dpi=dpi)

            #fig = plt.figure(figsize=[12, 12])

            make_image(os.path.join(output_dir, "{}_Contour.png".format(im_name)), size=(img.shape[1] / 80, img.shape[0] / 80))
            #plt.axis('off')
            #plt.savefig(os.path.join(output_dir, "{}_Contour.png".format(im_name)), bbox_inches='tight', pad_inches=0)
            #plt.close(fig)

    with open('val_res_minival_base_iuv_only.pkl', 'wb') as f:
        pickle.dump(results, f, 0)


    # with open('uv_results.pkl', 'rb') as hIn, \
    #         open('uv_results.json', 'w') as hOut:
    #     enc._savePngJson(hIn, hOut)