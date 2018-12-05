"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class DenseposeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "densepose"

    GPU_COUNT = 3

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1   # Background + human

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class DenseposeDataset(utils.Dataset):

    def load_densepose_coco(self, dataset_dir, subset, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        coco = COCO("{}/annotations/densepose_coco_2014_{}.json".format(dataset_dir, subset))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}2014".format(dataset_dir, subset)

        # All images
        image_ids = list(coco.imgs.keys())

        # Add one additional class for persons
        self.add_class("coco", 1, "person")

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(DenseposeDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))

                
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(DenseposeDataset, self).load_mask(image_id)

    def load_uv(self, image_id):
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(DenseposeDataset, self).load_mask(image_id)

        #print(image_info)

        instance_dps = []
        instance_dp_masks = []
        instance_bboxs = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.

        #print("--------->")
        #print(annotations)
        #print(len(annotations))
        
        counter = 0

        for idx, annotation in enumerate(annotations):
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))

            #print(class_id)
            # print(counter)
            counter = counter + 1

            if 'dp_masks' in annotation:
               
                dp, dp_mask  = self.annToUV(annotation, image_info["height"],
                                       image_info["width"]) 
                assert (dp.shape == (5, 196))
                instance_dps.append(dp)
                instance_dp_masks.append(dp_mask)   
                class_ids.append(class_id)
            else:

                dp = np.full((5, 196), -1)
                assert (dp.shape == (5, 196))
                instance_dps.append(dp)
                class_ids.append(class_id)

        if class_ids:
            #print("Stack 1")
            dps = np.stack(instance_dps, axis=2)
            #dp_masks = np.stack(instance_dp_masks, axis=2)
            #print(dps.shape)
            return dps
        else:
            #print("i")
            # Call super class to return an empty mask
            return super(DenseposeDataset, self).load_mask(image_id)

###
        '''
        for idx, annotation in enumerate(annotations):
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))

            print(class_id)

            if class_id:
                if 'dp_masks' in annotation:
                    # Segmentation masks
                    m = self.annToMask(annotation, image_info["height"],
                                       image_info["width"])
                    # Some objects are so small that they're less than 1 pixel area
                    # and end up rounded out. Skip those objects.
                    if m.max() < 1:
                        continue
                    # Is it a crowd? If so, use a negative class ID.
                    if annotation['iscrowd']:
                        # Use negative class ID for crowds
                        class_id *= -1
                        # For crowd masks, annToMask() sometimes returns a mask
                        # smaller than the given dimensions. If so, resize it.
                        if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                            m = np.ones([image_info["height"], image_info["width"]], dtype=bool)

                    print(">>load_uv")
                    print("INdex",idx)
                    
                    # DensePose stuff
                    # dp, dp_mask = self.annToUV(annotation, image_info["height"],
                    #                   image_info["width"])
                    dp = np.zeros((5, 196))
                    # bbr = np.array(annotation['bbox']).astype(int)
                    # i_mask, u_mask, v_mask = self.create_uv_supervision_masks(dp, bbr, m)

                    # instance_i_masks.append(i_mask)
                    # instance_u_masks.append(u_mask)
                    # instance_v_masks.append(v_mask)
                else:
                    dp = np.zeros((5, 196))
                # instance_bboxs.append(bbr)
                class_ids.append(class_id)
                instance_dps.append(dp)
                #instance_dp_masks.append(dp_mask)
        '''

        # Pack instance masks into an array



    def create_uv_supervision_masks(self, dp, bbox, mask):
        dp_xy = dp[0:2, :]
        dp_i = dp[2, :]

        c_bbox = np.round(bbox)

        i_mask = np.zeros(mask.shape, dtype=np.int32)
        u_mask = np.zeros(mask.shape, dtype=np.float32)
        v_mask = np.zeros(mask.shape, dtype=np.float32)
        i_mask.fill(-1)
        u_mask.fill(-1)
        v_mask.fill(-1)

        indeces = np.where(mask == 1)
        i_mask[indeces] = -2

        # Transform densepose xy to bbox
        Point_x = dp_xy[0, :] / 255. * c_bbox[2]
        Point_y = dp_xy[1, :] / 255. * c_bbox[3]
        x1, y1, x2, y2 = c_bbox[0], c_bbox[1], c_bbox[0] + c_bbox[2], c_bbox[1] + c_bbox[3]
        Point_x = np.round(Point_x + x1 + 1).astype(dtype=np.int32)
        Point_y = np.round(Point_y + y1 + 1).astype(dtype=np.int32)


        for i in range(dp_i.shape[0]):
            if dp_i[i] != -1:
                i_mask[Point_y[i], Point_x[i]] = dp_i[i]
            else:
                break


        return i_mask, u_mask, v_mask

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(DenseposeDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToRLEForUV(self, ann, height, width):
        dp_masks = ann['dp_masks']

        dp_masks_rle = []

        for dp_mask in dp_masks:
            # uncompressed RLE
            dp_mask_rle = maskUtils.frPyObjects(dp_mask['counts'], height, width)
            dp_masks_rle.append(dp_mask_rle)

        return dp_masks_rle

    def GetDensePoseMask(self, Polys):
        MaskGen = np.zeros([256, 256])
        for i in range(1, 15):
            if (Polys[i - 1]):
                current_mask = maskUtils.decode(Polys[i - 1])
                MaskGen[current_mask > 0] = i
        return MaskGen

    def annToUV(self, ann, height, width):
        # dp_masks_rles = self.annToRLEForUV(ann, height, width)
        #
        # dp_masks = []
        # for rle in dp_masks_rles:
        #     dp_masks.append(maskUtils.decode(rle))
        scaling_factor_for_pooling = 0.21875 # Hardcoded to output mask
        dp_mask = self.GetDensePoseMask(ann['dp_masks'])
        dp_I = np.array(ann['dp_I'])
        dp_U = np.array(ann['dp_U'])
        dp_V = np.array(ann['dp_V'])
        # Scale xy coords to output mask of network for loss pooling
        dp_x = np.array(ann['dp_x'])
        dp_x = np.multiply(dp_x, scaling_factor_for_pooling)
        dp_y = np.array(ann['dp_y'])
        dp_y = np.multiply(dp_y, scaling_factor_for_pooling)

        dp = np.stack([dp_x, dp_y, dp_I, dp_U, dp_V], axis=0)
        #print("Found {} ground truth points for this instance".format(dp.shape[1]))
        rest = 196 - dp.shape[1]
        n = np.zeros([5, rest])
        n.fill(-1)
        dp = np.concatenate((dp, n), axis=1)
        return dp, dp_mask

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def train(model):
    """Train the model."""
    # Training dataset.

    dataset_train = DenseposeDataset()
    dataset_train.load_densepose_coco(args.dataset, "train")
    dataset_train.prepare()


    # Validation dataset
    dataset_val = DenseposeDataset()
    dataset_val.load_densepose_coco(args.dataset, "valminusminival")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training all layers")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='all')


# def detect_and_color_splash(model, image_path=None, video_path=None):
#     assert image_path or video_path
#
#     # Image or video?
#     if image_path:
#         # Run model detection and generate the color splash effect
#         print("Running on {}".format(args.image))
#         # Read image
#         image = skimage.io.imread(args.image)
#         # Detect objects
#         r = model.detect([image], verbose=1)[0]
#         # Color splash
#         splash = color_splash(image, r['masks'])
#         # Save output
#         file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#         skimage.io.imsave(file_name, splash)
#     elif video_path:
#         import cv2
#         # Video capture
#         vcapture = cv2.VideoCapture(video_path)
#         width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = vcapture.get(cv2.CAP_PROP_FPS)
#
#         # Define codec and create video writer
#         file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
#         vwriter = cv2.VideoWriter(file_name,
#                                   cv2.VideoWriter_fourcc(*'MJPG'),
#                                   fps, (width, height))
#
#         count = 0
#         success = True
#         while success:
#             print("frame: ", count)
#             # Read next image
#             success, image = vcapture.read()
#             if success:
#                 # OpenCV returns images as BGR, convert to RGB
#                 image = image[..., ::-1]
#                 # Detect objects
#                 r = model.detect([image], verbose=0)[0]
#                 # Color splash
#                 splash = color_splash(image, r['masks'])
#                 # RGB -> BGR to save image to video
#                 splash = splash[..., ::-1]
#                 # Add image to video writer
#                 vwriter.write(splash)
#                 count += 1
#         vwriter.release()
#     print("Saved to ", file_name)


############################################################
#  Training
############################################################
def main(_args):
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect densepose.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    global args
    global config
    args = parser.parse_args(_args)


    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DenseposeConfig()
    else:
        class InferenceConfig(DenseposeConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "mrcnn_c_i_1x1", "mrcnn_r_u_1x1", "mrcnn_r_v_1x1"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    # elif args.command == "test":
    # TODO: Test model
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


if __name__ == '__main__':
    main(sys.argv[1:])
