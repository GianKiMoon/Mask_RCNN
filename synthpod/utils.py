import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import numpy as np
import cv2
import skimage as sk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import keras

def load_densepose_coco(dataset_dir, subset, return_coco=True):
    coco = COCO("{}/annotations/densepose_coco_2014_{}.json".format(dataset_dir, subset))
    if subset == "minival" or subset == "valminusminival":
        subset = "val"
    image_dir = "{}/{}2014".format(dataset_dir, subset)

    if return_coco:
        return coco, image_dir


def get_box_images(img_path=None, anns=None):
    img = np.array(cv2.imread(img_path))
    img_height = np.shape(img)[0]
    img_width = np.shape(img)[1]

    box_imgs = []
    box_anns = []
    for ann in anns:
        box_img = None
        if 'dp_masks'in ann:
            bbox = np.array(ann['bbox'])
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[0] + bbox[2])
            y2 = int(bbox[1] + bbox[3])
            box_img = img[y1:y2, x1:x2, :]
            box_img = cv2.resize(box_img, (256, 256))
            # Center images to interval [-1, 1]
            box_img = np.float32(box_img) / 127.5 - 1
            box_imgs.append(box_img)

            dp_mask = GetDensePoseMask(ann['dp_masks'])
            dp_mask = dp_mask[0::4, 0::4, :]
            box_anns.append(dp_mask)
            #cv2.imshow('image', give_color_to_seg_img(np.argmax(dp_mask, 2), 15))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


    box_imgs = np.stack(box_imgs)
    box_anns = np.stack(box_anns)

    return box_imgs, box_anns


def give_color_to_seg_img(seg, n_classes):
    '''
    seg : (input_width,input_height,3)
    '''

    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return (seg_img)


def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256, 256, 15])
    bg_mask = np.ones([256, 256])
    for i in range(1, 15):
        if (Polys[i - 1]):
            current_mask = maskUtils.decode(Polys[i - 1])
            MaskGen[current_mask > 0, i] = 1
            bg_mask[current_mask > 0] = 0
            #MaskGen[current_mask > 0] = i
    #MaskGen[:, :, 0] = bg_mask
    return MaskGen

def load_test_img(idx = 0):
    coco, image_dir = load_densepose_coco("../datasets/coco", subset="minival")
    image_ids = list(coco.imgs.keys())

    image_id = image_ids[idx]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None))
    imgs, anns = get_box_images(os.path.join(image_dir, coco.imgs[image_id]['file_name']),
                                    anns)

    return imgs[0,], np.argmax(anns[0, ], 2)

def load_data_and_split(dataset_dir, subset="minival", samples=10, split=0.8):

    print("Load data and split")
    coco, image_dir = load_densepose_coco(dataset_dir, subset=subset
                                          )
    image_ids = list(coco.imgs.keys())

    assert (image_ids.__len__() >= samples)

    # Train set
    x = []
    y = []
    for i in range(samples):
        image_id = image_ids[i]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None))
        box_images, box_anns = get_box_images(os.path.join(image_dir, coco.imgs[image_id]['file_name']),
                                    anns)

        if (x.__len__() == 0):
            x = box_images
            y = box_anns
        else:
            x = np.concatenate([x, box_images])
            y = np.concatenate([y, box_anns])

        print("Loaded data: {} %".format((i / samples) * 100))

    x_train = x[0:int(samples * split), :, :, :]
    x_test = x[int(samples * split):samples, :, :, :]
    y_train = y[0:int(samples * split), :, :, :]
    y_test = y[int(samples * split):samples, :, :, :]

    return x_train, y_train, x_test, y_test


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, coco, image_dir, labels=None, batch_size=32, dim=(64,64), n_channels=3,
                 n_classes=15, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.coco = coco
        self.image_dir = image_dir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 256, 256, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image_id = ID
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=None))
            box_images, box_anns = get_box_images(os.path.join(self.image_dir, self.coco.imgs[image_id]['file_name']),
                                                  anns)

            #cv2.imshow('image', box_images[0, :, :, :])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #
            # cv2.imshow('image', give_color_to_seg_img(np.argmax(box_anns[0, :, :, :], 2), 15))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Store sample
            X[i,:,:,:] = box_images[0,:,:,:]

            # Store class
            y[i,:,:,:] = box_anns[0,:,:,:]

        return X, y