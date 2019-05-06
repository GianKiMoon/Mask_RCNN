import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np
import cv2
import seaborn as sns
import numpy as np
import keras
from random import choices
from scipy.io import loadmat
from scipy.interpolate import griddata # not quite the same as `matplotlib.mlab.griddata`
from keras.utils.np_utils import to_categorical

def load_densepose_coco(dataset_dir, subset, return_coco=True):

    if subset == "synthdense_train":
        coco = COCO("{}/train_dataset/annotations/synthdense_annotations.json".format(dataset_dir))
        image_dir = "{}/train_dataset/train".format(dataset_dir)
    elif subset == "synthdense_val":
        coco = COCO("{}/val_dataset/annotations/synthdense_annotations.json".format(dataset_dir))
        image_dir = "{}/val_dataset/train".format(dataset_dir)
    else:
        coco = COCO("{}/annotations/densepose_coco_2014_{}.json".format(dataset_dir, subset))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}2014".format(dataset_dir, subset)

        image_ids = list(coco.imgs.keys())

    if return_coco:
        return coco, image_dir


def get_dense_pose_mask(polys):
    mask_gen = np.zeros([256, 256, 15])
    bg_mask = np.ones([256, 256])
    seg = np.zeros([256, 256])
    for i in range(1, 15):
        if polys[i - 1]:
            current_mask = maskUtils.decode(polys[i - 1])
            mask_gen[current_mask > 0, i] = 1
            bg_mask[current_mask > 0] = 0
            #MaskGen[current_mask > 0] = i
    mask_gen[:, :, 0] = bg_mask
    seg[bg_mask == 0] = 1
    seg = np.expand_dims(seg, 2)
    return mask_gen, seg


def get_fake_dense_pose_mask(dp_i, seg):
    dp_i = np.argmax(dp_i, 2)
    dp_mask = np.zeros((64, 64, 15))

    dp_mask[dp_i == 1, 1] = 1
    dp_mask[dp_i == 2, 1] = 1
    dp_mask[dp_i == 3, 2] = 1
    dp_mask[dp_i == 4, 3] = 1
    dp_mask[dp_i == 5, 4] = 1
    dp_mask[dp_i == 6, 5] = 1
    dp_mask[dp_i == 7, 6] = 1
    dp_mask[dp_i == 9, 6] = 1
    dp_mask[dp_i == 8, 7] = 1
    dp_mask[dp_i == 10, 7] = 1
    dp_mask[dp_i == 11, 8] = 1
    dp_mask[dp_i == 13, 8] = 1
    dp_mask[dp_i == 12, 9] = 1
    dp_mask[dp_i == 14, 9] = 1
    dp_mask[dp_i == 15, 10] = 1
    dp_mask[dp_i == 17, 10] = 1
    dp_mask[dp_i == 16, 11] = 1
    dp_mask[dp_i == 18, 11] = 1
    dp_mask[dp_i == 19, 12] = 1
    dp_mask[dp_i == 21, 12] = 1
    dp_mask[dp_i == 20, 13] = 1
    dp_mask[dp_i == 22, 13] = 1
    dp_mask[dp_i == 23, 14] = 1
    dp_mask[dp_i == 24, 14] = 1

    seg = np.invert(np.squeeze(seg).astype(np.bool))
    seg = seg.astype(np.uint8)
    dp_mask[:, :, 0] = seg

    return dp_mask.astype(np.uint8)


def get_dense_pose_uv(ann, seg):
    scaling_factor_for_pooling = 0.25  # 0.234375 #30:  0.1171875 # Hardcoded to output mask
    dp_I = np.array(ann['dp_I'])
    dp_U = np.array(ann['dp_U'])
    dp_V = np.array(ann['dp_V'])
    # Scale xy coords to output mask of network for loss pooling
    dp_x = np.array(ann['dp_x'])
    dp_x[dp_x > 255] = 255
    dp_x[dp_x < 0] = 0
    dp_x = np.multiply(dp_x, scaling_factor_for_pooling).astype(np.int32)
    dp_y = np.array(ann['dp_y'])
    dp_y[dp_y > 255] = 255
    dp_y[dp_y < 0] = 0
    dp_y = np.multiply(dp_y, scaling_factor_for_pooling).astype(np.int32)

    dp = np.stack([dp_x, dp_y, dp_I, dp_U, dp_V], axis=0)
    dp_i_grid, dp_u_grid, dp_v_grid = interpolate_uvs(dp, seg)
    # print("Found {} ground truth points for this instance".format(dp.shape[1]))
    rest = 196 - dp.shape[1]
    n = np.zeros([5, rest])
    n.fill(-1)
    dp = np.concatenate((dp, n), axis=1)

    return dp_i_grid, dp_u_grid, dp_v_grid


def interpolate_uvs(dp, seg, map_size=64):

    dp_x = dp[0,:]
    dp_y = dp[1,:]
    dp_i = np.array(dp[2, :]).transpose()
    dp_u = np.array(dp[3, :]).transpose()
    dp_v = np.array(dp[4, :]).transpose()

    xy = np.stack([dp_x, dp_y], 0).transpose().astype(np.int32)
    xx, yy = np.meshgrid(np.arange(0, map_size, 1), np.arange(0, map_size, 1))
    g = np.vstack([xx.ravel(), yy.ravel()]).transpose()

    grid_ii = np.zeros((64, 64, 25))
    try:
        grid_i = griddata(xy, dp_i, g, method='nearest')
        grid_i = np.reshape(grid_i, [map_size, map_size])
        seg = seg[0::4, 0::4, :]
        seg = seg.squeeze()
        grid_i[seg == 0] = 0

        grid_ii = to_categorical(grid_i, num_classes=25)
    except Exception as e:
        grid_ii[:, :, 0] = np.ones((map_size, map_size))
    #     print("DP_I ", dp_i)
    #     print(e)
    #     grid_ii.fill(-1)
    #     for n in range(xy.__len__()):
    #         grid_ii[xy[n, 0], xy[n, 1], dp_i[n]] = 1

    grid_uu = np.zeros((64, 64, 25))
    for j in range(1, 25):
        xy_j = xy[dp_i == j]
        dp_u_j = dp_u[dp_i == j]

        if xy_j.__len__() >= 4:
            try:

                grid_u = griddata(xy_j, dp_u_j, g, method='linear')
                grid_u = np.reshape(grid_u, [map_size, map_size])
                grid_u[np.isnan(grid_u)] = -1
            except Exception as e:
                grid_u = np.zeros((map_size, map_size))
                grid_u.fill(-1)
                for n in range(xy_j.__len__()):
                    grid_u[xy_j[n, 0], xy_j[n, 1]] = dp_u_j[n]

        else:
            grid_u = np.zeros((map_size, map_size))
            grid_u.fill(-1)
            for n in range(xy_j.__len__()):
                grid_u[xy_j[n, 0], xy_j[n, 1]] = dp_u_j[n]

        grid_uu[:, :, j] = grid_u


    grid_vv = np.zeros((64, 64, 25))
    for j in range(1, 25):
        xy_j = xy[dp_i == j]
        dp_v_j = dp_v[dp_i == j]

        if xy_j.__len__() >= 4:
            try:

                grid_v = griddata(xy_j, dp_v_j, g, method='linear')
                grid_v = np.reshape(grid_v, [map_size, map_size])
                grid_v[np.isnan(grid_v)] = -1
            except Exception as e:
                grid_v = np.zeros((map_size, map_size))
                grid_v.fill(-1)
                for n in range(xy_j.__len__()):
                    grid_v[xy_j[n, 0], xy_j[n, 1]] = dp_v_j[n]

        else:
            grid_v = np.zeros((map_size, map_size))
            grid_v.fill(-1)
            for n in range(xy_j.__len__()):
                grid_v[xy_j[n, 0], xy_j[n, 1]] = dp_v_j[n]

        grid_vv[:, :, j] = grid_v


    # grid_v = griddata(xy, dp_v, g, method='linear')
    # grid_v = np.reshape(grid_v, [map_size, map_size])
    # grid_v[np.isnan(grid_v)] = -1
    # grid_v = np.expand_dims(grid_v, axis=2)

    # img = cv2.resize(give_color_to_seg_img(seg, 2), (256, 256), interpolation=0)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # img = cv2.resize(give_color_to_seg_img(outgrid, 25), (256, 256), interpolation=0)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return grid_ii, grid_uu, grid_vv


def get_instance_ids(coco, image_ids):
    image_instance_ids = []
    for i in image_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[i], iscrowd=None))
        a = 0
        for ann in anns:
            if 'dp_masks' in ann and ann['dp_I'] and ann['dp_U'] and ann['dp_V']:
                image_instance_ids.append((i, a))
                a = a + 1

    return image_instance_ids


def get_box_image(instance_idx, img_path=None, anns=None):
    img = np.array(cv2.imread(img_path))
    assert (instance_idx is not None)

    idx = 0
    for ann in anns:
        if 'dp_masks' in ann:
            if idx == instance_idx:
                bbox = np.array(ann['bbox'])
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[0] + bbox[2])
                y2 = int(bbox[1] + bbox[3])
                box_img = img[y1:y2, x1:x2, :]
                box_img = cv2.resize(box_img, (256, 256))

                # cv2.imshow('image', box_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Center images to interval [-1, 1]
                box_img = np.float32(box_img) / 127.5 - 1

                dp_mask = []

                if ann['dp_masks']:
                    dp_mask, seg = get_dense_pose_mask(ann['dp_masks'])
                    dp_mask = dp_mask[0::4, 0::4, :]
                else:
                    rle = annToRLE(ann, img.shape[0], img.shape[1])
                    seg = maskUtils.decode(rle)
                    seg = seg[y1:y2, x1:x2]
                    seg = np.expand_dims(cv2.resize(seg, (256, 256), interpolation=0), 2)


                # cv2.imshow('image', give_color_to_seg_img(seg, 2))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                box_img = np.concatenate([box_img, seg], 2)


                gt_i_mask, gt_u_mask, gt_v_mask = get_dense_pose_uv(ann, seg)

                if not ann['dp_masks']:
                    seg_sub = seg[0::4, 0::4, :]
                    dp_mask = get_fake_dense_pose_mask(gt_i_mask, seg_sub)

                # cv2.imshow('image', cv2.resize(give_color_to_seg_img(np.argmax(dp_mask, 2), 15), (256, 256), interpolation=0))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                return box_img, dp_mask, gt_i_mask, gt_u_mask, gt_v_mask
            idx += 1

    return None


def annToRLE(ann, height, width):
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


def load_test_img(idx=0):
    coco, image_dir = load_densepose_coco("../datasets/coco", subset="minival")
    image_ids = list(coco.imgs.keys())
    image_id = image_ids[idx]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None))
    img, ann_mask, gt_i_mask, gt_u_mask, gt_v_mask = get_box_image(0, os.path.join(image_dir, coco.imgs[image_id]['file_name']),
                                    anns)

    return img, ann_mask, gt_i_mask, gt_u_mask, gt_v_mask


def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return seg_img


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

        ALP_UV = loadmat(os.path.join(os.path.dirname(__file__), 'UV_Processed.mat'))
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]
        ## Info to compute symmetries.
        self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.Index_Symmetry_List = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24,
                                    23];
        UV_symmetry_filename = os.path.join(os.path.dirname(__file__),
                                            'UV_symmetry_transforms.mat')

        self.UV_symmetry_transformations = loadmat(UV_symmetry_filename)

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
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 256, 256, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))
        #y_dp = np.empty((self.batch_size, 5, 196))
        y_i = np.empty((self.batch_size, *self.dim, 25))
        y_u = np.empty((self.batch_size, *self.dim, 25))
        y_v = np.empty((self.batch_size, *self.dim, 25))


        # Generate data
        for i, (ID1, ID2) in enumerate(list_IDs_temp):
            image_id = ID1
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=None))
            box_image, box_ann_mask, box_i_mask, box_u_mask, box_v_mask = \
                get_box_image(ID2, os.path.join(self.image_dir, self.coco.imgs[image_id]['file_name']), anns)

            # Store sample
            X[i,:,:,:] = box_image

            # Store mask
            y[i,:,:,:] = box_ann_mask

            y_i[i, :, :, :] = box_i_mask

            y_u[i, :, :, :] = box_u_mask

            y_v[i, :, :, :] = box_v_mask

        X, y, y_i, y_u, y_v = self.__fliplr(X, y, y_i, y_u, y_v, 0.5)

        inputs = [X, y, y_i, y_u, y_v]
        outputs = []

        return inputs, outputs

    def __fliplr(self, X, y, y_i, y_u, y_v, probability):
        for i in range(X.shape[0]):
            if choices([0, 1], [1-probability, probability]) == 1:
                X[i, :, :, :] = np.fliplr(X[i, :, :, :])
                y[i, :, :, :] = np.fliplr(y[i, :, :, :])
                y_i[i, :, :, :] = np.fliplr(y_i[i, :, :, :])
                y_u[i, :, :, :] = np.fliplr(y_u[i, :, :, :])
                y_v[i, :, :, :] = np.fliplr(y_v[i, :, :, :])

                # GT_x = y_dp[i, 0, :]
                # GT_y = y_dp[i, 1, :]
                # GT_I = y_dp[i, 2, :]
                # GT_U = y_dp[i, 3, :]
                # GT_V = y_dp[i, 4, :]
                # GT_I, GT_U, GT_V, GT_x, GT_y, _ = self.get_symmetric_densepose(GT_I, GT_U, GT_V, GT_x, GT_y, y[i, :, :, :])
                # y_dp[i, 0, :] = GT_x
                # y_dp[i, 1, :] = GT_y
                # y_dp[i, 2, :] = GT_I
                # y_dp[i, 3, :] = GT_U
                # y_dp[i, 4, :] = GT_V

        return X, y, y_i, y_u, y_v

    def get_symmetric_densepose(self, I, U, V, x, y, Mask):
        ### This is a function to get the mirror symmetric UV labels.
        Labels_sym = np.zeros(I.shape)
        U_sym = np.zeros(U.shape)
        V_sym = np.zeros(V.shape)
        ###
        for i in (range(24)):
            if i + 1 in I:
                Labels_sym[I == (i + 1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i + 1))
                ###
                U_loc = (U[jj] * 255).astype(np.int64)
                V_loc = (V[jj] * 255).astype(np.int64)
                ###
                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0, i][V_loc, U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0, i][V_loc, U_loc]
        ##
        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)
        #
        for i in (range(14)):
            Mask_flipped[Mask_flip == (i + 1)] = self.SemanticMaskSymmetries[i + 1]
        #
        [y_max, x_max] = Mask_flip.shape
        y_sym = y
        x_sym = x_max - x
        #

        return Labels_sym, U_sym, V_sym, x_sym, y_sym, Mask_flipped




# def load_data_and_split(dataset_dir, subset="minival", samples=10, split=0.8):
#
#     print("Load data and split")
#     coco, image_dir = load_densepose_coco(dataset_dir, subset=subset
#                                           )
#     image_ids = list(coco.imgs.keys())
#
#     assert (image_ids.__len__() >= samples)
#
#     # Train set
#     x = []
#     y = []
#     for i in range(samples):
#         image_id = image_ids[i]
#         anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None))
#         box_images, box_anns = get_box_images(os.path.join(image_dir, coco.imgs[image_id]['file_name']),
#                                     anns)
#
#         if (x.__len__() == 0):
#             x = box_images
#             y = box_anns
#         else:
#             x = np.concatenate([x, box_images])
#             y = np.concatenate([y, box_anns])
#
#         print("Loaded data: {} %".format((i / samples) * 100))
#
#     x_train = x[0:int(samples * split), :, :, :]
#     x_test = x[int(samples * split):samples, :, :, :]
#     y_train = y[0:int(samples * split), :, :, :]
#     y_test = y[int(samples * split):samples, :, :, :]
#
#     return x_train, y_train, x_test, y_test