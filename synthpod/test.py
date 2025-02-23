from keras.models import load_model
import keras
import synthpod.utils as util
import numpy as np
import cv2
import synthpod.model as modellib

#model = modellib.UNET(nClasses=15,
#              input_height=256,
#              input_width=256)
# model = modellib.FCN8(nClasses=15,
#              input_height=256,
#              input_width=256)
model = modellib.ResNet(nClasses=15,
             input_height=256,
             input_width=256)
#model.load_weights('UNET_64_75_Accuracy/mask_rcnn_dp_0057.h5')

#model.load_weights('FCN8_60E/mask_rcnn_dp_0053.h5')
#model.load_weights('dp20190116T1330/mask_rcnn_dp_0009.h5')

#model.load_weights('dp20190118T2029/mask_rcnn_dp_0028.h5')
#model.load_weights('dp20190126T1538/mask_rcnn_dp_0193.h5')
model.load_weights('ResNet_60E/mask_rcnn_dp_0028.h5')

img2 = cv2.imread("../samples/densepose/test_person3.png")
img2 = cv2.resize(img2, (256, 256))

img, ann_m, ann_i, ann_u, ann_v = util.load_test_img(3)#9

#img2 = np.concatenate([img2, img[:, :, 2:3]], 2)
#img = img2
res_img = cv2.resize(img[:, :, 0:3], (1024, 1024))


# cv2.imshow('image', cv2.resize(util.give_color_to_seg_img(ann, 15), (256, 256), interpolation=0))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#img = np.float32(img) / 127.5 - 1

img = np.expand_dims(img, 0)
img2 = np.expand_dims(img2, 0)
print("Image with shape: ", img.shape)
ann_m = np.expand_dims(ann_m, 0)
ann_i = np.expand_dims(ann_i, 0)
ann_u = np.expand_dims(ann_u, 0)
ann_v = np.expand_dims(ann_v, 0)
#m, i, u, v, m_loss, i_loss, u_loss, v_loss = model.predict([img, ann_m, ann_i, ann_u, ann_v])
res = model.predict([img])

res_argmax = np.argmax(res, 3)
res_argmax = np.squeeze(res_argmax)
#res_argmax = res_argmax.astype(np.float32)
res_argmax = util.give_color_to_seg_img(res_argmax, 15)
res_argmax = cv2.resize(res_argmax, (256, 256), interpolation=0)
numpy_horizontal = np.hstack((cv2.resize(util.give_color_to_seg_img(np.argmax(np.squeeze(ann_m), 2), 15), (256, 256), interpolation=0),
                              np.squeeze(img[:, :, :, 0:3], 0), res_argmax))
img = np.squeeze(img)
img = img.astype(np.float32)
img = cv2.resize(img, (1024, 1024), interpolation=1)

cv2.imshow('detection', numpy_horizontal)
cv2.imwrite("det.jpg", numpy_horizontal.astype(np.uint16))
cv2.waitKey(0)
cv2.destroyAllWindows()