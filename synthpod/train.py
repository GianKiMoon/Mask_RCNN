import datetime

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras import callbacks
import utils as util
import model as modellib
warnings.filterwarnings("ignore")


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
#config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))

print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))

dataset_path = "../datasets/coco"

model = modellib.FCN8(nClasses=15,
             input_height=256,
             input_width=256)
model.summary()


optimizer = optimizers.SGD(lr=0.01, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

#X_train, y_train, X_test, y_test = util.load_data_and_split(dataset_path, subset="train", samples=2)

if 1:
    coco, image_dir = util.load_densepose_coco(dataset_path, subset="train")
    image_ids = list(coco.imgs.keys())

    split = 0.8
    size = image_ids.__len__()
    train_image_ids = image_ids[0:int(size * split)]
    print("# train images: ", train_image_ids.__len__())
    test_image_ids = image_ids[int(size * split):-1]
    print("# test images: ", test_image_ids.__len__())

    params = {'dim': (64, 64),
              'batch_size': 16,
              'n_classes': 15,
              'n_channels': 3,
              'shuffle': True}

    training_generator = util.DataGenerator(train_image_ids, coco, image_dir, **params)
    validation_generator = util.DataGenerator(test_image_ids, coco, image_dir, **params)

# Directory for training logs
log_dir = os.path.join("./", "{}{:%Y%m%dT%H%M}".format(
    "dp", datetime.datetime.now()))

# Path to save after each epoch. Include placeholders that get filled by Keras.
checkpoint_path = os.path.join(log_dir, "mask_rcnn_{}_*epoch*.h5".format(
    "dp"))
checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

callbacks = [
            callbacks.TensorBoard(log_dir=log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            callbacks.ModelCheckpoint(checkpoint_path,
                                            verbose=0, save_weights_only=True, save_best_only=True, mode='max'),
        ]

model.fit_generator(generator=training_generator,
                   validation_data=validation_generator,
                   callbacks=callbacks,
                   use_multiprocessing=False,
                   workers=1,
                   epochs=200)
'''
hist1 = model.fit(X_train,y_train,
                   validation_data=(X_test,y_test),
                   batch_size=1,epochs=200,verbose=1, callbacks=callbacks)
'''