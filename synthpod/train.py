import datetime
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras import callbacks
import synthpod.utils as util
import synthpod.model as modellib
from synthpod.config import Config as model_config
import synthpod.loss as loss

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
#config.gpu_options.visible_device_list = "2"
set_session(tf.Session(config=config))
print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))

# Path to coco folder
dataset_path = "../datasets/coco"

# Directory for training logs
log_dir = os.path.join("./", "{}{:%Y%m%dT%H%M}".format("dp", datetime.datetime.now()))

# Path to save after each epoch. Include placeholders that get filled by Keras.
checkpoint_path = os.path.join(log_dir, "mask_rcnn_{}_*epoch*.h5".format("dp"))
checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

# Initialize model
#model = modellib.FCN8(nClasses=15,
#              input_height=256,
#              input_width=256)
#model = modellib.UNET(nClasses=15,
#              input_height=256,
#              input_width=256)
model = modellib.ResNet(nClasses=15,
             input_height=256,
             input_width=256)
model.summary()

# Define optimizer
#optimizer = optimizers.Adam(lr=0.001)#SGD(lr=0.002, decay=5**(-4), momentum=0.9, nesterov=True)
optimizer = optimizers.Adam(lr=model_config.LEARNING_RATE)

# Add losses to model
for name in model_config.LOSS_NAMES:
    layer = model.get_layer(name)
    if layer.output in model.losses:
        continue
    loss = (
        tf.reduce_mean(layer.output, keepdims=True)
        * model_config.LOSS_WEIGHTS.get(name, 1.))
    model.add_loss(loss)

# Add L2 Regularization
# Skip gamma and beta weights of batch normalization layers.
reg_losses = [regularizers.l2(model_config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
    for w in model.trainable_weights
    if 'gamma' not in w.name and 'beta' not in w.name]
model.add_loss(tf.add_n(reg_losses))

# Compile the model
model.compile(loss=[None] * len(model.outputs), loss_weights=model_config.LOSS_WEIGHTS,
              optimizer=optimizer,
              metrics=['accuracy'])

# Add metrics for losses
for name in model_config.LOSS_NAMES:
    if name in model.metrics_names:
        continue
    layer = model.get_layer(name)
    model.metrics_names.append(name)
    loss = (
            tf.reduce_mean(layer.output, keepdims=True)
            * model_config.LOSS_WEIGHTS.get(name, 1.))
    model.metrics_tensors.append(loss)

# Save model file for inspection
model.save("resnet.h5")

# Load coco instance ids
coco_train, train_image_dir = util.load_densepose_coco(dataset_path, subset="train")
coco_val, val_image_dir = util.load_densepose_coco(dataset_path, subset="valminusminival")
train_image_ids = list(coco_train.imgs.keys())
val_image_ids = list(coco_val.imgs.keys())
train_instance_ids = util.get_instance_ids(coco_train, train_image_ids)
val_instance_ids = util.get_instance_ids(coco_val, val_image_ids)
print("Using {} train instances and {} val instances.".format(train_instance_ids.__len__(), val_instance_ids.__len__()))

# Define data generator parameters
params = {'dim': (64, 64),
          'batch_size': 16,
          'n_classes': 15,
          'n_channels': 4,
          'shuffle': True}

# Declare generators
training_generator = util.DataGenerator(train_instance_ids, coco_train, train_image_dir, **params)
validation_generator = util.DataGenerator(val_instance_ids, coco_val, val_image_dir, **params)

# Add callbacks for TensorBoard and ModelCheckpoint
if 1:
    callbacks = [
                callbacks.TensorBoard(log_dir=log_dir,
                                      histogram_freq=0, write_graph=True, write_images=False),
                callbacks.ModelCheckpoint(checkpoint_path,
                                          verbose=0, save_weights_only=True, save_best_only=True,
                                          monitor='val_loss', mode='min'),
        ]

# Fit model to generators
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

#X_train, y_train, X_test, y_test = util.load_data_and_split(dataset_path, subset="train", samples=2)

#split = 0.8
#size = image_ids.__len__()

#train_image_ids = image_ids[0:int(size * split)]
#val_image_ids = image_ids[int(size * split):-1]
