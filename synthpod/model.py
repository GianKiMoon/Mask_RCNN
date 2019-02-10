import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.utils import plot_model
import keras.applications.resnet50 as rn50
import utils as util
import keras.layers as KL
import synthpod.loss as loss
warnings.filterwarnings("ignore")

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

VGG_Weights_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop_new.h5"

def FCN8(nClasses, input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 4))  ## Assume 224,224,3

    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 14, 14, 512)
    pool4 = Dropout(0.5)(pool4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 7, 7, 512)
    pool5 = Dropout(0.5) (pool5)

    vgg = Model(img_input, pool5)
    #vgg.load_weights(VGG_Weights_path)  ## loading VGG weights for the encoder parts of FCN8

    n = 4096
    o = (Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    # 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False, data_format=IMAGE_ORDERING)(
        conv7)
    conv7_4 = (BatchNormalization())(conv7_4)
    # (None, 224, 224, 10)
    # 2 times upsampling for pool411
    pool411 = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (
        Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(
        pool411)
    pool411_2 = (BatchNormalization())(pool411_2)

    pool311 = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
    pool311 = (BatchNormalization())(pool311)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING)(o)  #o)

    o = (Activation('softmax'))(o)
    print("Output shape ", o.shape)
    model = Model(img_input, o)
    #plot_model(model, to_file='model.png')


    return model

def FCN8_Multi(nClasses, input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 4))  ## Assume 224,224,3

    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 14, 14, 512)
    pool4 = Dropout(0.5)(pool4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 7, 7, 512)
    pool5 = Dropout(0.5) (pool5)

    vgg = Model(img_input, pool5)
    #vgg.load_weights(VGG_Weights_path)  ## loading VGG weights for the encoder parts of FCN8

    n = 4096
    o = (Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    # Part Segmentation

    # 4 times upsamping for pool4 layer
    conv7_4_p = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False, data_format=IMAGE_ORDERING)(
        conv7)
    conv7_4_p = (BatchNormalization())(conv7_4_p)
    # (None, 224, 224, 10)
    # 2 times upsampling for pool411
    pool411_p = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2_p = (
        Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(
        pool411_p)
    pool411_2_p = (BatchNormalization())(pool411_2_p)

    pool311_p = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
    pool311_p = (BatchNormalization())(pool311_p)

    p = Add(name="add")([pool411_2_p, pool311_p, conv7_4_p])
    p = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING)(p)

    p = (Activation('softmax'))(p)

    # Part index

    # 4 times upsamping for pool4 layer
    conv7_4_i = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False,
                                data_format=IMAGE_ORDERING)(
        conv7)
    conv7_4_i = (BatchNormalization())(conv7_4_i)
    # (None, 224, 224, 10)
    # 2 times upsampling for pool411
    pool411_i = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2_i = (
        Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(
        pool411_i)
    pool411_2_i = (BatchNormalization())(pool411_2_i)

    pool311_i = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)
    pool311_i = (BatchNormalization())(pool311_i)

    i = Add(name="add")([pool411_2_i, pool311_i, conv7_4_i])
    i = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING)(i)

    i = (Activation('softmax'))(i)


    model = Model(img_input, [p, i, u, v])

    # plot_model(model, to_file='model.png')
    return model

def UNET(nClasses, input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 4))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
    x = Dropout(0.5)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
    f5 = x

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    vgg = Model(img_input, f5)
    #vgg.load_weights(VGG_Weights_path)

    levels = [f1, f2, f3, f4, f5]

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=3))
    o = Dropout(0.5)(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=3))
    o = Dropout(0.5)(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    '''
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f1], axis=3))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    '''
    o = Conv2D(nClasses, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
    o_shape = Model(img_input, o).output_shape
    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    #o = (Reshape((outputHeight * outputWidth, nClasses)))(o)
    #o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    #model.load_weights('dp20190108T1348/mask_rcnn_dp_0142.h5')
    model.outputWidth = outputWidth


    model.outputHeight = outputHeight

    return model

def ResNet(nClasses, input_height=224, input_width=224, mode="train"):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    model = ResNet50(include_top=False, weights=None, input_tensor=Input(shape=(input_height, input_width, 4)),
                     input_shape=(input_height, input_width, 4), pooling=None, classes=nClasses, mode=mode)

    print(model.output_shape)
    return model

def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000, mode="train"):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # Determine proper input shape
    input_shape = rn50._obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    print("img input shape ", img_input.shape)

    if mode == "train":
        pmask_gt = Input(shape=(64, 64, classes), name="pmask_gt")
        i_mask_gt = Input(shape=(64, 64, 25), name="i_mask_gt")
        u_mask_gt = Input(shape=(64, 64, 25), name="u_mask_gt")
        v_mask_gt = Input(shape=(64, 64, 25), name="v_mask_gt")
        #iuv_gt = Input(shape=(5, 196), name="iuv_gt")

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = rn50.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = rn50.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = rn50.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = rn50.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = rn50.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = rn50.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = rn50.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    #x = rn50.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    #x = rn50.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    #x = rn50.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    #x = rn50.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    #x = rn50.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    #x = rn50.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    #x = rn50.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    #x = rn50.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    #x = rn50.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    #x = AveragePooling2D((7, 7), name='avg_pool')(x)

    IMAGE_ORDERING = "channels_last"

    o1 = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(x)
    o1 = Dropout(0.5)(o1)
    o1 = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o1)
    o1 = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o1)
    o1 = (BatchNormalization())(o1)

    m = Conv2D(classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o1)
    m = (Activation('softmax'))(m)

    for a in range(8):
        x = (Conv2D(256, (3, 3), strides=(1, 1), padding="same", data_format=IMAGE_ORDERING))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)

    o2 = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(x)
    o2 = Dropout(0.5)(o2)
    o2 = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o2)
    o2 = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o2)
    o2 = (BatchNormalization())(o2)

    i = Conv2D(25, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o2)
    i = (Activation('softmax'))(i)

    o3 = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(x)
    o3 = Dropout(0.5)(o3)
    o3 = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o3)
    o3 = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o3)
    o3 = (BatchNormalization())(o3)

    u = Conv2D(25, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o3)
    u = (Activation('sigmoid'))(u)

    o4 = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(x)
    o4 = Dropout(0.5)(o4)
    o4 = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o4)
    o4 = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o4)
    o4 = (BatchNormalization())(o4)

    v = Conv2D(25, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o4)
    v = (Activation('sigmoid'))(v)

    inputs = []
    outputs = []

    if mode == "train":
        m_loss = KL.Lambda(lambda rx: loss.m_loss(*rx), name="m_loss")([pmask_gt, m])
        i_loss = KL.Lambda(lambda rx: loss.i_loss(*rx), name="i_loss")([i_mask_gt, i])
        u_loss = KL.Lambda(lambda rx: loss.uv_loss(*rx), name="u_loss")([u_mask_gt, i_mask_gt, u])
        v_loss = KL.Lambda(lambda rx: loss.uv_loss(*rx), name="v_loss")([v_mask_gt, i_mask_gt, v])

        inputs = [img_input, pmask_gt, i_mask_gt, u_mask_gt, v_mask_gt]
        outputs = [m, i, u, v, m_loss, i_loss, u_loss, v_loss]
    elif mode == "inference":
        inputs = [img_input]
        outputs = [m, i, u, v]

    # Create model.
    model = Model(inputs, outputs, name='resnet50_densepose_{}'.format(mode))

    return model
