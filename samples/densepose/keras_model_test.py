import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Sequential, load_model,model_from_json



#model = load_model('mask_rcnn_densepose.h5')
model = load_model('mask_rcnn_coco.h5')

'''
result = model.predict([[[5,5]]]) # Das Ergebnis müsste ungefähr bei 10 liegen
'''