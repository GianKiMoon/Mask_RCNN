#
# Modell trainieren
#

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.models import Sequential, load_model,model_from_json
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import metrics
from pprint import pprint
import numpy as np

# Einfaches Addieren 
input_data = np.array([
[	1	,	1	]	,
[	2	,	2	]	,
[	3	,	3	]	,
[	4	,	4	]	,
[	5	,	5	]])

output_data = np.array([[	2	],
[	4	],
[	6	],
[	8	],
[	10	]])

my_model = Sequential()
my_model.add(Dense(1024,input_dim=2,activation="linear"))
my_model.add(Dense(1,activation="linear"))
my_model.summary()

sgd = SGD(lr=0.001)
my_model.compile(loss="mean_squared_error", optimizer=sgd,metrics=[metrics.mae])

def simple_generator():
    i = 0
    limit = 10
    inputs = []
    outputs = []
    print("Bin der Generator!")
    while True:
         for i in range(10):
            i = i + 1
            inputs.append([i,i])
            outputs.append([i])

            yield inputs, outputs
    #if(i==limit):
    #    return


my_model.fit_generator(simple_generator(),initial_epoch=0,epochs=5,steps_per_epoch=100)

# my_model.fit(input_data, output_data, batch_size=1, epochs=100, verbose=1)




