import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from random import shuffle
from tqdm import tqdm
import glob
from PIL import Image
import matplotlib.pyplot as plt

def read_images(directory, resize_to=(50,50)):
    images = []
    labels = []
    
    for f in tqdm(os.listdir(directory)):
        path = os.path.join(directory, f)
        im = Image.open(path).convert('RGB')
        im = im.resize(resize_to)
        im = np.array(im)/255.0
        im = im.astype('float32')
        images.append(im)
        
        if 'rich' in f.lower():
            label = 0
        elif 'law' in f.lower():
            label = 1
        elif 'player' in f.lower():
            label = 2
        elif 'president' in f.lower():
            label = 3
        else:
            label = 4
        
        #label = 0 if 'act' in f.lower()
        #label = 1 if 'law' in f.lower()
        #label = 2 if 'players' in f.lower()
        #label = 3 if 'president' in f.lower()
        #label = 4 if 'rich' in f.lower()
        
        labels.append(label)
        
    num = np.unique(labels, axis = 0)
    num = num.shape[0]
    labels = np.eye(num)[labels]
    return np.array(images), np.array(labels)
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = 5, activation = tf.nn.relu, padding = 'SAME', input_shape = (50, 50, 3)))
    model.add(keras.layers.MaxPool2D(padding = "SAME"))
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 5, activation = tf.nn.relu, padding = 'SAME'))
    model.add(keras.layers.MaxPool2D(padding = "SAME"))
    model.add(keras.layers.Conv2D(filters = 128, kernel_size = 5, activation = tf.nn.relu, padding = 'SAME'))
    model.add(keras.layers.MaxPool2D(padding = "SAME"))
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 5, activation = tf.nn.relu, padding = 'SAME'))
    model.add(keras.layers.MaxPool2D(padding = "SAME"))
    model.add(keras.layers.Conv2D(filters = 32, kernel_size = 5, activation = tf.nn.relu, padding = 'SAME'))
    model.add(keras.layers.MaxPool2D(padding = "SAME"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation = tf.nn.relu))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Dense(5))
    return model
def random_batch(arr_x, arr_y):
    num_images = len(arr_x)
    idx = np.random.choice(num_images, size=batch_size, replace=False)
    x_batch = arr_x[idx]
    y_batch = arr_y[idx]
    
    return x_batch, y_batch
tf.enable_eager_execution()
tf.executing_eagerly()
TRAIN_DIR = './Images/train/'
TEST_DIR = './Images/test/'
LR = 0.001
IM_WIDTH = 50
IM_HEIGHT = 50
X_train, y_train = read_images(directory=TRAIN_DIR, resize_to=(50,50))
X_train.shape
y_train.shape
X_test, y_test = read_images(directory=TEST_DIR , resize_to=(IM_WIDTH,IM_HEIGHT))
X_train.shape
y_train.shape

model = create_model()
model.summary()
optimizer = tf.train.AdamOptimizer(0.00001)
batch_size = 100
total_batch = int(len(X_train)/batch_size)
print('total batch:',total_batch)

for step in range(20):
    total_cost = 0
    total_accuracy = 0

    for i in range(total_batch):
        X, Y = random_batch(X_train, y_train)
        with tf.GradientTape() as tape:
            hypothesis = model(X, training=True)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = hypothesis, labels = Y))
            grads = tape.gradient(cost, model.variables)
        
        optimizer.apply_gradients(zip(grads,model.variables))
        total_cost += cost
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        total_accuracy += accuracy
    print(step)
    print("Loss:{:.4f} accuracy:{:.4f}".format(total_cost/total_batch, total_accuracy/total_batch))

predict = model(X_test, training = False)
predict
predict = tf.argmax(predict, 1)

fig = plt.figure(figsize=(20,70))
for i in range(160):
    subplot = fig.add_subplot(32, 5, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    model_out = predict[i].numpy()
    
    if model_out == 0:
        str_label = "act"
    elif model_out == 1:
        str_label = "law"
    elif model_out == 2 :
        str_label = "players"
    elif model_out == 3 :
        str_label = "president"
    else :
        str_label = "rich"
    
    subplot.set_title('pre:%s' % str_label)
    subplot.imshow(X_test[i].reshape((50, 50, 3)))
    
plt.show()