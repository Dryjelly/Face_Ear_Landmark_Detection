import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib.image as im

import numpy as np
import glob
import os
import sys
import cv2
import glob

from model_config.model import *
from scipy import io

data_dir = 'data/AFW/'
landmark_dir = 'data/AFW_landmark/'

file_list = glob.glob(data_dir+'*.jpg')
file_list = map(os.path.basename, file_list)
# file_list = os.listdir(data_dir)
name_list = list(set(map(lambda x : x[:-4], file_list)))

INPUT_SIZE = 368
MAP_SIZE = 100
MAP_SIGMA = 2.5

LANDMARK_NUM = list(range(68))#[4,9,16,36] # 원하는 point 를 입력
LANDMARK_SIZE = len(LANDMARK_NUM) # ear : 55 / face : 68
LEARNING_RATE = 0.0001
BATCH_SIZE = 10
EPOCH = 100

split_rate = 0.9


def _stage_loss(y_true, y_pred):
    stage = 6
    #y_ture = tf.image.resize(y_true, size=[feat_size, feat_size])
    #threshold = 0.0001
    mask = y_true != 0 #> threshold
    resized_mask   = tf.tile(mask, [1,1,1,stage])
    resized_y_true = tf.tile(y_true, [1,1,1,stage])
    resized_y_pred = y_pred
    #resized_y_pred = tf.image.resize(y_pred, size=[INPUT_SIZE, INPUT_SIZE])
    resized_y_pred = tf.image.resize(y_pred, size=[MAP_SIZE, MAP_SIZE])
    
    loss = tf.math.reduce_mean(tf.math.square(resized_y_true - resized_y_pred) * tf.cast(resized_mask, tf.float32), axis=-1)
    #loss = tf.math.reduce_mean(tf.math.square(resized_y_true - resized_y_pred), axis=-1)
    
    #loss = tf.keras.losses.mean_squared_error(resized_y_true, resized_y_pred)
    return loss

# model = tf.keras.Model(inputs=[x], outputs=[output])
model = model_openpose_a2a_v2(LANDMARK_SIZE=LANDMARK_SIZE)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=_stage_loss)

# model = keras.models.load_model('saved_model_openpose_ears_ver4', compile=False)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=_stage_loss)
model.summary()

input_size_h = INPUT_SIZE/2
feat_size = model.get_layer('feature_map').output_shape[1]

#--------------------------------------

train_len = int(len(name_list)*split_rate)
test_len = len(name_list)-train_len
print(train_len, test_len)

train_dataset = tf.data.Dataset.from_tensor_slices(name_list[:train_len])
test_dataset = tf.data.Dataset.from_tensor_slices(name_list[train_len:])

def process_path(name):
    image_path = data_dir+name+'.jpg'
    #image_path = data_dir+name+'.png'
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    #image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.cond(tf.shape(image)[-1] != 3,
                    lambda: tf.image.grayscale_to_rgb(image),
                    lambda: tf.identity(image))
    image_shape = tf.shape(image)
    
    label = tf.numpy_function(_read_mat, [landmark_dir+name+'_pts.mat', image_shape], tf.float32)
    #label = tf.numpy_function(_read_txt, [landmark_dir+name+'.txt', image_shape], tf.float32)
    
    #image, label = tf.numpy_function(_shrink_image_one, [image, label], [tf.float32, tf.float32])
    #image, label = tf.numpy_function(_crop_image_one, [image, label], [tf.float32, tf.float32])  
    # image, label = tf.numpy_function(_downsize_image, [image, label], [tf.float32, tf.float32]) 
    image, label = tf.numpy_function(_shift_image_one, [image, label], [tf.float32, tf.float32])
    image, label = tf.py_function(_flip_image_one, [image, label], [tf.float32, tf.float32])
    
    label = tf.numpy_function(_make_confidence_map, [label, MAP_SIGMA], tf.float32)
    
    image.set_shape([None, None, None])
    image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
    
    #image.set_shape([INPUT_SIZE, INPUT_SIZE, 3])
    label.set_shape([MAP_SIZE, MAP_SIZE, LANDMARK_SIZE])
    #image = tf.reshape(image, shape=[input_size, input_size, 3])
    #label = tf.reshape(label, shape=[-1])
    
    return image, label

def _read_mat(file_path, input_shape):
    matfile = io.loadmat(file_path)
    norm = [input_shape[1]/2, input_shape[0]/2]
    label = ((matfile['pts_2d']-norm)/norm).astype(np.float32)
    return label

def _read_txt(file_path, input_shape):
    with open(file_path, 'r') as f:
        lines_list = f.readlines()
        temp = list(map(lambda l : list(map(float, l.split(' '))), lines_list[3:-1]))
        temp = np.array(temp)
    norm = [input_shape[1]/2, input_shape[0]/2]
    label = ((temp-norm)/norm).astype(np.float32)
    return label[LANDMARK_NUM]

def _make_confidence_map(label, sigma = 2.5):
    
    norm = [MAP_SIZE/2, MAP_SIZE/2]
    new_label = label*norm+norm
    
    grid_x = np.tile(np.arange(MAP_SIZE), (MAP_SIZE, 1))
    grid_y = np.tile(np.arange(MAP_SIZE), (MAP_SIZE, 1)).transpose()
    grid_x = np.tile(np.expand_dims(grid_x, axis=-1),LANDMARK_SIZE)
    grid_y = np.tile(np.expand_dims(grid_y, axis=-1),LANDMARK_SIZE)
    
    grid_distance = (grid_x - new_label[:,0]) ** 2 + (grid_y - new_label[:,1]) ** 2
    confidence_map = np.exp(-1 * grid_distance / sigma ** 2) # why 0.5?
    
    return confidence_map.astype(np.float32)

def _crop_image_one(img, label): # with label norm
    pad = 1
    
    img_h, img_w, img_c = img.shape
    
    idx = np.array([img_w/2, img_h/2])
    
    label = label*idx+idx
    
    label_x_info = np.array([min(label[:,0]), max(label[:,0])])
    label_y_info = np.array([min(label[:,1]), max(label[:,1])])
    
    ear_w = label_x_info[1]-label_x_info[0]
    ear_h = label_y_info[1]-label_y_info[0]
    
#     s_x = max(int(label_x_info[0]-ear_w*1-pad), 0)
#     e_x = min(int(label_x_info[0]+ear_w*8), img_w)
#     s_y = max(int(label_y_info[0]-ear_h*2-pad), 0)
#     e_y = min(int(label_y_info[0]+ear_h*3), img_h)

    s_x = max(int(label_x_info[0]-ear_w*10), 0)
    e_x = min(int(label_x_info[0]+ear_w*10), img_w)
    s_y = max(int(label_y_info[0]-ear_h*10), 0)
    e_y = min(int(label_y_info[0]+ear_h*10), img_h)

#     s_x = max(int(label_x_info[0]-ear_w*1-pad), 0)
#     e_x = min(int(label_x_info[1]+ear_w*1), img_w)
#     s_y = max(int(label_y_info[0]-ear_h*1-pad), 0)
#     e_y = min(int(label_y_info[1]+ear_h*1), img_h)
    
#     s_x = max(int(label_x_info[0]-pad), 0)
#     e_x = min(int(label_x_info[1]+pad), img_w)
#     s_y = max(int(label_y_info[0]-pad), 0)
#     e_y = min(int(label_y_info[1]+pad), img_h)

    c_img = img[s_y:e_y, s_x:e_x, :]
    c_label = label - np.array([s_x, s_y])
    
    new_img_h, new_img_w, _ = c_img.shape
    
    norm = [new_img_w/2, new_img_h/2]
    c_label = ((c_label-norm)/norm).astype(np.float32)
    return c_img, c_label

def _shrink_image_one(img, label):
    img_h, img_w, img_c = img.shape
    idx = np.array([img_w/2, img_h/2])
    label = label*idx+idx
    
    max_ratio = 4
    sh_ratio = np.random.randint(1,max_ratio)
    
    min_x = int(np.min(label[:,0]))
    max_x = int(np.max(label[:,0]))

    image_left = img[:,0:min_x,:]
    image_right = img[:,max_x:-1,:]
    image_mid = img[:,min_x:max_x:sh_ratio,:]

    sh_img = np.concatenate((image_left, image_mid, image_right), axis = 1)
    sh_label = (label-np.array([min_x,0]))/np.array([sh_ratio,1])+np.array([min_x,0])
    
    new_img_h, new_img_w, _ = sh_img.shape
    norm = [new_img_w/2, new_img_h/2]
    sh_label = ((sh_label-norm)/norm).astype(np.float32)
    
    return sh_img, sh_label
    
def _flip_image_one(img, label):
    c = np.random.randint(2)
    f_img, f_label = tf.cond(c==1,
                             lambda: (tf.image.flip_left_right(img), label * np.array([-1, 1])),
                             lambda: (img, label))
    #f_img = tf.image.resize(f_img, [input_size, input_size])
    return f_img, f_label

def _shift_image_one(img, label, padding = b'zero'):
    img_h, img_w, img_c = img.shape
    label_p, _ = label.shape
    
    if padding == b'ori': s_img = img[:]
    elif padding == b'zero': s_img = np.zeros_like(img, dtype=np.float32)
    s_label = np.expand_dims(label, axis=0)
    
    label_x_info = np.array([min(label[:,0]), max(label[:,0])])* img_w/2 + img_w/2
    label_y_info = np.array([min(label[:,1]), max(label[:,1])])* img_h/2 + img_h/2
    
    label_x_info = label_x_info.astype(np.int)
    label_y_info = label_y_info.astype(np.int)
    
    shift_x = np.random.randint(-label_x_info[0], img_w - label_x_info[1])
    shift_y = np.random.randint(-label_y_info[0], img_h - label_y_info[1])
    
    shift_x = min(shift_x, img_w//6)
    shift_y = min(shift_y, img_h//6)
    
    if shift_x < 0:
        get_x = (-shift_x, img_w)
        put_x = (0, img_w + shift_x)
    else:
        get_x = (0, img_w - shift_x)
        put_x = (shift_x, img_w)
    if shift_y < 0:
        get_y = (-shift_y, img_h)
        put_y = (0, img_h + shift_y)
    else:
        get_y = (0, img_h - shift_y)
        put_y = (shift_y, img_h)

    if padding == b'edge': s_img = np.pad(img[get_y[0]:get_y[1], get_x[0]:get_x[1], :], ((img_h-get_y[1],get_y[0]),(img_w-get_x[1],get_x[0]),(0,0)), mode='edge')
    else: s_img[put_y[0]:put_y[1], put_x[0]:put_x[1], :] = img[get_y[0]:get_y[1], get_x[0]:get_x[1], :]
    s_label = np.append(np.expand_dims(label[:,0] + (shift_x)/(img_w/2), axis = -1),
                          np.expand_dims(label[:,1] + (shift_y)/(img_h/2), axis = -1), axis = 1)
    
    return s_img, s_label

def _downsize_image(img, label):
    img_h, img_w, img_c = img.shape

    size = 1 + np.random.randint(101)/100 # 1~2 (100 step)
    resized_image = cv2.resize(img ,(int(img_w/size), int(img_h/size)))
    #resized_image = tf.image.resize(image ,[int(img_w/size), int(img_h/size)])

    new_img_h, new_img_w, _ = resized_image.shape

    str = np.array([int(img_w/2 * (1 - 1/size)), int(img_h/2 * (1 - 1/size))])
    dst = str + np.array([new_img_w, new_img_h])

    idx = np.array([new_img_w/2, new_img_h/2])
    new_label = (label*idx)+idx + str

    pad_image = np.zeros_like(img, dtype=np.float32)
    pad_image[str[1]:dst[1],str[0]:dst[0],:] = resized_image

    norm = [img_w/2, img_h/2]
    new_label = ((new_label-norm)/norm).astype(np.float32)

    return pad_image, new_label

AUTOTUNE = tf.data.experimental.AUTOTUNE


train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

test_dataset = test_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.repeat()
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


train_step = train_len//BATCH_SIZE
test_step = test_len//BATCH_SIZE

history = model.fit(train_dataset,
                    epochs=EPOCH,
                    steps_per_epoch=train_step,
                    validation_steps=test_step,
                    validation_data=test_dataset,
                    verbose=2)

model.save('saved_model/saved_model_openpose_face_v1.h5')
