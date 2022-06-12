
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import glob
import numpy as np
import os
from scipy import io

from model_config.model import *

model_t = tf.keras.models.load_model('saved_model/saved_model_openpose_face_v1.h5', compile=False)

model = model_openpose_a2a_v2(LANDMARK_SIZE=68)
model.set_weights(model_t.get_weights())

model_t = tf.keras.Model(inputs=[model_t.input], outputs=[tf.tile(model_t.get_layer("s6").output, [1,1,1,6])])
model_t = tf.keras.Model(inputs=[model_t.input], outputs=[model_t.get_layer("feature_map").output, 
                                                          model_t.get_layer("stage_1").output,
                                                          model_t.get_layer("conv2d_19").output, 
                                                          model_t.get_layer("conv2d_25").output, 
                                                          model_t.get_layer("conv2d_31").output, 
                                                          model_t.get_layer("conv2d_37").output,]+[model_t.output])
model_t.summary()


feature_map = model.get_layer("feature_map").output
s2_b = model.get_layer("stage_1").output
s3_b = model.get_layer("conv2d_19").output
s4_b = model.get_layer("conv2d_25").output
s5_b = model.get_layer("conv2d_31").output
s6_b = model.get_layer("conv2d_37").output

s1 = tf.keras.layers.Conv2D(55, kernel_size=(1,1), name='s1_e')(feature_map)
s2 = tf.keras.layers.Conv2D(55, kernel_size=(1,1), name='s2_e')(s2_b)
s3 = tf.keras.layers.Conv2D(55, kernel_size=(1,1), name='s3_e')(s3_b)
s4 = tf.keras.layers.Conv2D(55, kernel_size=(1,1), name='s4_e')(s4_b)
s5 = tf.keras.layers.Conv2D(55, kernel_size=(1,1), name='s5_e')(s5_b)
s6 = tf.keras.layers.Conv2D(55, kernel_size=(1,1), name='s6_e')(s6_b)

output = tf.keras.layers.concatenate([s1, s2, s3, s4, s5, s6], axis=-1, name='output_layer_e')

model_ear = tf.keras.Model(inputs=[model.input], outputs=[model.output, output])
model_ear = tf.keras.Model(inputs=[model.input], outputs=[feature_map, 
                                                          s2_b,
                                                          s3_b,
                                                          s4_b,
                                                          s5_b,
                                                          s6_b,]+model_ear.output)



model_ear.summary()



MAP_SIZE = 100
LEARNING_RATE = 0.0001

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


"""https://keras.io/examples/vision/knowledge_distillation/"""

class Distiller(keras.Model):
    def __init__(self, student, teacher, conv_num=0):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.conv_num = conv_num

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        at_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.at_loss_fn = at_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def zscore(self, x_t, x_s):
        mean=tf.math.reduce_mean(x_t, -1, True)
        std=tf.math.reduce_std(x_t, -1, True)
        return (x_t - mean)/std, (x_s - mean)/std

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            if type(student_predictions) != list : student_predictions = [student_predictions]

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions[-1])

            distillation_loss = 0
            for i in range(self.conv_num, len(student_predictions)-1):
                distillation_loss += self.distillation_loss_fn(
                    # tf.nn.softmax(teacher_predictions[i] / self.temperature, axis=1),
                    # tf.nn.softmax(student_predictions[i] / self.temperature, axis=1),
                    teacher_predictions[i],
                    student_predictions[i],
                )

            at_loss = 0
            a2a_loss = 0
            for i in range(self.conv_num):
                # new_teacher_predictions, new_student_predictions = self.zscore(
                #     tf.reshape(teacher_predictions[i] , [-1,teacher_predictions[i].shape[1]*teacher_predictions[i].shape[2]*teacher_predictions[i].shape[3]]),
                #     tf.reshape(student_predictions[i] , [-1,student_predictions[i].shape[1]*student_predictions[i].shape[2]*student_predictions[i].shape[3]])
                # )
                at_loss += self.at_loss_fn(
                    tf.math.l2_normalize(tf.reduce_mean(teacher_predictions[i], axis=-1)),
                    tf.math.l2_normalize(tf.reduce_mean(student_predictions[i], axis=-1)),
                    # tf.nn.softmax(tf.reshape(teacher_predictions[i] / self.temperature, [-1,teacher_predictions[i].shape[1]*teacher_predictions[i].shape[2]*teacher_predictions[i].shape[3]]), axis=1),
                    # tf.nn.softmax(tf.reshape(student_predictions[i] / self.temperature, [-1,student_predictions[i].shape[1]*student_predictions[i].shape[2]*student_predictions[i].shape[3]]), axis=1),
                    # teacher_predictions[i],
                    # student_predictions[i],
                    # new_teacher_predictions,
                    # new_student_predictions,
                )

                p = tf.cast(tf.math.greater(teacher_predictions[i], 0), tf.float32)
                mu_1 = tf.ones_like(teacher_predictions[i]) # * mu
                a2a_loss += tf.reduce_mean(
                    tf.math.multiply(p,     tf.nn.relu(mu_1 - student_predictions[i])) +
                    tf.math.multiply((1-p), tf.nn.relu(mu_1 + student_predictions[i]))
                )

            loss = self.alpha[0] * student_loss + self.alpha[1] * distillation_loss + self.alpha[2] * at_loss + self.alpha[3] * a2a_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions[-1])

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss, "at_loss": at_loss, "a2a_loss": a2a_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        student_predictions = self.student(x, training=False)
        if type(student_predictions) != list : student_predictions = [student_predictions]
        y_prediction = student_predictions[-1]

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


distiller = Distiller(student=model_ear, teacher=model_t, conv_num = 6)

distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=None,#[keras.metrics.MeanSquaredError()],
            student_loss_fn=_stage_loss,
            distillation_loss_fn=keras.losses.MeanSquaredError(),
            at_loss_fn=keras.losses.MeanSquaredError(),
            alpha=[1.0, 1.0, 0.5, 0.0],
            temperature=2,
        )




data_dir_train = 'data/CollectionA/train/'
data_dir_test = 'data/CollectionA/test/'

# file_list = os.listdir(data_dir)
file_list_train = glob.glob(data_dir_train+'*.png')
file_list_test = glob.glob(data_dir_test+'*.png')

name_list_train = list(set(map(lambda x : x[:-4], file_list_train)))
name_list_test = list(set(map(lambda x : x[:-4], file_list_test)))
    

INPUT_SIZE = 368
MAP_SIZE = 100
MAP_SIGMA = 2.5

LANDMARK_NUM = list(range(55))#[4,9,16,36] # 원하는 point 를 입력
LANDMARK_SIZE = len(LANDMARK_NUM) # ear : 55 / face : 68
LEARNING_RATE = 0.0001
BATCH_SIZE = 10
EPOCH = 100

split_rate = 0.9

input_size_h = INPUT_SIZE/2
feat_size = model_ear.get_layer('feature_map').output_shape[1]

#--------------------------------------

train_len = len(name_list_train)
test_len = len(name_list_test)

train_dataset = tf.data.Dataset.from_tensor_slices(name_list_train)
test_dataset = tf.data.Dataset.from_tensor_slices(name_list_test)

def process_path(name):
    #image_path = data_dir+name+'.jpg'
    image_path = name+'.png'
    image = tf.io.read_file(image_path)
    #image = tf.image.decode_jpeg(image)
    image = tf.image.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.cond(tf.shape(image)[-1] != 3,
                    lambda: tf.image.grayscale_to_rgb(image),
                    lambda: tf.identity(image))
    image_shape = tf.shape(image)
    
    #label = tf.numpy_function(_read_txt, [landmark_dir+name+'_pts.mat', image_shape], tf.float32)
    # label = tf.numpy_function(_read_txt, [landmark_dir+name+'.txt', image_shape], tf.float32)
    label = tf.numpy_function(_read_pts, [name+'.pts', image_shape], tf.float32)
    
    image, label = tf.numpy_function(_shrink_image_one, [image, label], [tf.float32, tf.float32])
    image, label = tf.numpy_function(_crop_image_one, [image, label], [tf.float32, tf.float32])  
    #image, label = tf.numpy_function(_shift_image_one, [image, label], [tf.float32, tf.float32])
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

def _read_pts(file_path, input_shape):
    ptsfile = np.loadtxt(str(file_path, 'utf-8'), comments=("version:", "n_points:", "{", "}"))
    norm = [input_shape[1]/2, input_shape[0]/2]
    label = ((ptsfile-norm)/norm).astype(np.float32)
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

history = distiller.fit(train_dataset,
                    epochs=EPOCH,
                    steps_per_epoch=train_step,
                    validation_steps=test_step,
                    validation_data=test_dataset,
                    verbose=2)

model_ear.save('saved_model_openpose_face2ear_v1.h5')