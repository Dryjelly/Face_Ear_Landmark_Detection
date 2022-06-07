import tensorflow as tf
import numpy as np
import dlib
import time
from datetime import datetime
from scipy.ndimage.filters import gaussian_filter

import cv2

def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def apply_blur(img, landmark):
    blur = _gaussian_kernel(5, 2.5, landmark, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1,1,1,1], 'SAME')
    return img[0]

recording = False
recording_time = 20

# model = tf.keras.models.load_model('saved_model_openpose_ears_ver6', compile=False)
# model_face = tf.keras.models.load_model('saved_model_openpose_face', compile=False)

model = tf.keras.models.load_model('saved_model_openpose_ears_ver6.h5', compile=False)
model_face = tf.keras.models.load_model('saved_model_openpose_face.h5', compile=False)

pred = tf.keras.backend.function([model.input], [model.get_layer('s6').output])
pred_face = tf.keras.backend.function([model_face.input], [model_face.get_layer('s6').output])
map_size = 100

input_size = 200
landmark_size = 55

intput_size_face = 200
landmark_size_face = 68

IM_W = 640
IM_H = 480
mg = 20

face_detect_rate = 5
count = 0

if recording:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(datetime.today().strftime("%Y%m%d%H%M%S")+'.avi', fourcc, 25.0, (400,400))

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, IM_W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_H)


start = time.time()  # 시작 시간 저장

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    
    frame = cv2.resize(frame, (400,400), interpolation = cv2.INTER_AREA)
    frame = cv2.flip(frame, 1)
    
    # ear-detect
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
    result = pred([np.expand_dims(image, axis=0)/255.])[0]
    result[result < 0.5] = 0 # threshold setting
    result = tf.image.resize(result, [200, 200])#.numpy()
    # result = gaussian_filter(result, sigma=2.5)
    result = apply_blur(result, 55).numpy()

    result = np.argmax(result.reshape(-1,landmark_size), axis=0)
    prev_xy = []
    for idx in result:
        # x, y = idx%map_size/map_size, idx//map_size/map_size
        x, y = idx%200/200, idx//200/200
        #x, y = idx%400, idx//400
        x *= 400
        y *= 400
        if x < 1 or y < 1 : continue
        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
        prev_xy.append([int(x),int(y)])
    if len(prev_xy) == landmark_size :cv2.polylines(frame, [np.asarray(prev_xy)], True , (0, 255, 0), 1)

    # face-detect
    result = pred_face([np.expand_dims(image, axis=0)/255.])[0]
    #result[result < 0.5] = 0 # threshold setting
    result = tf.image.resize(result, [200, 200], method=tf.image.ResizeMethod.BICUBIC)
    #result = gaussian_filter(result, sigma=2.5)
    result = apply_blur(result, 68).numpy()
    result = np.argmax(result.reshape(-1,landmark_size_face), axis=0)
    # for i, idx in enumerate(result):
    #     x, y = idx%map_size/map_size, idx//map_size/map_size
    #     x *= 400
    #     y *= 400
    for i, idx in enumerate(result):
        x, y = idx%200/200*400, idx//200/200*400
        if x < 1 or y < 1 : continue
        if i in [0, 1, 2, 14, 15, 16]: cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        else: cv2.circle(frame, (int(x), int(y)), 2, 
        (0, 0, 255), -1)

    if ret and recording:
        if (time.time() - start) > recording_time: break
        out.write(frame)

    cv2.ellipse(frame, (200,200), (70,100), 0, 0, 360, (0, 255, 0), 1)

    cv2.imshow("VFrame", frame)
    count -= 1
    

capture.release()
if recording: out.release()
cv2.destroyAllWindows()