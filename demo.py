import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np




def _stage_loss(y_true, y_pred):
    resized_y_true = tf.tile(y_true, [1,1,1,6])
    resized_y_pred = y_pred
    loss = tf.keras.losses.mean_squared_error(resized_y_true, resized_y_pred)
    return loss

model = keras.models.load_model('saved_model_openpose_ears_ver7', compile=False)
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=_stage_loss)

pred = tf.keras.backend.function([model.input], [model.get_layer('s6').output])
map_size = 100#model.get_layer('feature_map').output_shape[1]

input_size = 200
landmark_size = 55 # 68 # 55

IM_W = 640
IM_H = 480
mg = 20


capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, IM_W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_H)

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv.equalizeHist(frame_gray)
#     faces = face_cascade.detectMultiScale(gray)#, 1.1, 4)
    
#     for (x, y, w, h) in faces:
#         #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         start_x = max(0, x-mg)
#         start_y = max(0, y-mg)
#         end_x   = min(IM_W, x+w+mg)
#         end_y   = min(IM_H, y+h+mg)
#         face = frame[start_y:end_y, start_x:end_x]
#         image = cv2.resize(face, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
#         #image_copy = image[:]
#         #image_pad = cv2.copyMakeBorder(image, 48, 10, 29, 29, cv2.BORDER_CONSTANT, value = [0,0,0])
#         #cv2.imshow("input", image_pad)
#         image = cv2.cvtColor(image_pad, cv2.COLOR_BGR2RGB)
#         result = model.predict(np.expand_dims(image_pad, axis=0)/255.)[0]
        
#         new_w = w + 2*mg
#         new_h = h + 2*mg
#         for x_i, y_i in result.reshape(-1, 2):
#             cv2.circle(frame, (start_x+int(x_i*new_w+new_w), start_y+int(y_i*new_h+new_h)), 1, (0, 255, 0), -1)
            
#         pro = 128/70
#         down = 19/128
#         for x_i, y_i in result.reshape(-1, 2):
#             cv2.circle(image_copy, (int(x_i*pro*35+35), int((y_i-down)*pro*35+35)), 1, (0, 255, 0), -1)
#         cv2.imshow('output', face)
#         cv2.imshow("VideoFrame", image_copy)
    
#     image_cut = frame[:,80:560,:]
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
#     result = model.predict(np.expand_dims(image, axis=0)/255.)[0]
    result = pred([np.expand_dims(image, axis=0)/255.])[0]
    result[result < 0.5] = 0 # threshold setting
    result = np.argmax(result.reshape(-1,landmark_size), axis=0)
    #print(result, result.shape)
    prev_xy = []
    for idx in result:
        x, y = idx%map_size/map_size, idx//map_size/map_size
        x *= IM_W
        y *= IM_H
        if x < 1 or y < 1 : continue
        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
        prev_xy.append([int(x),int(y)])
    #print(prev_xy)
    if len(prev_xy) == landmark_size :cv2.polylines(frame, [np.asarray(prev_xy)], True , (0, 255, 0), 1)
    
#     for x, y in result.reshape(-1, 2):
#         cv2.circle(image, (int(x*64+64), int(y*64+64)), 2, (0, 255, 0), -1)
#         cv2.circle(frame, (80+int(x*240+240), int(y*240+240)), 2, (0, 255, 0), -1)
#     cv2.imshow("VideoFrame", image)
    cv2.imshow("VFrame", frame)
    

capture.release()
cv2.destroyAllWindows()