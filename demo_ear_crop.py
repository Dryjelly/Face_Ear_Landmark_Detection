import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import copy

#-------------------------------------------------- model load ---#
def _stage_loss(y_true, y_pred):
    resized_y_true = tf.tile(y_true, [1,1,1,6])
    resized_y_pred = y_pred
    loss = tf.keras.losses.mean_squared_error(resized_y_true, resized_y_pred)
    return loss

model = keras.models.load_model('saved_model_openpose_ears_ver7', compile=False)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=_stage_loss)

pred = tf.keras.backend.function([model.input], [model.get_layer('s6').output])
map_size = 100 # model.get_layer('feature_map').output_shape[1]*2

input_size = 200
landmark_size = 55 # face: 68 / ear: 55

#-----------------------------------------------------------------#
IM_W = 640
IM_H = 480
mg = 20

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, IM_W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_H)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    #prev_left_ear_loc = []
    #prev_right_ear_loc = []
    
    if len(faces) == 0: 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
        result = pred([np.expand_dims(image, axis=0)/255.
                      ])[0]
        result[result < 0.25] = 0 # threshold setting
        result = np.argmax(result.reshape(-1,landmark_size), axis=0)
        prev_xy = []
        for idx in result:
            x, y = idx%map_size/map_size, idx//map_size/map_size
            x *= IM_W
            y *= IM_H
            if x < 1 or y < 1 : continue
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            prev_xy.append([int(x),int(y)])
        if len(prev_xy) == landmark_size :cv2.polylines(frame, [np.asarray(prev_xy)], True , (0, 255, 0), 1)
        cv2.imshow("VFrame", frame)                                                
        continue
        
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        left_ear  = frame[y:y+h, max(0, x-w//2):x+w//2] # frame 기준
        right_ear = frame[y:y+h, x+w//2:min(IM_W, x+3*w//2)]
        
        left_ear  = cv2.resize(left_ear, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
        right_ear = cv2.resize(right_ear, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
        
        _left_ear  = np.expand_dims(left_ear, axis=0)
        _right_ear = np.expand_dims(right_ear, axis=0)
        
        ear_batch = np.append(_left_ear, _right_ear, axis=0)
        
        result = pred([ear_batch/255.])[0]
       
        result[result < 0.5] = 0 # threshold setting
        result = np.argmax(result.reshape(2,-1,landmark_size), axis=1)

#         x_idx, y_idx = result%map_size/map_size * 200, result//map_size/map_size * 200
        x_idx, y_idx = result%map_size/map_size * w, result//map_size/map_size * h
        
        saved_xy = []
        for x_i, y_i in zip(x_idx[0], y_idx[0]): # left ear
            if x_i < 1 or y_i < 1 : continue
            x_f, y_f = x_i + x-w//2, y_i + y
            cv2.circle(frame, (int(x_f), int(y_f)), 2, (255, 0, 0), -1)
            saved_xy.append([int(x_f),int(y_f)])
        if len(saved_xy) == landmark_size :cv2.polylines(frame, [np.asarray(saved_xy)], True , (0, 255, 0), 1)
            
        saved_xy = []
        for x_i, y_i in zip(x_idx[1], y_idx[1]): # right ear
            if x_i < 1 or y_i < 1 : continue
            x_f, y_f = x_i + x+w//2, y_i + y
            cv2.circle(frame, (int(x_f), int(y_f)), 2, (255, 0, 0), -1)
            saved_xy.append([int(x_f),int(y_f)])
        if len(saved_xy) == landmark_size :cv2.polylines(frame, [np.asarray(saved_xy)], True , (0, 255, 0), 1)
        
        saved_xy = []
        for x_i, y_i in zip(x_idx[0], y_idx[0]): # left ear
            if x_i < 1 or y_i < 1 : continue
            x_f, y_f = x_i + x-w//2, y_i + y
            cv2.circle(left_ear, (int(x_i), int(y_i)), 5, (255, 0, 0), -1)
            saved_xy.append([int(x_i),int(y_i)])
        if len(saved_xy) == landmark_size :cv2.polylines(left_ear, [np.asarray(saved_xy)], True , (0, 255, 0), 3)
        
#         saved_xy = []
#         for x_i, y_i in zip(x_idx[1], y_idx[1]): # right ear
#             if x_i < 1 or y_i < 1 : continue
#             x_f, y_f = x_i + x+w//2, y_i + y
#             cv2.circle(right_ear, (int(x_i), int(y_i)), 5, (255, 0, 0), -1)
#             saved_xy.append([int(x_i),int(y_i)])
#         if len(saved_xy) == landmark_size :cv2.polylines(right_ear, [np.asarray(saved_xy)], True , (0, 255, 0), 3)
            
        cv2.imshow("left", left_ear)
        cv2.imshow("right", right_ear)
        

#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
    
#     result = pred([np.expand_dims(image, axis=0)/255.])[0]
#     result[result < 0.25] = 0 # threshold setting
#     result = np.argmax(result.reshape(-1,landmark_size), axis=0)
    
#     prev_xy = []
#     for idx in result:
#         x, y = idx%map_size/map_size, idx//map_size/map_size
#         x *= IM_W
#         y *= IM_H
#         if x < 1 or y < 1 : continue
#         cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
#         prev_xy.append([int(x),int(y)])
        
#     if len(prev_xy) == landmark_size :cv2.polylines(frame, [np.asarray(prev_xy)], True , (0, 255, 0), 3)
        
        
        
    
#     for x, y in result.reshape(-1, 2):
#         cv2.circle(image, (int(x*64+64), int(y*64+64)), 2, (0, 255, 0), -1)
#         cv2.circle(frame, (80+int(x*240+240), int(y*240+240)), 2, (0, 255, 0), -1)
#     cv2.imshow("VideoFrame", image)
    cv2.imshow("VFrame", frame)
    
    

capture.release()
cv2.destroyAllWindows()