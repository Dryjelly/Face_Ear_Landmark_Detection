import tensorflow as tf
import numpy as np
import dlib
import time
from datetime import datetime
import cv2

recording = False
recording_time = 10

model = tf.keras.models.load_model('saved_model_openpose_ears_ver7', compile=False)
pred = tf.keras.backend.function([model.input], [model.get_layer('s6').output])
map_size = 100

input_size = 200
landmark_size = 55

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

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

start = time.time()  # 시작 시간 저장

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    
    frame = cv2.resize(frame, (400,400), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if count == 0: 
        rects = detector(gray, 1)
        count = face_detect_rate

    if len(rects)==0:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)
        result = pred([np.expand_dims(image, axis=0)/255.])[0]
        result[result < 0.5] = 0 # threshold setting
        result = np.argmax(result.reshape(-1,landmark_size), axis=0)
        prev_xy = []
        for idx in result:
            x, y = idx%map_size/map_size, idx//map_size/map_size
            x *= 400
            y *= 400
            if x < 1 or y < 1 : continue
            cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
            prev_xy.append([int(x),int(y)])
        if len(prev_xy) == landmark_size :cv2.polylines(frame, [np.asarray(prev_xy)], True , (0, 255, 0), 1)
    else:
        for i, rect in enumerate(rects):
            l = rect.left()
            t = rect.top()
            b = rect.bottom()
            r = rect.right()
            shape = predictor(gray, rect)
            for j in range(68):
                x, y = shape.part(j).x, shape.part(j).y
                if j in [0, 1, 2, 14, 15, 16]: cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                else: cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            #cv2.circle(frame, (shape.part(1).x, shape.part(1).y), 2, (0, 0, 255), -1)
            #cv2.circle(frame, (shape.part(15).x, shape.part(15).y), 2, (0, 0, 255), -1)
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 1)

    if ret and recording:
        if (time.time() - start) > recording_time: break
        out.write(frame)

    cv2.imshow("VFrame", frame)
    count -= 1
    

capture.release()
out.release()
cv2.destroyAllWindows()