# Importing the libraries
import cv2
import numpy as np
import pandas as pd
import time
import threading
from threading import Thread
import math
import speech_recognition as sr
import pyaudio
import wave
import os
from datetime import datetime,date
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
import warnings
# import multiprocessing

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import tensorflow as tf
import pymysql
import pymysql.cursors
import base64
tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

# global main_terminator
# main_terminator=False


def Integrator(U_Id,timer):

    from queue import Queue
    from Face_detector_and_landmarks import get_face_detector, find_faces, get_landmark_model, detect_marks, draw_marks;
    from Eye_Detector_libraries import eye_on_mask, find_eyeball_position, contouring, process_thresh, print_eye_pos, \
        nothing;
    from Mouth_Open_libraries import return_distances
    from MobilePhone_and_MultipleFaces_libraries import YoloV3, load_darknet_weights, draw_outputs
    from Active_window_detection import Tabs_monitoring
    from HeadPose_Estimation_libraries import FaceDetector, MarkDetector, draw_annotation_box, Pose
    from Speech_Text_Conversion_libraries import speech_analysis

    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    d_outer, d_inner = return_distances()

    yolo = YoloV3()
    load_darknet_weights(yolo, 'C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Yolov3\\yolov3.weights')
    userid=str(U_Id)
    timer=timer
    out=''

    #SQL Connection
    connection_eyes_mouth = pymysql.connect(host='127.0.0.1',
                                 port=3306,
                                 user='root',
                                 password='Welcome$123',
                                 database='proctoring')
    mycursor_eyes_mouth = connection_eyes_mouth.cursor()

    def Insert_data_mysql_eyes_mouth(userid, textdata, now, capturedtype):
        sql = "INSERT INTO all_data (user_id, text_data,captured_date_time,captured_type) VALUES (%s, %s,%s, %s)"
        val = (userid, textdata, now, capturedtype)
        mycursor_eyes_mouth.execute(sql, val)
        connection_eyes_mouth.commit()

    #SQL Connection
    connection_mobile = pymysql.connect(host='127.0.0.1',
                                 port=3306,
                                 user='root',
                                 password='Welcome$123',
                                 database='proctoring')
    mycursor_mobile = connection_mobile.cursor()

    def Insert_data_mysql_mobile(userid, textdata, now, capturedtype,capturedimage):
        sql = "INSERT INTO all_data (user_id, text_data,captured_date_time,captured_type,images_captured) VALUES (%s, %s,%s, %s,%s)"
        val = (userid, textdata, now, capturedtype,capturedimage)
        mycursor_mobile.execute(sql, val)
        connection_mobile.commit()

    #SQL Connection
    connection_head = pymysql.connect(host='127.0.0.1',
                                 port=3306,
                                 user='root',
                                 password='Welcome$123',
                                 database='proctoring')
    mycursor_head = connection_head.cursor()

    def Insert_data_mysql_head(userid, textdata, now, capturedtype):
        sql = "INSERT INTO all_data (user_id, text_data,captured_date_time,captured_type) VALUES (%s, %s,%s, %s)"
        val = (userid, textdata, now, capturedtype)
        mycursor_head.execute(sql, val)
        connection_head.commit()

    # Connect to the database
    connection_Identification = pymysql.connect(host='127.0.0.1',
                                 port=3306,
                                 user='root',
                                 password='Welcome$123',
                                 database='proctoring')

    mycursor_Identification = connection_Identification.cursor()

    # Connect to the database
    connection_livefeed = pymysql.connect(host='127.0.0.1',
                                          port=3306,
                                          user='root',
                                          password='Welcome$123',
                                          database='proctoring')

    mycursor_livefeed = connection_livefeed.cursor()

    def Insert_data_mysql_track(userid, textdata, now, capturedtype):
        sql = "INSERT INTO all_data (user_id, text_data,captured_date_time,captured_type) VALUES (%s, %s,%s, %s)"
        val = (userid, textdata, now, capturedtype)
        mycursor_Identification.execute(sql, val)
        connection_Identification.commit()

        # Connect to the database
        connection_testfeed = pymysql.connect(host='127.0.0.1',
                                                    port=3306,
                                                    user='root',
                                                    password='Welcome$123',
                                                    database='proctoring')

        mycursor_testfeed = connection_testfeed.cursor()

        def Insert_data_mysql_feed(userid, now, image):
            sql = "INSERT INTO test_videofeed (userid,time_stamp,image) VALUES (%s, %s,%s)"
            val = (userid, now, image)
            mycursor_testfeed.execute(sql, val)
            connection_testfeed.commit()



    #_, frame_size = cap.read()
    #ret, img = cap.read()


    def eyes_mouth():
        #cap = cv2.VideoCapture(0)
        #ret, img = cap.read()
        #thresh = img.copy()
        left = [36, 37, 38, 39, 40, 41]
        right = [42, 43, 44, 45, 46, 47]

        outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
        inner_points = [[61, 67], [62, 66], [63, 65]]
        font = cv2.FONT_HERSHEY_SIMPLEX


        try:
            while time.time()<t_end:
                ret, img = cap.read()
                thresh = img.copy()
                cv2.namedWindow('image')
                kernel = np.ones((9, 9), np.uint8)
                cv2.createTrackbar('threshold', 'image', 75, 255, nothing)
                rects = find_faces(img, face_model)
                count=0

                for rect in rects:
                    shape = detect_marks(img, landmark_model, rect)

                    #Mouth
                    cnt_outer = 0
                    cnt_inner = 0
                    #userid=1234
                    textdata = 'Mouth Open'
                    capturedtype='Mouth Open detection'
                    draw_marks(img, shape[48:])
                    for i, (p1, p2) in enumerate(outer_points):
                        if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                            cnt_outer += 1
                    for i, (p1, p2) in enumerate(inner_points):
                        if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
                            cnt_inner += 1
                    #print("cnt_outer:",cnt_outer)
                    #print("cnt_inner:", cnt_inner)
                    if cnt_outer > 3 and cnt_inner > 2:
                        count=count+1
                        if count>=5:
                            print('Mouth open')
                            cv2.putText(img, 'Mouth open', (60, 60), font,
                                        1, (0, 255, 255), 2)
                            now = datetime.now()
                            Insert_data_mysql_eyes_mouth(userid, textdata, now, capturedtype)
                            count=0



                    #Eyes
                    capturedtype_eyes='Eyeball detection'
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    mask, end_points_left = eye_on_mask(mask, left, shape)
                    mask, end_points_right = eye_on_mask(mask, right, shape)
                    mask = cv2.dilate(mask, kernel, 5)
                    eyes = cv2.bitwise_and(img, img, mask=mask)
                    mask = (eyes == [0, 0, 0]).all(axis=2)
                    eyes[mask] = [255, 255, 255]
                    mid = (shape[42][0] + shape[39][0]) // 2
                    eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                    threshold = cv2.getTrackbarPos('threshold', 'image')
                    _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                    thresh = process_thresh(thresh)
                    eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
                    eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
                    txt_eyes=print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
                    # now = datetime.now()
                    # if txt_eyes != '':
                    #     out_q.put(txt_eyes)
                    #     #Insert_data_mysql_eyes_mouth(userid, txt_eyes, now, capturedtype_eyes)



                cv2.imshow('Result', img)
                cv2.imshow("image", thresh)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except :
            eyes_mouth()

    def count_people_and_phones():
        try:

            while time.time()<t_end:
                capturedtype_phone='mobile and multiple_persons detection'
                ret, img = cap.read()
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (320, 320))
                frame = frame.astype(np.float32)
                frame = np.expand_dims(frame, 0)
                frame = frame / 255
                class_names = [c.strip() for c in open(r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Yolov3\classes.TXT").readlines()]
                boxes, scores, classes, nums = yolo(frame)
                count=0
                for i in range(nums[0]):
                    if int(classes[0][i] == 0):
                        count +=1
                    if int(classes[0][i] == 67) or int(classes[0][i] == 65):
                        now = datetime.now()
                        txt_mobile="Mobile Phone Detected"
                        #print(txt_mobile)
                        ret, buffer = cv2.imencode('.jpg', img)
                        frame1 = base64.b64encode(buffer)
                        Insert_data_mysql_mobile(userid, txt_mobile, now, capturedtype_phone,frame1)
                        print(txt_mobile)
                if count == 0:
                    now = datetime.now()
                    txt_person = 'No person detected'
                    Insert_data_mysql_mobile(userid, txt_person, now, capturedtype_phone,img)
                    print(txt_person)
                elif count > 1:
                    now = datetime.now()
                    txt_person = 'More than one person detected'
                    Insert_data_mysql_mobile(userid, txt_person, now, capturedtype_phone,img)
                    print(txt_person)

                image = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imshow('Prediction', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            count_people_and_phones()

    def Headpose():
        mark_detector = MarkDetector()
        ret, img = cap.read()
        size = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])
        capturedtype_head='Head and Eye Detector'
        # Camera internals
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        try:

            while time.time()<t_end:
                ret, img = cap.read()
                #size = img.shape
                if ret == True:
                    #print("1st stage of head")
                    faceboxes = mark_detector.extract_cnn_facebox(img)
                    for facebox in faceboxes:
                        #             print('facebox[0]:',facebox[0])
                        #             print('facebox[1]:',facebox[1])
                        #             print('facebox[2]:',facebox[2])
                        #             print('facebox[3]:',facebox[3])
                        face_img = img[facebox[1]: facebox[3],
                                   facebox[0]: facebox[2]]
                        face_img = cv2.resize(face_img, (128, 128))
                        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        marks = mark_detector.detect_marks([face_img])
                        marks *= (facebox[2] - facebox[0])
                        marks[:, 0] += facebox[0]
                        marks[:, 1] += facebox[1]
                        shape = marks.astype(np.uint)
                        # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                        image_points = np.array([
                            shape[30],  # Nose tip
                            shape[8],  # Chin
                            shape[36],  # Left eye left corner
                            shape[45],  # Right eye right corne
                            shape[48],  # Left Mouth corner
                            shape[54]  # Right mouth corner
                        ], dtype="double")
                        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                                      dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

                        # Project a 3D point (0, 0, 1000.0) onto the image plane.
                        # We use this to draw a line sticking out of the nose

                        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                         translation_vector, camera_matrix, dist_coeffs)

                        for p in image_points:
                            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                        p1 = (int(image_points[0][0]), int(image_points[0][1]))
                        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                        x1, x2 = draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix)

                        cv2.line(img, p1, p2, (0, 255, 255), 2)
                        cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                        # for (x, y) in shape:
                        #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                        # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                        try:
                            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                            ang1 = int(math.degrees(math.atan(m)))
                        except:
                            ang1 = 90

                        try:
                            m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                            ang2 = int(math.degrees(math.atan(-1 / m)))
                        except:
                            ang2 = 90

                            # print('div by zero error')
                        # print('ang1-',str(ang2))
                        # print('p1-',tuple(p2))
                        cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
                        cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
                        txt_head=Pose(img, ang1, ang2)
                        # now = datetime.now()
                        # eye_data = in_q.get()
                        # if txt_head != '' and txt_head==eye_data:
                        #     txt = 'User looking in other direction'
                        #     print(txt)
                        #     Insert_data_mysql_head(userid, txt, now, capturedtype_head)
                        #print("Entering final stage into head")

                    cv2.imshow('Headpose', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break

        except:
            Headpose()

    def user_tracker():
        try:
            capturedtype='Face Tracker'
            recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
            recognizer.read(r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Face_Identification\TrainingImageLabel\Trainner.yml")
            harcascadePath = r'C:\Users\bareddy\PycharmProjects\Online Proctoring\Haar Cascades\haarcascade_frontalface_default.xml'
            faceCascade = cv2.CascadeClassifier(harcascadePath);
            col_names = ['Id', 'Name']
            mycursor_Identification.execute("SELECT user_id,name FROM users")
            data = mycursor_Identification.fetchall()
            df = pd.DataFrame(data, columns=['Id', 'Name'])
            #print("DF",df)
            font = cv2.FONT_HERSHEY_SIMPLEX
            count=0

            while time.time()<t_end:
                ret,im = cap.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                #print(type(faces))
                if len(faces)==0:
                    count=count+1
                    if count>=30:
                        txt="User looking in other direction"
                        print(txt)
                        now=datetime.now()
                        Insert_data_mysql_track(userid, txt, now, capturedtype)
                        count=0
                else:
                    count=0
                    for (x, y, w, h) in faces:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        #print("pred ID",Id)
                        if (conf < 50):

                            #aa = df.loc[df['Id'] == str(Id)]['Name'].values
                            tt = str(Id)
                        else:
                            Id = 'Unknown'
                            tt = str(Id)
                        if (conf > 75):
                            noOfFile = len(os.listdir("C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Face_Identification\\ImagesUnknown")) + 1
                            cv2.imwrite("C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Face_Identification\\ImagesUnknown\\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
                        cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
                cv2.imshow('Face_Track', im)
                if (cv2.waitKey(1) == ord('q')):
                    break
        except():
            user_tracker()

    def store_live_feed():
        try:
            col_names = ['Id', 'Name']
            mycursor_livefeed.execute("SELECT user_id,name FROM users")
            data = mycursor_livefeed.fetchall()
            df = pd.DataFrame(data, columns=['Id', 'Name'])
            name_users = (df.loc[df['Id'] == userid]['Name'].values)
            uid=str(userid)
            date_uid=str(datetime.now().date())
            #date_uid = date_uid.replace('-', '_')
            for i in name_users:
                user_name=str(i)
            #print(type(user_name))
            video=user_name+"_"+uid+"_"+date_uid+".webm"
            #video=str(video)
            #print(video)
            video_path="C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\static\\live_feed\\" + video
            #os.path.join('C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\live_feed', video)
            print(video_path)
            cap.set(3, 640)
            cap.set(4, 480)
            #
            # frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'VP80')#*'mp4v'*'XVID'*'VP80'
            #global out
            out = cv2.VideoWriter(filename=video_path, fourcc=fourcc,fps=20.0, frameSize=(640,480))

            #os.path.join('../data/vedio','car2.mp4')
            # loop runs if capturing has been initialized.
            while time.time()<t_end:
                ret,frame = cap.read()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(datetime.now()), (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                out.write(frame)

                cv2.imshow('Original', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            out.release()
        except():
            store_live_feed()

    # def test_feed():
    #     while time.time() < t_end:
    #         ret, im = cap.read()
    #         ret, buffer = cv2.imencode('.jpg', im)
    #         frame1 = base64.b64encode(buffer)
    #         now = datetime.now()
    #         Insert_data_mysql_feed(userid, now, frame1)
    # def live_feed():
    #     t_start_live_feed = time.time()
    #     t_end_live_feed = t_start_live_feed + 60 * timer
    #     while time.time() < t_end_live_feed:
    #         success, frame = cap.read()
    #         if not success:
    #             break
    #         else:
    #             ret, buffer = cv2.imencode('.jpg', frame)
    #             frame = buffer.tobytes()
    #             yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    cap = cv2.VideoCapture(0)

    t_start = time.time()
    t_end = t_start + 60 * timer

    t1 = threading.Thread(target=eyes_mouth, daemon=True)
    t2 = threading.Thread(target=count_people_and_phones, daemon=True)
    t3 = threading.Thread(target=Tabs_monitoring, args=(U_Id, timer,), daemon=True)
    t4 = threading.Thread(target=Headpose, daemon=True)
    t5 = threading.Thread(target=speech_analysis, args=(U_Id, timer,), daemon=True)
    t6 = threading.Thread(target=user_tracker, daemon=True)
    t7 = threading.Thread(target=store_live_feed, daemon=True)


    # if flag==False:
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t7.join()
    t6.join()
    t5.join()
    t4.join()
    t3.join()
    t2.join()
    t1.join()


    # elif flag==True:
    #     try:
    #         print("entered in to integrator")
    #         while time.time() < t_end:
    #             success, frame = cap.read()
    #             if not success:
    #                 break
    #             else:
    #                 ret, buffer = cv2.imencode('.jpg', frame)
    #                 frame = buffer.tobytes()
    #                 yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    #     except():
    #         pass


    cap.release()

    cv2.destroyAllWindows()

#Integrator("12",2)
