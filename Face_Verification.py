import cv2,os
import csv
import numpy as np
import pandas as pd
import datetime
import time
import pymysql
import pymysql.cursors
from PIL import Image, ImageTk

def Reg_user_verification(user_id,passwd):
    # Connect to the database
    connection_Identification = pymysql.connect(host='127.0.0.1',
                                     port=3306,
                                     user='root',
                                     password='Welcome$123',
                                     database='proctoring')

    mycursor_Identification = connection_Identification.cursor()

    # def Insert_data_mysql_track(userid, textdata, now, capturedtype):
    #     sql = "INSERT INTO all_data (user_id, text_data,captured_date_time,captured_type) VALUES (%s, %s,%s, %s)"
    #     val = (userid, textdata, now, capturedtype)
    #     mycursor_Identification.execute(sql, val)
    #     connection_Identification.commit()

    def Registered_user():
        #print("Entering")
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
            recognizer.read(
                r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Face_Identification\TrainingImageLabel\Trainner.yml")
            harcascadePath = 'C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Haar Cascades\\haarcascade_frontalface_default.xml'
            faceCascade = cv2.CascadeClassifier(harcascadePath);
            # col_names = ['Id', 'Name']
            # df = pd.read_csv(
            #     r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Face_Identification\Student_details\StudentDetails.csv",
            #     names=col_names)
            mycursor_Identification.execute("SELECT user_id,password FROM users")
            data = mycursor_Identification.fetchall()
            df = pd.DataFrame(data, columns=['Id', 'Password'])

            db_pwd= df.loc[df['Id'] == str(user_id)]['Password'].values
            #print("db_pwd:",db_pwd)
            if db_pwd== passwd:
                #print("Entering into loop")
                cam = cv2.VideoCapture(0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                count=0
                count_not_verified = 0
                # col_names = ['Id', 'Name', 'Date', 'Time']
                # attendance = pd.DataFrame(columns=col_names)
                while True:
                    ret, im = cam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

                        if (conf < 50):
                            print("conf after successful login", conf)
                            if user_id==str(Id):

                                txt='User verified'

                                cam.release()
                                cv2.destroyAllWindows()
                            else:
                                print("LoginID",user_id)
                                print("Pred_ID",Id)
                                count_not_verified=count_not_verified+1

                                if count_not_verified>=5:
                                    txt = 'User not verified'

                                    cam.release()
                                    cv2.destroyAllWindows()
                                    count_not_verified = 0


                        else:
                            print("conf for unsuccessful login",conf)
                            txt = 'Face not visible'
                            cam.release()
                            cv2.destroyAllWindows()
                            # count=count+1
                            # if count>=5:
                            #
                            #     Id = 'Unknown'
                            #     txt='Face not visible'
                            #     #print(txt)
                            #
                            #     count=0
                            #     cam.release()
                            #     cv2.destroyAllWindows()
                        cv2.imshow('Face Verification', im)
                        if (cv2.waitKey(1) == ord('q')):
                            break

            else:
                txt='Password Incorrect'
        except:
            return txt
        return txt

    ret1=Registered_user()
    return ret1
# ret=Reg_user_verification(123,123)
# print(ret)

def signup_process(user_id,passwd,name,email):
    # Connect to the database
    connection_Identification = pymysql.connect(host='127.0.0.1',
                                     port=3306,
                                     user='root',
                                     password='Welcome$123',
                                     database='proctoring')

    mycursor_Identification = connection_Identification.cursor()

    def DB_storage():

        now = datetime.datetime.now()
        sql = "INSERT INTO users (user_id, name,email,password,created_at) VALUES (%s, %s,%s, %s,%s)"
        val = (user_id,name,email,passwd,now)
        mycursor_Identification.execute(sql, val)
        connection_Identification.commit()
        print("User data stored in DB")


    def TakeImages():

        cam = cv2.VideoCapture(0)
        harcascadePath = 'C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Haar Cascades\\haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Face_Identification\\TrainingImage\\ " + name + "." + str(user_id) + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 120:
                res = "Your Images are Captured"
                print(res)
                #self.message.configure(text=res)
                break
        cam.release()
        cv2.destroyAllWindows()
        #res = "Images Saved for ID : " + str(Id) + " Name : " + name

        #self.message.configure(text=res)


    def TrainImages():
        recognizer = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
        harcascadePath = 'C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Haar Cascades\\haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(harcascadePath)
        faces, Id = getImagesAndLabels(r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Face_Identification\TrainingImage")
        recognizer.train(faces, np.array(Id))
        recognizer.save("C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Face_Identification\\TrainingImageLabel\\Trainner.yml")
        res = "Images Trained"  # +",".join(str(f) for f in Id)
        print(res)
        #self.message.configure(text=res)

    def getImagesAndLabels(path):
        # get the path of all the files in the folder
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # print(imagePaths)

        # create empth face list
        faces = []
        # create empty ID list
        Ids = []
        # now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # getting the Id from the image
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces.append(imageNp)
            #print("ID",Id)
            Ids.append(Id)
        return faces, Ids

    mycursor_Identification.execute("SELECT user_id FROM users")
    data = mycursor_Identification.fetchall()
    df = pd.DataFrame(data, columns=['Id'])
    # df = pd.read_csv(r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Face_Identification\Student_details\StudentDetails.csv", names=col_names)
    if user_id not in df.values:
        TakeImages()
        TrainImages()
        DB_storage()
    else:
        print("Your Id is already Registered")


def Admin_Reg_user_verification(user_id,passwd):
    # Connect to the database
    connection_admin_Identification = pymysql.connect(host='127.0.0.1',
                                     port=3306,
                                     user='root',
                                     password='Welcome$123',
                                     database='proctoring')

    mycursor_admin_Identification = connection_admin_Identification.cursor()

    mycursor_admin_Identification.execute("select user_id,password from admin_details;")
    data = mycursor_admin_Identification.fetchall()
    df_admin = pd.DataFrame(data, columns=['Id', 'Password'])

    db_pwd = df_admin.loc[df_admin['Id'] == str(user_id)]['Password'].values

    if db_pwd == passwd:
        txt = 'User verified'
        print(txt)
    else:
        txt = 'Password Incorrect'
        print(txt)
    return txt

