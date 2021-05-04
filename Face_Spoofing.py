import numpy as np
import cv2
from sklearn.externals import joblib
def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)
modelFile = r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Face Spoofing\res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Face Spoofing\deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
#p_file = "C:\Users\bareddy\Desktop\Bavik\GUS\Opencv\Proctoring-AI-master\Proctoring-AI-master\models\face_spoofing.pkl"

clf = joblib.load("C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Face Spoofing\\face_spoofing - Copy.pkl" )

cap = cv2.VideoCapture(0)
# width = 320
# height = 240
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

sample_number = 1
count = 0
measures = np.zeros(sample_number, dtype=np.float)

while True:
    ret, img = cap.read()
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    faces3 = net.forward()

    measures[count % sample_number] = 0
    height, width = img.shape[:2]
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.5:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            # cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 5)
            roi = img[y:y1, x:x1]

            point = (0, 0)

            img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
            img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

            ycrcb_hist = calc_hist(img_ycrcb)
            luv_hist = calc_hist(img_luv)

            feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
            feature_vector = feature_vector.reshape(1, len(feature_vector))

            prediction = clf.predict_proba(feature_vector)
            prob = prediction[0][1]

            measures[count % sample_number] = prob

            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)

            point = (x, y - 5)

            print(measures, np.mean(measures))
            if 0 not in measures:
                text = "True"
                if np.mean(measures) >= 0.7:
                    text = "False"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                                thickness=2, lineType=cv2.LINE_AA)
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9,
                                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    count += 1
    cv2.imshow('img_rgb', img)

    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()