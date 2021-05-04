import cv2
from Face_detector_and_landmarks import get_face_detector, find_faces,get_landmark_model, detect_marks,draw_marks;

face_model = get_face_detector()
landmark_model = get_landmark_model()
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0] * 5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0] * 3
font = cv2.FONT_HERSHEY_SIMPLEX

def return_distances():
    cap = cv2.VideoCapture(0)

    while (True):
        ret, img = cap.read()
        rects = find_faces(img, face_model)
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            for i in range(100):
                for i, (p1, p2) in enumerate(outer_points):
                    d_outer[i] += shape[p2][1] - shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points):
                    d_inner[i] += shape[p2][1] - shape[p1][1]
            break
        break
    cv2.destroyAllWindows()
    d_outer[:] = [x / 100 for x in d_outer]
    d_inner[:] = [x / 100 for x in d_inner]

    return d_outer,d_inner


#d_outer, d_inner = return_distances()