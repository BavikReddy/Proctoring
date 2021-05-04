# Proctoring
Online Proctoring: It is a service which is used to monitor activities of a person through their webcam and microphone.
This Project aim is to create an automated proctoring system for Online Examination. 

app.py -- This is the main file which is built using FLask framework. Here dataflow from frontend(website build using HTML pages) to backend is managaed and viceversa.

The project is divided into multiple tasks. 

1.	Face & Eye detection -- Face_and_EyeDetection.py
2.	Eyeball Detection  -- Eye_Detector_libraries.py
3.	Head pose estimator -- HeadPose_Estimation_libraries.py
4.	Mouth open or close Detection -- Mouth_Open_libraries.py
5.	speech Detection -- Speech_Text_Conversion_libraries.py
6.	Speech to text conversion -- Speech_Text_Conversion_libraries.py
7.	Mobile phone detection -- MobilePhone_and_MultipleFaces_libraries.py
8.	Multiple face detection -- MobilePhone_and_MultipleFaces_libraries.py
9.	Face spoofing -- Face_Spoofing.py
10.	User verification -- Face_Verification.py
11.	Active window detection -- Active_window_detection.py
12.	Integrating all these detections in single application -- Smart_webcam.py

1)	Face & Eye detection:

•	Face and Eyes are detected for any person who is present in front of webcam.
•	There are multiple ways to achieve this task and I have checked all those possibilities and selected best algorithm and features that are required to capture accurately.

2)	Eyeball Detection:

•	Eyeball detection is required to track whether the person is attentive or not.
•	It can able to identify in which direction person is looking and this can be achieved by first finding facial landmarks and then detecting eye balls using those landmarks.
•	This also depends on lighting of the location of person and for that I have built a scrollable Threshold setter which can be adjusted for all lighting conditions.

3)	Head Pose Estimator:

•	Head pose estimator is used to find in which direction the head is facing.
•	It is a challenging problem in computer vision because of the various steps are required to solve it.
•	Firstly, we need to locate the face in the frame and then the various facial landmarks. After getting landmarks the webcam position is also required to find the accurate angle in which the head is posing.



4)	Mouth Open or Close Detection:

•	The detection of mouth in the face is required because we need to identify whether the person is speaking or not.
•	This task can be achieved by finding mouth landmarks in face and using those landmarks we can able to identify if the mouth is open or close.

5)	Speech Detection:

•	If the person is speaking then we need to detect the speech using Microphone as it is essential to recognize what the person is speaking.

6)	Speech to Text Conversion:

•	It is essential to convert speech to text because we can analyse what the person is speaking whether it is relevant or irrelevant to exam.
•	Here we use Natural Language processing to achieve this task.

7)	Mobile Phone Detection:

•	Mobile phone detection is required to identify whether the person is using mobile phone while attempting exam.
•	This can be achieved using YOLO designed algorithms where we need to integrate in our code.

8)	Multiple Face Detection:

•	Multiple face detection is required to count how many persons are present while attempting exam.
•	It counts no. of faces present and calculates how much time 2nd person is present.



9)	Face Spoofing:

•	Face spoofing is finding the real face and fake face(Face show in mobile/paper etc towards webcam).
•	It is helpful whether true person is attempting exam or he is faking by proxy.

10)	User Verification:

•	User verification is required while starting the exam whether the person mentioned in ID is attempting the exam or not.
•	It comes under Face Recognition concept.

11)	Active Window Detection:

•	If the person is switching tab or opening any other thing while attempting an exam we need to capture it notify user not to do that and  also we need to keep a track of that.

12)	Integrating all these detections in single application:

•	All the above mentioned tasks need to be integrated in single application which is more challenging part of all tasks.
•	This task can be done by using Threading technique.




