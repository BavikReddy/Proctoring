import speech_recognition as sr
import pyaudio
import wave
import time
import threading
import os
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymysql
import pymysql.cursors

#SQL Connection
connection_speech = pymysql.connect(host='127.0.0.1',
                             port=3306,
                             user='root',
                             password='Welcome$123',
                             database='proctoring')
mycursor_speech = connection_speech.cursor()



def speech_analysis(U_Id,timer):
    userid = str(U_Id)
    capturedtype_speech = 'Speech Analysis'
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    filename="C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Speech data\\Recorded Text\\test_" + str(dt_string) + ".txt"

    def Insert_data_mysql(userid, textdata, now, capturedtype):
        sql = "INSERT INTO all_data (user_id, text_data,captured_date_time,captured_type) VALUES (%s, %s,%s, %s)"
        val = (userid, textdata, now, capturedtype)
        mycursor_speech.execute(sql, val)
        connection_speech.commit()

    def read_audio(stream, filename):
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 44100  # Record at 44100 samples per second
        seconds = 10  # Number of seconds to record at once
        filename = filename
        frames = []  # Initialize array to store frames

        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
        # Stop and close the stream
        stream.stop_stream()
        stream.close()


    def convert(i):
        if i >= 0:
            sound = "C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Speech data\\Recorded Speech\\record"+str(i)+".wav"
            r = sr.Recognizer()

            with sr.AudioFile(sound) as source:
                r.adjust_for_ambient_noise(source)
                print("Converting Audio To Text and saving to file..... ")
                audio = r.listen(source)
            try:
                value = r.recognize_google(audio)  ##### API call to google for speech recognition
                os.remove(sound)
                if str is bytes:
                    result = u"{}".format(value).encode("utf-8")
                else:
                    result = "{}".format(value)

                with open(filename,"a") as f:
                    f.write(result)
                    f.write(" ")
                    f.close()

            except sr.UnknownValueError:
                print("")
            except sr.RequestError as e:
                print("{0}".format(e))
            except KeyboardInterrupt:
                pass

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100

    def save_audios(i):
        stream = p.open(format=sample_format,channels=channels,rate=fs,
                    frames_per_buffer=chunk,input=True)
        filename = "C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Speech data\\Recorded Speech\\record"+str(i)+".wav"
        read_audio(stream, filename)

    for i in range((timer*60)//10): # Number of total seconds to record/ Number of seconds per recording
        t1 = threading.Thread(target=save_audios, args=[i])
        x = i-1
        t2 = threading.Thread(target=convert, args=[x]) # send one earlier than being recorded
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        if i==2:
            flag = True
    if flag:
        convert(i)
        p.terminate()

    ###Text Analysis
    try:
        file = open(filename) ## Student speech file
        data = file.read()
        file.close()

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(data)

        filtered_sentence1= [w for w in word_tokens if not w in stop_words]

        filtered_sentence = []

        for w in word_tokens:  ####### Removing stop words
            if w not in stop_words:
                filtered_sentence.append(w)

        ####### creating a final file
        f=open("C:\\Users\\bareddy\\PycharmProjects\\Online Proctoring\\Speech data\\Recorded Speech\\final.txt","w")
        for ele in filtered_sentence:
            f.write(ele+' ')
        f.close()

        ##### checking whether proctor needs to be alerted or not
        file = open(r"C:\Users\bareddy\PycharmProjects\Online Proctoring\Speech data\Steps for building a remote proctor.txt") ## Question file
        data = file.read()
        file.close()

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(data) ######### tokenizing sentence
        filtered_questions = [w for w in word_tokens if not w in stop_words]


        def common_member(a, b):
            a_set = set(a)
            b_set = set(b)

            # check length
            if len(a_set.intersection(b_set)) > 0:
                return (a_set.intersection(b_set))
            else:
                return ([])

        comm = common_member(filtered_questions, filtered_sentence)
        if len(comm) != 0:
            print(comm)
            now = datetime.now()
            txt_common='Person is speaking related to Exam'
            Insert_data_mysql(userid, txt_common, now, capturedtype_speech)
        else:
            txt_notcommon='Not Related to Exam'
            now = datetime.now()
            Insert_data_mysql(userid, txt_notcommon, now, capturedtype_speech)
            print(txt_notcommon)
    except:
        txt_unrelated = "There is noise,No words are detected"
        now = datetime.now()
        Insert_data_mysql(userid, txt_unrelated, now, capturedtype_speech)
        print(txt_unrelated)
