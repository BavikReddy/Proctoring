import os
import sys
import json
import time
import numpy as np
import cv2
from datetime import datetime,date
from flask_mysqldb import MySQL
from flask import Flask,render_template,request, Response
from Face_Verification import Reg_user_verification,signup_process,Admin_Reg_user_verification
from Smart_webcam import Integrator
# from live_feed import live_feed_frames


#from DB_dashboard_data import DB_data

template_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
template_dir = os.path.join(template_dir, 'PycharmProjects')
template_dir = os.path.join(template_dir, 'Online Proctoring')
template_dir = os.path.join(template_dir, 'Templates')

U_Id=''
t_end=''
A_Id=''
today = datetime.now()
today=str(today)
#print(today)
#
# connection_alerts = pymysql.connect(host='127.0.0.1',
#                                  port=3306,
#                                  user='root',
#                                  password='Welcome$123',
#                                  database='proctoring')


app1=Flask(__name__, template_folder=template_dir)

# app1.config['CACHE_TYPE']='null'
# cache.init_app(app1)

app1.config['MYSQL_HOST'] = '127.0.0.1'
app1.config['MYSQL_PORT'] = 3306
app1.config['MYSQL_USER'] = 'root'
app1.config['MYSQL_PASSWORD'] = 'Welcome$123'
app1.config['MYSQL_DB'] = 'proctoring'

mysql = MySQL(app1)


@app1.route('/')
def home():
    return render_template('index.html')

@app1.route('/sample')
def sample():
    return render_template('sample.html')

@app1.route('/Login_page',methods=['GET','POST'])
def login_page():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        user_id = request.form.get("UserId")
        passwd = request.form.get("password")
        # print("App",user_id)
        global U_Id
        U_Id = user_id
        # print("user_id:", user_id, '/n', 'type', type(user_id))
        # print("passwd:", passwd)
        ret = Reg_user_verification(user_id, passwd)
        if ret == 'User verified':

            return render_template('loading.html')
        elif ret == 'User not verified':
            error = "User not verified"
            return render_template('login.html', error=error)
        elif ret == 'Password Incorrect':
            error = 'Invalid Credentials. Please try again.'
            return render_template('login.html', error=error)
        else:
            error = "Face not visible to webcam"
            return render_template('login.html', error=error)

@app1.route('/home')
def Home_page():
    return render_template('home.html')

@app1.route('/Login_page/home',methods=['GET','POST'])
def Home():
    user_id=request.form.get("UserId")
    passwd=request.form.get("password")
    #print("App",user_id)
    global U_Id
    U_Id=user_id
    #print("user_id:", user_id, '/n', 'type', type(user_id))
    # print("passwd:", passwd)
    ret=Reg_user_verification(user_id,passwd)
    if ret=='User verified':
        return render_template('home.html')
    elif ret=='User not verified':
        error = "User not verified"
        return render_template('login.html', error=error)
    elif ret== 'Password Incorrect':
        error = 'Invalid Credentials. Please try again.'
        return render_template('login.html', error=error)
    elif ret== 'Face not visible':
        error = "Face not visible to webcam"
        return render_template('login.html', error=error)

@app1.route('/Login_page/home/UPES',methods=['GET','POST'])
def UPES():
    if request.method == "GET":

        return render_template('startexam.html')


@app1.route('/Login_page/home/ULAW',methods=['GET','POST'])
def ULAW():
    return render_template('startexam.html')

@app1.route('/Login_page/home/BSBI',methods=['GET','POST'])
def BSBI():
    return render_template('startexam.html')

@app1.route('/Login_page/home/PEARL',methods=['GET','POST'])
def PEARL():
    return render_template('startexam.html')

@app1.route('/Signup',methods=['GET'])
def signup():
    return render_template('signup.html')

@app1.route('/register',methods=['POST'])
def register():
    user_id=request.form.get("UserId")
    passwd=request.form.get("password")
    email=request.form.get("email")
    name=request.form.get("firstname")
    signup_process(user_id, passwd, name, email)
    return render_template('login.html')

@app1.route('/results')
def Results():
    return render_template('results.html')

@app1.route('/monitoring',methods=['GET'])
def monitor():
    global today
    today = datetime.now()
    today = str(today)
    Integrator(U_Id,5)

    return "monitored"


##Dashboard
@app1.route('/admin_login',methods=['GET','POST'])
def login_admin():
    if request.method == 'GET':
        return render_template('admin-login.html')

    else:

        admin_id = request.form.get("AdminId")
        admin_passwd = request.form.get("password")
        # print("App",user_id)

        # print("admin_id:", admin_id, '/n', 'type', type(admin_id))
        # print("passwd:", admin_passwd)
        ret = Admin_Reg_user_verification(admin_id, admin_passwd)
        if ret == 'User verified':
            return render_template('admin-dashboard.html')
        else:
            error = 'Invalid Credentials. Please try again.'
            return render_template('admin-login.html', error=error)


@app1.route('/dashboard',methods=['GET','POST'])
def dashboard():
    return render_template('admin-dashboard.html')

@app1.route('/monitor_exam',methods=['GET'])

def monitor_exam():
    mycursor_alerts=mysql.connection.cursor()
    #print("SQL connected")
    sql_distict = "select (a.user_id),u.name from all_data as a join users as u where a.user_id=u.user_id order by a.captured_date_time desc;"

    mycursor_alerts.execute(sql_distict)
    result = mycursor_alerts.fetchall();
    out_list=[]
    result1 = []
    for i in result:
        if i in result1:
            pass
        else:
            result1.append(i)
    out = [t for t in result1 ]
    #print(out)
    for i in out:
        sql = "select text_data,count(text_data) as count,captured_type,(CAST(max(captured_date_time) as char)) as Capture_Time from all_data where user_id="+i[0]+" and captured_date_time >='"+today+"' group by user_id,text_data order by Capture_Time Desc;"
        mycursor_alerts.execute(sql)
        result1 = mycursor_alerts.fetchall()

        list = []
        #print("Before for loop",result1)

        for k in result1:
            alert = {'message': k[0], 'count': k[1], 'type': k[2], 'datetime': k[3]}
            list.append(alert)

        out1 = {'user_id': i[0],'user_name':i[1], 'alerts': list}
        out_list.append(out1)
    mycursor_alerts.close()
    return render_template('admin-monitor-exam.html',data=out_list)

#
# def monitor_exam1():
#     mycursor_alerts=mysql.connection.cursor()
#     #print("SQL connected")
#     sql_distict = "select (a.user_id),u.name from all_data as a join users as u where a.user_id=u.user_id order by a.captured_date_time desc;"
#
#     mycursor_alerts.execute(sql_distict)
#     result = mycursor_alerts.fetchall();
#     out_list=[]
#     result1 = []
#     for i in result:
#         if i in result1:
#             pass
#         else:
#             result1.append(i)
#     out = [t for t in result1 ]
#     #print(out)
#     for i in out:
#         sql = "select b.text_data,b.count,b.captured_type,b.Capture_Time,a.images_captured from all_data as a join (select text_data,count(text_data) as count,captured_type,(CAST(max(captured_date_time) as char)) as Capture_Time from all_data where user_id="+i[0]+" and captured_date_time like '2021-03%' group by user_id,text_data order by Capture_Time Desc) as b where a.captured_date_time=b.Capture_Time order by b.Capture_Time Desc;"
#         mycursor_alerts.execute(sql)
#         result1 = mycursor_alerts.fetchall()
#
#         list = []
#         #print("Before for loop",result1)
#
#         for k in result1:
#             imS=None
#             if k[4]!=None:
#                 jpg_as_np = np.frombuffer(k[4], dtype=np.uint8)
#                 image_buffer = cv2.imdecode(jpg_as_np, flags=1)
#                 imS = cv2.resize(image_buffer, (960, 540))
#                 print("type of Ims",type(imS))
#                 print("shape of Ims", imS.shape)
#             alert = {'message': k[0], 'count': k[1], 'type': k[2], 'datetime': k[3],'image': imS}
#             list.append(alert)
#
#         out1 = {'user_id': i[0],'user_name':i[1], 'alerts': list}
#         out_list.append(out1)
#     mycursor_alerts.close()
#     return render_template('admin-monitor-exam.html',data=out_list)


# @app1.route('/feeds',methods=['GET'])
# def video_feed():
#     print("Entering into video feed")
#     return Response(live_feed_frames('12',3),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':

    app1.run(debug=True)