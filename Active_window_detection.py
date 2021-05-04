import win32gui
import time
from win10toast import ToastNotifier
import pymysql
import pymysql.cursors
from datetime import datetime
import sys
import os


# Connect to the database
connection_tabs = pymysql.connect(host='127.0.0.1',
                             port=3306,
                             user='root',
                             password='Welcome$123',
                             database='proctoring')

mycursor_tabs = connection_tabs.cursor()


def Insert_data_mysql(userid, textdata, now, capturedtype):
    sql = "INSERT INTO all_data (user_id, text_data,captured_date_time,captured_type) VALUES (%s, %s,%s, %s)"
    val = (userid, textdata, now, capturedtype)
    mycursor_tabs.execute(sql, val)
    connection_tabs.commit()

def Tabs_monitoring(U_Id,timer):
    userid = str(U_Id)
    #print("userid",userid,'\n',"Type",type(userid))
    timer=timer

    capturedtype = 'Active window detection'
    toaster = ToastNotifier()
    window = win32gui.GetForegroundWindow()
    active_window_name = win32gui.GetWindowText(window)
    tab=str(active_window_name)
    t_start=time.time()
    t_end = t_start + 60 * timer
    count = 0

    while time.time()<t_end:

        window = win32gui.GetForegroundWindow()
        active_window_name = win32gui.GetWindowText(window)
        if str(active_window_name) == tab:
            pass
        else:
            count = count + 1
            toaster.show_toast("Alert",
                               "Please dont switch windows during Exam.", duration=5)

            print(str(active_window_name))
            now = datetime.now()
            textdata=str(active_window_name)
            Insert_data_mysql(userid, textdata, now, capturedtype)


#Tabs_monitoring(123,1)