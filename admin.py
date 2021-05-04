import os
import pymysql
import pandas as pd
import json
from flask import Flask,render_template,request
from DB_dashboard_data import DB_data

template_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
template_dir = os.path.join(template_dir, 'PycharmProjects')
template_dir = os.path.join(template_dir, 'Online Proctoring')
template_dir = os.path.join(template_dir, 'Templates')

app1=Flask(__name__, template_folder=template_dir)
@app1.route('/admin_login')
def login_admin():
    return render_template('admin-login.html')
@app1.route('/dashboard',methods=['POST'])
def dashboard():
    return render_template('admin-dashboard.html')

@app1.route('/monitor_exam',methods=['GET'])
def monitor_exam():
    out_list=DB_data()
    return render_template('admin-monitor-exam.html',data=out_list)

@app1.route('/index')
def index():
    return render_template('index.html')

if __name__=='__main__':
    app1.run(debug=True)