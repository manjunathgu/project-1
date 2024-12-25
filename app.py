# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pickle
import numpy as np
import mysql.connector as mq
from mysql.connector import Error
from markupsafe import Markup

# Load the Random Forest CLassifier model
filename = 'dt.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'


def dbconnection():
    con = mq.connect(host='localhost', database='diabetis',user='root',password='root')
    return con

@app.route('/loginpage')
def loginpage():
    return render_template('login.html', title='Login')

@app.route('/')
def home():
    return render_template('login.html', title='Login')

@app.route('/registerpage')
def registerpage():
    return render_template('register.html', title='Login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        con = dbconnection()
        cursor = con.cursor()
        cursor.execute("select * from user where email='{}' and password='{}'".format(email,password))
        res = cursor.fetchall()
        if res==[]:
            message = Markup("<h3>Failed! Invalid Email or Password</h3>")
            flash(message)
            return redirect(url_for('loginpage'))
        else:
            return render_template('index.html', title='check')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        address = request.form['address']
        con = dbconnection()
        cursor = con.cursor()
        cursor.execute("select * from user where email='{}'".format(email))
        res = cursor.fetchall()
        if res==[]:
                cursor.execute("insert into user(name,email,phone,password,address)values('{}','{}','{}','{}','{}')".format(name,email,phone,password,address))
                con.commit()
                con.close()
                message = Markup("<h3>Success! Registration success</h3>")
                flash(message)
                return redirect(url_for('loginpage'))
        else:
           message = Markup("<h3>Failed! Email Id already Exist</h3>")
           flash(message)
           return redirect(url_for('registerpage'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
