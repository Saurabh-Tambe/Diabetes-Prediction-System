from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
import csv
import pickle
import numpy as np
import pandas as pd
from flask_mail import Mail
from flask_mail import Message
import mysql
from fpdf import FPDF
from flask import flash
import _datetime
from flask import send_file,current_app
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
import mysql.connector as sql_db
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root123'
app.config['MYSQL_DB'] = 'pythonlogin'


app.config.update(
	DEBUG=True,
	#EMAIL SETTINGS
	MAIL_SERVER='smtp.gmail.com',
	MAIL_PORT=465,
	MAIL_USE_SSL=True,
	MAIL_USE_TLS = False,
	MAIL_USERNAME = 'diabetespredictionsystem@gmail.com',
	MAIL_PASSWORD = 'Admin@123',
	DEFAULT_MAIL_SENDER = 'diabetespredictionsystem@gmail.com'
)
# Intialize MySQL
mysql = MySQL(app)
import smtplib
mail = Mail(app)
 
@app.route('/DPS/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/DPS/contact',methods=['GET', 'POST'])
def contact():

    # Output message if something goes wrong...
    msg = ''
    # Check if "username",  and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'name' in request.form and 'email' in request.form and 'comment' in request.form:
        # Create variables for easy access
        name = request.form['name']
        email = request.form['email']
        comment = request.form['comment']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
         
        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', name):
            msg = 'Username must contain only characters and numbers!'
        elif not name or not email or not comment:
            msg = 'Please fill out the form!'
        else:
 
            cursor.execute('INSERT INTO contacts VALUES (NULL, %s, %s, %s)', (name, email, comment))
            mysql.connection.commit()
            msg = 'Thanks for contacting us, we will get back to you shortly.'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show contact us form with message (if any)
    return render_template('contact.html', msg=msg)
 

@app.route('/DPS/about')
def about():
     
    return render_template('about.html')


@app.route('/forgot',methods=['GET', 'POST'])
def forgot():
    msg = ''
    # Check if email POST requests exist (user submitted form)
    if request.method == 'POST' and 'email' in request.form:
        # Create variables for easy access
        email = request.form['email']
 
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s', [email])
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
           # session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            session['email']=account['email']
             
            msg=Message("Diabetes Prediction::Password Reset",sender="diabetespredictionsystem@gmail.com",recipients=[email])
            msg.body = "Password Reset Link:\n/home/saurabh/Desktop/SEM8/DPS/templates/reset.html"
 
            msg.html="""<html><head></head><body><p>Hi Greetings from Diabetes Prediction System!!<br>You or someone else has requested that a new password be generated for your account. If you made this request, then please click this link: <a href="/home/saurabh/Desktop/DPS/templates/reset.html">Reset Password</a> <br><br> If you did not make this request, then you can simply ignore this email.<br><br>Thanks,<br>Admin <br>Diabetes Prediction System</p></body></html>"""

           # msg.html = render_template('/reset.html')

            mail.send(msg)
            flash("Password reset link sent!")
        else:
            # Account doesnt exist or username/password incorrect
            flash('Incorrect Email!')
 
    return render_template('forgot.html')
 
 

@app.route('/update',methods=['GET', 'POST'])
def update():
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the update page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        username=session['username']
       # username = request.form['username']
        password = request.form['password']
        email = request.form['email'] 


        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        sql_update_query = """Update accounts set password = %s where username = %s"""
        inputData = (password,username)
        cursor.execute(sql_update_query, inputData)
 
        msg = 'Profile successfully Updated!'
        # Show the profile page with account info
        return render_template('profile.html', account=account,msg=msg)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

    

@app.route('/up_pass',methods=['GET', 'POST'])
def up_pass():
    msg = ''
    # Check if  "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'pass' in request.form:
        # Create variables for easy access
        #username = request.form['username']
        password = request.form['pass']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
      # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
 
        if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
          cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
          cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
          account = cursor.fetchone()
          username=session['username']
       # username = request.form['username']
          password = request.form['pass']
       # email = request.form['email'] 
          connection = sql_db.connect(host='localhost',
                                         database='pythonlogin',
                                        user='root',
                                         password='root123')
          cursor = connection.cursor()
           
          cursor.execute('update accounts set password=%s where username=%s',(password,username))
          connection.commit()
 

             
          flash('Profile successfully Updated!')
        # Show the profile page with account info
        return render_template('profile.html', account=account,msg=msg)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
     
 

@app.route('/up_mail',methods=['GET', 'POST'])
def up_mail():
    msg = ''
    # Check if "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'email' in request.form:
        # Create variables for easy access
        #username = request.form['username']
        email = request.form['email']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
      # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
 
        if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
          cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
          cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
          account = cursor.fetchone()
          username=session['username']
       # username = request.form['username']
          email = request.form['email']
       # email = request.form['email'] 
          connection = sql_db.connect(host='localhost',
                                         database='pythonlogin',
                                        user='root',
                                         password='root123')
          cursor = connection.cursor()
 
          cursor.execute('update accounts set email=%s where username=%s',(email,username))
          connection.commit()
 

             
          flash('Profile successfully Updated!')
        # Show the profile page with account info
        return render_template('profile.html', account=account,msg=msg)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/DPS/dashboard')
def dashboard():
     
    return render_template('dashboard.html')


@app.route('/train',methods=['GET', 'POST'])
def train():
    mssg = ''

    filename = session['filename']
    #data = pd.read_csv(filename, header=0)


    data_set = pd.read_csv (filename)  #import data

    features = data_set.iloc [ : , :-1 ]  #EXCEPT LAST COLUMN
    labels = data_set.iloc [ : , -1 ]    #ONLY LAST

    features = features.values
    labels = labels.values

    info=data_set.describe(); #View Statistics related to data
    #print(info)
 

    from sklearn.model_selection import train_test_split
    train_features, test_features, train_labels, test_labels = train_test_split ( features, labels,     test_size=0.25)

    #from sklearn.naive_bayes import GaussianNB
    #classifier = GaussianNB()
    models = []
    RF=RandomForestClassifier()
    #LR=LogisticRegression(solver='liblinear', multi_class='ovr')
 
    BG=BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators =20)
    classifier=VotingClassifier( estimators= [('RF',RF),('BG',BG)], voting = 'hard')
    classifier.fit(train_features, train_labels)

    pickle.dump(classifier, open('model.pkl','wb'))
    flash("Model Trained!!")

# Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
     
    return render_template('dashboard.html',mssg=mssg)

@app.route('/fetch',methods=['GET', 'POST'])
def fetch():
 
    filename = session['filename']
    data = pd.read_csv(filename, header=0)
    stocklist = list(data.values)
    return render_template('dashboard.html', stocklist=stocklist)

@app.route('/perform',methods=['GET', 'POST'])
def perform():
    try:
      filename = session['filename']
      data = pd.read_csv(filename, header=0)








      array = data.values
      X = array[:,0:8]
      y = array[:,8]
      X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.33,random_state=1)
 
 
# Spot Check Algorithms
      models = []
      models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
      #models.append(('LDA', LinearDiscriminantAnalysis()))
      models.append(('KNN', KNeighborsClassifier()))
      models.append(('CART', DecisionTreeClassifier(criterion="entropy")))
      models.append(('NB', GaussianNB()))
      #models.append(('SVM', SVC(gamma='auto')))
      #models.append(('ANN', MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500, random_state=42)))
 
      models.append(('RF', RandomForestClassifier()))
 
      models.append(('BG',BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features =1.0, n_estimators =20)))

      models.append(('ADA',AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)))

      LR=LogisticRegression(solver='liblinear', multi_class='ovr')
      LDA=LinearDiscriminantAnalysis()
      KNN=KNeighborsClassifier()
      CART=DecisionTreeClassifier(criterion="entropy")
      RF=RandomForestClassifier()
      NB=GaussianNB()
      SVM=SVC(gamma='auto')
      ANN=MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500, random_state=42)
      BG=BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5, max_features = 1.0, n_estimators =20)
      ADA=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 5, learning_rate = 1)
 
      models.append(('VOTE',VotingClassifier( estimators= [('RF',RF),('BG',BG)], voting = 'hard')))

# evaluate each model in turn
      results = []
      names = []
      for name, model in models:
  	    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
  	    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
  	    results.append(cv_results)
  	    names.append(name)
  	  #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


      


# Compare Algorithms
      plt.boxplot(results, labels=names)
      plt.title('Algorithm Comparison')
      plt.plot()
      session['strFile']="./static/images/perf.png"

      strFile = "./static/images/perf.png"

      if os.path.isfile(strFile):
        os.remove(strFile)
      plt.savefig(strFile)
      plt.close()

    except KeyError:
      flash('Dataset not uploaded!')

    #plt.savefig('/home/saurabh/Desktop/DPS/static/images/perf.png')
 
    
    return render_template('dashboard.html', name = 'Plot Showing Accuracy of Different Algorithms:', url ='/static/images/perf.png')

@app.route('/report/')
def report():
 
   
        #path = "/home/saurabh/Desktop/DPS/Report.pdf"

        files = session['files']
 
        base_filename='Report'
        #format = 'pdf'
        suffix = '.pdf'
        filess=os.path.join(files, base_filename + suffix)
        
        return send_file(filess, as_attachment=True)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            flash('Logged in successfully!')
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            flash('Incorrect username/password!')
    # Show the login form with message (if any)
    return render_template('login.html', msg=msg)


# http://localhost:5000/python/logout - this will be the logout page
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   session.pop('filename',None)
   session.pop('files',None)
   session.pop('strFile',None)
   #path = "/home/saurabh/Desktop/DPS/Report.pdf"
   #os.remove(path)
   # Redirect to login page
   return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash('Account already exists!')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username must contain only characters and numbers!')
        elif not username or not password or not email:
            flash('Please fill out the form!')
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            flash('You have successfully registered!')
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash('Please fill out the form!')
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)




 
@app.route('/pythonlogin/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

def save_data(result):
    # This is to make sure the HTTP method is POST and not any other
    if request.method == 'POST':
        # request.form is a dictionary that contains the form sent through
        # the HTTP request. This work by getting the name="xxx" attribute of
        # the html form field. So, if you want to get the name, your input
        # should be something like this: <input type="text" name="name" />.
        pregnant_times = request.form['TimesPregnant']
        glucose_tolerance_test = request.form['GlucoseConcentration']
        Diastolic_blood_pressure = request.form['BloodPrs']
        Triceps_skin_fold_thickness = request.form['SkinThickness']
        serum_insulin = request.form['Serum']
        Body_mass_index = request.form['BMI']
        Diabetes_pedigree_function = request.form['DiabetesFunct']
        Age = request.form['Age']

        # This array is the fields your csv file has and in the following code
        # you'll see how it will be used. Change it to your actual csv's fields.
        #fieldnames = ['name', 'comment']
        #fieldnames = ['pregnant', 'glucose_tolerance_test','Diastolic_blood_pressure','Triceps_skin_fold_thickness','serum_insulin','Body_mass_index','Diabetes_pedigree_function','Age']
        # We repeat the same step as the reading, but with "w" to indicate
        # the file is going to be written.
        from collections import OrderedDict
        ordered_fieldnames = OrderedDict([('pregnant',None), ('glucose_tolerance_test',None),('Diastolic_blood_pressure',None),('Triceps_skin_fold_thickness',None),('serum_insulin',None),('Body_mass_index',None),('Diabetes_pedigree_function',None),('Age',None),('class',None)])
        with open('data.csv','a') as inFile:
            # DictWriter will help you write the file easily by treating the
            # csv as a python's class and will allow you to work with
            # dictionaries instead of having to add the csv manually.
            dw = csv.DictWriter(inFile, delimiter='\t', fieldnames=ordered_fieldnames)
            if inFile.tell() == 0:
            #w.writeheader()
               dw.writeheader()
            #writer = csv.DictWriter(inFile, fieldnames=fieldnames)

            # writerow() will write a row in your csv file
            dw.writerow({'pregnant':pregnant_times, 'glucose_tolerance_test':glucose_tolerance_test,'Diastolic_blood_pressure':Diastolic_blood_pressure,'Triceps_skin_fold_thickness':Triceps_skin_fold_thickness,'serum_insulin':serum_insulin,'Body_mass_index':Body_mass_index,'Diabetes_pedigree_function':Diabetes_pedigree_function,'Age':Age,'class':result})

            pdf = FPDF()
            pdf.add_page()
		
            page_width = pdf.w - 2 * pdf.l_margin
		
            pdf.set_font('Times','B',14.0) 
            pdf.cell(page_width, 0.0, 'DIABETES PREDICTION SYSTEM', align='C')
            

            pdf.ln(10)

            pdf.set_font('Courier', 'B', 12)
		
            col_width = page_width/4
		
            pdf.ln(1)
		
            th = pdf.font_size
            #dt = datetime.now()
            today = _datetime.date.today()
            pdf.image("/home/saurabh/Desktop/SEM8/DPS/static/images"+'/'+"img1"+'.jpg',w=50,h=50)
            #date = dt.date()
            pdf.cell(190, 10, ('Date: %s' % today),ln=1,align="R")
            pdf.cell(190, 10, txt="Patient Details:", ln=1, align="C")
            #pdf.cell(200, 10, txt="pregnant_times:", ln=1, align="C")
            pdf.cell(190, 10, ('Number of times Pregnant: %s' % pregnant_times),ln=1,align="L",border=1)
            pdf.cell(190, 10, ('Glucose Tolerance test: %s' % glucose_tolerance_test),ln=1,align="L",border=1)
            pdf.cell(190, 10, ('Diastolic_blood_pressure: %s' % Diastolic_blood_pressure),ln=1,align="L",border=1)
            pdf.cell(190, 10, ('Triceps_skin_fold_thickness: %s' % Triceps_skin_fold_thickness),ln=1,align="L",border=1)
            pdf.cell(190, 10, ('serum_insulin: %s' % serum_insulin),ln=1,align="L",border=1)
            pdf.cell(190, 10, ('Body_mass_index: %s' % Body_mass_index),ln=1,align="L",border=1)
            pdf.cell(190, 10, ('Diabetes_pedigree_function: %s' % Diabetes_pedigree_function),ln=1,align="L",border=1)
            pdf.cell(190, 10, ('Age: %s' % Age),ln=1,align="L",border=1)
            if result==1:
              pred="Patient suffers from Diabetes"
              pdf.cell(190, 10, ('Prediction: %s' % pred),ln=1,align="L",border=1)
              sugg="1.Check your blood sugar level frequently.\n2.Make healthy food choices.\n3.Exercise daily.\n4.Avoid smoking and alcohol consumption\n5.Take Medicines on time."
              pdf.multi_cell(190, 10, ('Suggestions: %s' % sugg),align="L",border=1)
              pdf.image("/home/saurabh/Desktop/SEM8/DPS/static/images"+'/'+"sol"+'.jpg',w=180,h=200)
            else:
              pred="Patient does not suffer from Diabetes"
              pdf.cell(190, 10, ('Prediction: %s' % pred),ln=1,align="L",border=1)
              sugg="1.Don't skip meals.\n2.Make healthy food choices.\n3.Exercise daily.\n4.Avoid smoking and alcohol consumption\n5.Lose weight if you are overweight."
              pdf.multi_cell(190, 10, ('Suggestions: %s' % sugg),align="L",border=1)
              pdf.image("/home/saurabh/Desktop/SEM8/DPS/static/images"+'/'+"prev"+'.jpeg',w=180,h=220)
            #pdf.cell(200, 10, txt=pregnant_times, ln=1, align="C")
            pdf.cell(190, 10, txt="Thank You!!", ln=1, align="C")
            #pdf.output("Report.pdf")
            session['files'] =  pdf.output("Report.pdf")
		
            pdf.ln(10)
		
            pdf.set_font('Times','',10.0) 
            pdf.cell(page_width, 0.0, '- end of report -', align='C')



        # And you return a text or a template, but if you don't return anything
        # this code will never work.
        return 'Thanks for your input!'
model = pickle.load(open('model.pkl', 'rb'))
def ValuePredictor(to_predict_list): 
	to_predict = np.array(to_predict_list).reshape(1, 8) 
	loaded_model = pickle.load(open("model.pkl", "rb")) 
	result = loaded_model.predict(to_predict) 
	return result 

@app.route('/predict', methods = ['POST']) 
def predict():
	#res= save_data()		 
	if request.method == 'POST': 
		to_predict_list = request.form.to_dict() 
		to_predict_list = list(to_predict_list.values()) 
		to_predict_list = list(map(float, to_predict_list)) 
		result = ValuePredictor(to_predict_list)		 
		if int(result)== 1: 
			prediction ='YES'
			res= save_data(result=1)		 
		else: 
			prediction ='NO'
			res= save_data(result=0)		 			
		return render_template("result.html",prediction_text = prediction) 
	return 'ok'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])

 
 
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global UPLOAD_FOLDER
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            session['filename'] = filename

            #UPLOAD_FOLDER = './upload_dir/'
            #CreateNewDir()
            #global UPLOAD_FOLDER
            file.save(filename)
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('dashboard.html')
           


@app.route('/uploaded', methods=['GET', 'POST'])
def uploaded_file():
    flash("File uploaded Successfully!!")
    return render_template("dashboard.html")


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)

