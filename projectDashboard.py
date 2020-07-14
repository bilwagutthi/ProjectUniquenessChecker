# our models
from models import db ,Projects, Colleges

# Flask app files
from flask import Flask , redirect ,url_for , render_template

# Flask login

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import InputRequired, Email, Length
from flask_login import UserMixin, LoginManager

from flask_login import current_user, login_user , logout_user

from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bootstrap import Bootstrap
# Flask Admin Dashboard
from flask_admin import Admin,AdminIndexView
from flask_admin.contrib.sqla  import ModelView

# Creating the app
app = Flask(__name__)
app.config.from_pyfile('config.py')
app.config['FLASK_ADMIN_SWATCH'] = 'darkly'
bootstrap = Bootstrap(app)
# Initializing the up Database
db.init_app(app)
db.create_all(app=app)

# Creating a login system
login_manager = LoginManager(app)
login_manager.login_view = 'login'

userID=''
username=''
@login_manager.user_loader
def load_user(user_id):
    return Colleges.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('E-Mail', validators=[InputRequired(), Length(min=4, max=15), Email()])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=1, max=80)])
    remember = BooleanField('Remember me')

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = Colleges.query.filter_by(email=form.username.data).first()
        global userID
        global username
        userID=user
        print('\n\n\n\nUSER:',user,type(user),userID)
        username=Colleges.query.filter_by(email=form.username.data).with_entities(Colleges.name).first()
    
        if user:
            login_user(user, remember=form.remember.data)
            
            global userid
            userid=user
            return redirect('admin')
            passw= Colleges.query.filter_by(password=form.password.data).first()
            if passw:
                login_user(user, remember=form.remember.data)
                return redirect('admin')
            '''if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('admin'))'''
        return '<h1>Invalid username or password</h1>'
        #return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'
    return render_template('login.html', form=form)



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

class ProjIndexView(AdminIndexView):
    def is_accessible(self):
        return current_user.is_authenticated

# Admin Dashboard
class ProjectView(ModelView):
    global userID
    global username
   # column_exclude_list = ['colleges', ]
    

    
    form_choices = {'dept': [
                                ('CSE', 'CSE'),
                                ('IT', 'IT'),
                                ('CV', 'CV'),
                                ('ME', 'po'), ],
                    'collegeID': [(userID,username),]
                    }
    

   
    def is_accessible(self):
        return current_user.is_authenticated

    def inaccessible_callback(self,name,**kwargs):
        return redirect(url_for('login'))




admin = Admin(app, name='Projects Dashboard', template_mode='bootstrap3', template='admin/myhome.html',
        url='/)
admin.add_view(ProjectView(Projects,db.session))

if __name__ == '__main__':
    app.run(debug=True )
    
    