from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class Colleges(db.Model,UserMixin):
    __tablename__='colleges'
    collegeID=db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(15), unique=True, nullable=False)
    name = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    address = db.Column(db.String(150), unique=True, nullable=False)
    phone = db.Column(db.String(120), unique=True, nullable=False)
    projects = db.relationship('Projects', backref='colleges')

    def __init__(self,username,password,name,email,address,phone):
        self.username=username
        self.password=password
        self.name=name
        self.email=email
        self.address=address
        self.phone=phone


    def is_active(self):
        """True, as all users are active."""
        return True

    def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        return self.collegeID

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return self.authenticated

    def is_anonymous(self):
        """False, as anonymous users aren't supported."""
        return False

class Projects(db.Model):
    __tablename__ = 'projects'
    projectID=db.Column(db.Integer, primary_key=True, autoincrement=True)
    title=db.Column(db.String(100),unique=True, nullable=False)
    abstract=db.Column(db.Text,unique=True, nullable=False)
    year=db.Column(db.Text, nullable=False)
    dept=db.Column(db.String(5), nullable=False)
    collegeID = db.Column(db.Integer, db.ForeignKey('colleges.collegeID'))

    def __init__(self,title,abstract,year,dept):
        self.title=title
        self.abstract=abstract
        self.year=year
        self.dept=dept
        



