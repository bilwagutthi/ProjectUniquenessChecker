import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'projects.db')
SECRET_KEY = 'testkey' # keep this key secret during production 
SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(DB_PATH)
SQLALCHEMY_TRACK_MODIFICATIONS = False
DEBUG = True
SQLALCHEMY_ECHO = True