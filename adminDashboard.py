# Our models
from models import db ,Projects, Colleges
from variables import WORD_VEC_MODEL , DIMENSIONS

# flask , flask admin
from flask import Flask, render_template, request, redirect , url_for, jsonify

from flask_admin import Admin , BaseView , expose, AdminIndexView
from flask_admin.contrib.sqla  import ModelView
from flask_bootstrap import Bootstrap

# Re-training word2vec
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords , preprocess_string
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors

# Flask app + database
app = Flask(__name__)
app.config.from_pyfile('config.py')
app.config['FLASK_ADMIN_SWATCH'] = 'superhero'
db.init_app(app)
bootstrap = Bootstrap(app)

# Flask admin dashboard


class ProjectModelView(ModelView):
    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True
    
class CollegesModelView(ModelView):
    column_exclude_list = ['password', ]  
    form_excluded_columns = ['projects']  

class RetrainView(BaseView):
    @expose('/',methods=['GET','POST'])
    def index(self):
        if request.method=='GET':
            print('in get method')
            arg1='hello'
            return self.render('retrain.html',arg1=arg1)
        if request.method=='POST':
            print('Post method ')
            result={'wordcount':20}
            return jsonify(result)

admin = Admin(app, name='Admin Dashboard', template_mode='bootstrap3')
admin.add_view(CollegesModelView(Colleges,db.session))
admin.add_view( ProjectModelView(Projects,db.session))
admin.add_view(RetrainView(name='Re-train'))

@app.route('/')
def home():
    return redirect('admin')

# Re-training model !
def retrain():
    with app.app_context():
        temp=Projects.query.with_entities(Projects.title).all()
        titles=[i[0] for i in temp]
        temp=Projects.query.with_entities(Projects.abstract).all()
        abstracts=[i[0] for i in temp]

        msrcsv='MetaData/'+'MSRTrainData.csv'
        leecsv='MetaData/'+'LeeDocSimTrain.csv'
        tit_df=pd.read_csv(msrcsv, error_bad_lines=False)
        abs_df=pd.read_csv(leecsv, error_bad_lines=False)
        word_model = Word2VecKeyedVectors.load("MetaData/"+WORD_VEC_MODEL)
        new_words_list=[]
        for index,row in tit_df.iterrows():
            for i in [row['Sentence1'],row['Sentence2']]:
                new_words_list.append(preprocess_string( remove_stopwords(i)))
                    
        for index,row in abs_df.iterrows():
            for i in [row['Document1'],row['Document2']]:
                new_words_list.append(preprocess_string( remove_stopwords(i)))

        for i in titles:new_words_list.append(preprocess_string( remove_stopwords(i)))
        for i in abstracts:new_words_list.append(preprocess_string( remove_stopwords(i)))

        new_model = Word2Vec(new_words_list, size=DIMENSIONS, window=5, min_count=1, workers=4)
        word_vecs=[]
        words=[]
        for lis in new_words_list:
            for word in lis:
                words.append(word)
                word_vecs.append(new_model.wv[word])
        word_model.add(words,word_vecs,replace=False)
        word_model.save("MetaData/"+WORD_VEC_MODEL)



if __name__ == '__main__':
    app.run()
