from flask import Flask, render_template, request, redirect , url_for
from main import main
app=Flask(__name__)

@app.route('/score/<title><abstract>',)
def score(title,abstract):
    result=main(title,abstract)
    return result


@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        test_title=request.form['title']
        test_abstract=request.form['abstract']
        return hi #redirect(url_for('score',title=test_title,abstract=test_abstract))


if __name__=="__main__":
    app.run(debug=True)