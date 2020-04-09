from flask import Flask, render_template, request, redirect , url_for
from main import main
app=Flask(__name__)

@app.route('/score',methods=['GET','POST'])
def score(title,abstract):
    if request.method=='POST':
        test_title=request.form['title']
        test_abstract=request.form['abstract']
        return result
    else:
        return "hi"
    result=main(title,abstract)
    return result


@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('score.html')

    if request.method=='POST':
        title = request.form.get('title') 
        abstract = request.form.get('abstract')
        result= main(title, abstract)
        score=result[0][0]
        response={'score':score}
        return jsonify(response)


if __name__=="__main__":
    app.run(debug=True)