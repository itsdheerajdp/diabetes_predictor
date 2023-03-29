from flask import Flask, render_template,request,redirect
import pickle
app=Flask(__name__)
#load the model
model=pickle.load(open("model.pkl", "rb"))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictor',methods=["GET","POST"])
def predictor():
    if(request.method=="POST"):
        preg=request.form["pregnancies"]
        gluc=request.form["glucose"]
        bp=request.form["bp"]
        skn=request.form["sknthickness"]
        ins=request.form["insulin"]
        bmi=request.form["bmi"]
        age=request.form["age"]
        prediction=model.predict([[preg,gluc,bp,skn,ins,bmi,age]])
        return render_template('prediction.html',predict=prediction)
    else:
         return render_template('predictor.html')
if __name__=="__main__":
    app.run(debug=True)
