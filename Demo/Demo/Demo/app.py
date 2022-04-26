from urllib import request
from flask import Flask ,render_template, request

import model as m

app= Flask(__name__)
ap = ""
name= ""

@app.route("/", methods = ["GET","POST"])
def Fun_knn():            
    return render_template("index.html")


@app.route("/sub", methods = ["GET","POST"])
def submit():
    global ap
    global name
    if request.method == "POST":
        mod = request.form['mod']
        name = mod
        if mod == "knn":
            acc_pred  = m.knnModel()
            ap= acc_pred
           
        elif mod == "svm":
            acc_pred  = m.SVMModel()
            ap= acc_pred
        elif mod == "nn":
            acc_pred  = m.NeuralNetworksModel()
            ap= acc_pred
        elif mod == "logistic":
            acc_pred  = m.LogisticRegressionModel()
            ap = acc_pred
              
    return render_template("sub.html", mod_acc= ap, mod_name= name) 

if __name__=="__main__":
    app.run(debug=True)     