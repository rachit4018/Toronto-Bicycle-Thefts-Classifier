
from urllib import request
import joblib as jb
from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import json
import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

app = Flask(__name__)
ap = ""
name = ""
y_train =jb.load('y_train.pkl')
y_test = jb.load('y_test.pkl')
y_test1=""
accuracy = ""
y_test_predict = ""
@app.route("/", methods=["GET", "POST"])
def Fun_knn():
    return render_template("index.html")


@app.route("/sub", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        input_dict = request.form.to_dict()
        model = input_dict.pop('model')
        int_features = pd.DataFrame.from_dict([input_dict])
        #int_features = [int(x) for x in request.form.values()]
        #final = [np.array(int_features)]
        #model = final[26]
        #print(int_features)
        #print(final)

        if model == 'knn':
            loaded_model = jb.load('knn_model.pkl')
            y_test1=jb.load('knnYtest_pred.pkl')
            y_test_predict = jb.load('knn_pred.pkl')
            predictions = loaded_model.predict(int_features).tolist()                        
            accuracy = accuracy_score(y_test1,y_test_predict)
            output = predictions[0]
            print('Accuracy KNN: ',accuracy)
            if output == 0:
                return render_template("sub.html", prediction_text="Bike will not be recovered", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Happy to say, bike will be recovered", accuracy = accuracy,model = model)

        elif model == 'svm':
            loaded_model = jb.load('svm_model.pkl')
            y_test1=jb.load('SVM_Y_test.pkl')
            y_test_predict = jb.load('SVM_test_predict.pkl')
            predictions = loaded_model.predict(int_features).tolist()
            accuracy = accuracy_score(y_test1,y_test_predict)
            output = predictions[0]
            print('Accuracy SVM: ',accuracy)
            if output == 0:
                return render_template("sub.html", prediction_text="Bike will not be recovered",accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Happy to say, bike will be recovered", accuracy = accuracy,model = model)

        elif model == 'nn':
            loaded_model = jb.load('nn_model.pkl')
            predictions = loaded_model.predict(int_features).tolist()
            output = predictions[0]
            y_test_predict = jb.load('nn_y_pred.pkl')
            accuracy = accuracy_score(y_test,y_test_predict)
            if output == 0:
                return render_template("sub.html", prediction_text="Bike will not be recovered", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Happy to say, bike will be recovered", accuracy = accuracy,model = model)
        elif model == 'dt':
            loaded_model = jb.load('dct_model2.pkl')
            y_test_predict = jb.load('dct_y_pred.pkl')
            predictions = loaded_model.predict(int_features).tolist()
            
            output = predictions[0]
            accuracy = accuracy_score(y_test,y_test_predict)
            if output == 0:
                return render_template("sub.html", prediction_text="Bike will not be recovered", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Happy to say, bike will be recovered", accuracy = accuracy,model = model)
        elif model == 'rf':
            loaded_model = jb.load('rf_model.pkl')
            
            y_test_predict = jb.load('rf_y_pred.pkl')
            predictions = loaded_model.predict(int_features).tolist()
            
            output = predictions[0]
            accuracy = accuracy_score(y_test,y_test_predict)
            
            if output == 0:
                return render_template("sub.html", prediction_text="Bike will not be recovered", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Happy to say, bike will be recovered",accuracy = accuracy,model = model)
        elif model == 'lr':
            loaded_model = jb.load('logistic_model.pkl')
            y_test1= jb.load('lrYtest.pkl')
            y_test_predict = jb.load('lrYpred.pkl')
            predictions = loaded_model.predict(int_features).tolist()
            #print('Accuracy', accuracy_score(y_test, y_grid_pred))
            output = predictions[0]
            accuracy = accuracy_score(y_test1,y_test_predict)
            output = predictions[0]
            if output == 0:
                return render_template("sub.html", prediction_text="Bike will not be recovered", accuracy = accuracy,model = model)
            else:
                return render_template("sub.html", prediction_text="Happy to say, bike will be recovered", accuracy = accuracy,model = model)

if __name__=="__main__":
    app.run(debug=True) 
