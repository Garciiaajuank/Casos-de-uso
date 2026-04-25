from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from LogisticRegression import getMetrics
from RandomForest import getMetricsRF
import Clustering

app = Flask(__name__)

df = pd.read_csv("TSLA.csv")

X = df[['Open']]
y = df['Close']

model = LinearRegression()

model.fit(X,y)

def predictPrice(open_price):

    prediction = model.predict([[open_price]])

    return round(prediction[0],2)



@app.route('/FirstPage')
def firstpage(): 
    return render_template ('Case1.html')

@app.route('/SecondPage')
def secondpage(): 
    return render_template ('Case2.html')

@app.route('/ThirdPage')
def  thirdpage(): 
    return render_template ('Case3.html')

@app.route('/FourthPage')
def fourthpage(): 
    return render_template ('Case4.html')

@app.route('/')
def homePage(): 
    return render_template ('Page.html')

@app.route('/LinearRegression')
def predictPriceRoute(): 
    return render_template ('linearRegressionTesla.html')

@app.route('/LogisticRegression')
def logisticRegression(): 
    return render_template ('LogisticRegression.html')

@app.route('/RandomForest')
def randomForest(): 
    return render_template ('randomForest.html')

@app.route('/ExampleLO')
def exampleLogistic():

    accuracy,precision,recall,f1,auc = getMetrics()

    return render_template(

        "exampleLO.html",

        accuracy=accuracy,

        precision=precision,

        recall=recall,

        f1=f1,

        auc=auc

    )

@app.route('/ExampleLI', methods=["GET","POST"])
def examplepage():

    result = None
    open_price = None

    if request.method == "POST":

        open_price = float(request.form["open_price"])

        result = predictPrice(open_price)

        if not os.path.exists("static"):
            os.makedirs("static")

        predictions = model.predict(X)

        plt.figure(figsize=(8,5))

        plt.scatter(X,y,alpha=0.3,label="Real data")

        plt.plot(X,predictions,color="red",label="Regression line")

        plt.scatter(
            open_price,
            result,
            color="blue",
            s=120,
            label="Prediction"
        )

        plt.xlabel("Open Price")
        plt.ylabel("Close Price")

        plt.title("Tesla Linear Regression")

        plt.legend()

        plt.savefig("static/regression.png")

        plt.close()

    return render_template(
        "exampleLI.html",
        result=result,
        open_price=open_price
    )


@app.route('/ExampleRA')
def exampleRA():

    accuracy, precision, recall, f1, auc = getMetricsRF()

    return render_template(
        "exampleRA.html",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc
    )

@app.route('/Clustering')
def clustering():
    data = Clustering.applyClustering()
    return str(data["centers"])
