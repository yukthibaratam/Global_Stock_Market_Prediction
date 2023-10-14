from flask import Flask, render_template, request
import pandas_datareader as pdr
from pandas_datareader._utils import RemoteDataError
from model import get_stock, predict_next_day
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import json
import plotly
import datetime


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_name = request.form['stock-name'].replace(' ', '')
        algorithm=request.form['algo'].replace(' ','')
        if algorithm != "lr" and algorithm != "rfr" and algorithm != "dtr" and algorithm != "svr" and algorithm != "lassor" and algorithm != "rr" and algorithm != "cmp":
            error = "Select The Algorithm"
            return render_template("index.html", error=error, graph_container_style="display: none;",chart_style="display: none;")
        try:
            date = str(pdr.get_data_yahoo(stock_name).index[-1]).split(' ')[0]
        except (TypeError, KeyError, RemoteDataError):
            error = "Stock name does not exist"
            return render_template("index.html", error=error, graph_container_style="display: none;",chart_style="display: none;")
            
            
        else:
            graphJSON_open, open_price = get_stock(stock_name, 'Open')
            graphJSON_close, close_price = get_stock(stock_name, 'Close')
            graphJSON_high, high_price = get_stock(stock_name, 'High')
            graphJSON_low, low_price = get_stock(stock_name, 'Low')

            if algorithm == "cmp":
                category = "Close"

                df = pdr.get_data_yahoo(stock_name)
                forecast_out = 1  # One day prediction
                df['Prediction'] = df[[category]].shift(-forecast_out)

                X = np.array(df.drop(['Prediction'], 1))
                X = X[:-forecast_out]
                y = np.array(df['Prediction'])
                y = y[:-forecast_out]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

                lr=LinearRegression()
                lrmodel = lr.fit(X_train, y_train)
                lracc = lr.score(X_test,y_test)

                rf=RandomForestRegressor()
                rfrmodel = rf.fit(X_train, y_train)
                rfracc = rf.score(X_test,y_test)

                dt= DecisionTreeRegressor()
                dtrmodel = dt.fit(X_train, y_train)
                dtracc = dt.score(X_test,y_test)

                sv=SVR()
                svrmodel = sv.fit(X_train, y_train)
                svracc = sv.score(X_test,y_test)

                clf = linear_model.Lasso(alpha=0.1)
                lassormodel = clf.fit(X_train, y_train)
                lassoracc=clf.score(X_test,y_test)

                rra=Ridge()
                rrmodel= rra.fit(X_train, y_train)
                rracc= rra.score(X_test,y_test)

                acc = [lracc,rfracc,dtracc,svracc,lassoracc,rracc]
                labels = ["lr","rfr","dtr","svr","lasso","ridge"]
                bar_labels=labels
                bar_values=acc
                return render_template('index.html',
                                      title='comparision',
                                      max=1, labels=bar_labels,
                                      values=bar_values,
                                      graph_container_style="display: none;",
                                      chart_style="display: block;")

            else:
                open_prediction = predict_next_day(stock_name,algorithm, 'Open')
                close_prediction = predict_next_day(stock_name,algorithm, 'Close')
                high_prediction = predict_next_day(stock_name,algorithm, 'High')
                low_prediction = predict_next_day(stock_name,algorithm, 'Low')

            return render_template("index.html",
                                   stock_name=stock_name,
                                   graphJSON_open=graphJSON_open,
                                   graphJSON_close=graphJSON_close,
                                   graphJSON_high=graphJSON_high,
                                   graphJSON_low=graphJSON_low,
                                   graph_container_style="display: block;",
                                   stock_date=date,
                                   open_price=open_price,
                                   close_price=close_price,
                                   high_price=high_price,
                                   low_price=low_price,
                                   open_prediction=open_prediction,
                                   close_prediction=close_prediction,
                                   high_prediction=high_prediction,
                                   low_prediction=low_prediction)
    else:
        return render_template('index.html', graph_container_style="display: none;",chart_style="display: none;")


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
