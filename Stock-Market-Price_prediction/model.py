import pandas_datareader as pdr
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


def predict_next_day(stock_name, algorithm, category):
    
    if algorithm == "lr":
        df = pdr.get_data_yahoo(stock_name)
        forecast_out = 1  # One day prediction
        df['Prediction'] = df[[category]].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LinearRegression().fit(X_train, y_train)
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        print(x_forecast)

        pred = model.predict(x_forecast)

        return f'{pred[0]: 0.2f}'
    
    elif algorithm == "dtr":
        df = pdr.get_data_yahoo(stock_name)
        forecast_out = 1  # One day prediction
        df['Prediction'] = df[[category]].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = DecisionTreeRegressor().fit(X_train, y_train)
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        print(x_forecast)

        pred = model.predict(x_forecast)

        return f'{pred[0]: 0.2f}'
    
    elif algorithm == "svr":
        df = pdr.get_data_yahoo(stock_name)
        forecast_out = 1  # One day prediction
        df['Prediction'] = df[[category]].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = SVR().fit(X_train, y_train)
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        print(x_forecast)

        pred = model.predict(x_forecast)

        return f'{pred[0]: 0.2f}'
    
    elif algorithm == "lassor":
        df = pdr.get_data_yahoo(stock_name)
        forecast_out = 1  # One day prediction
        df['Prediction'] = df[[category]].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        clf = linear_model.Lasso(alpha=0.1)

        model = clf.fit(X_train, y_train)
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        print(x_forecast)

        pred = model.predict(x_forecast)

        return f'{pred[0]: 0.2f}'
    
    elif algorithm == "rr":
        df = pdr.get_data_yahoo(stock_name)
        forecast_out = 1  # One day prediction
        df['Prediction'] = df[[category]].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = Ridge().fit(X_train, y_train)
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        print(x_forecast)

        pred = model.predict(x_forecast)

        return f'{pred[0]: 0.2f}'
    
    elif algorithm == "rfr":
        
        df = pdr.get_data_yahoo(stock_name)
        forecast_out = 1  # One day prediction
        df['Prediction'] = df[[category]].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestRegressor().fit(X_train, y_train)
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        print(x_forecast)

        pred = model.predict(x_forecast)

        return f'{pred[0]: 0.2f}'


#  https://plotly.com/python/range-slider/
def get_stock(stock_name, category):
    """
    Returns the current price and JSON object containing the Plotly graph.

    :precondition: category must be the string "open", "close", "high", or "low"
    :param stock_name: the name of the stock, as a string
    :param category: either the strings "open", "close", "high", or "low"
    :return:
    """
    df = pdr.get_data_yahoo(stock_name)
    current_price = f'{df[category][-1]: .2f}'
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=list(df.index), y=list(df[category])))
    fig.update_layout(autosize=False)

    # Set title
    fig.update_layout(
        title_text=f'{stock_name.upper()} {category} Stock Price',
    )

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True,
            ),
            type="date",
        )
    )
    start_date = datetime.datetime.now() - datetime.timedelta(30)
    initial_range = [
        start_date, datetime.datetime.now()
    ]

    fig['layout']['xaxis'].update(range=initial_range)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON, current_price



