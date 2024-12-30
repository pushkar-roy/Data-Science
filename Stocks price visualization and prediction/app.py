from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go

app = Flask(__name__)

def get_stock_data(stock_name, start_date, end_date):
    try:
        data = yf.download(stock_name + ".NS", start=start_date, end=end_date)
        data.reset_index(inplace=True)
        # Rename columns as per the requirement
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        print(f"Error fetching data for {stock_name}: {e}")
        return None

def predict_stock_prices(historical_data, days=30):
    historical_data['Day_Index'] = np.arange(len(historical_data))
    X = historical_data[['Day_Index']]
    y = historical_data['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = np.arange(len(historical_data), len(historical_data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    prediction_range = 0.02 * predictions  # Add a 2% variability range
    upper_bound = predictions + prediction_range
    lower_bound = predictions - prediction_range
    
    prediction_dates = pd.date_range(start=historical_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)
    return prediction_dates, predictions, upper_bound, lower_bound

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        comparison_stock = request.form.get('comparison_stock')
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        primary_data = get_stock_data(stock_name, start_date, end_date)
        comparison_data = None

        if comparison_stock:
            comparison_data = get_stock_data(comparison_stock, start_date, end_date)

        if primary_data is not None:
            # Find highest and lowest prices for primary stock
            highest_price_primary = primary_data['High'].max()
            lowest_price_primary = primary_data['Low'].min()

            # Historical Graph
            historical_fig = go.Figure()
            historical_fig.add_trace(go.Scatter(x=primary_data['Date'], y=primary_data['Close'],
                                                mode='lines', name=f'{stock_name} Close'))
            
            if comparison_data is not None:
                # Find highest and lowest prices for comparison stock
                highest_price_comparison = comparison_data['High'].max()
                lowest_price_comparison = comparison_data['Low'].min()
                
                historical_fig.add_trace(go.Scatter(x=comparison_data['Date'], y=comparison_data['Close'],
                                                    mode='lines', name=f'{comparison_stock} Close'))
            else:
                highest_price_comparison = None
                lowest_price_comparison = None

            historical_fig.update_layout(title='Historical Stock Prices', xaxis_title='Date', yaxis_title='Price')

            # Prediction Graph
            prediction_fig = go.Figure()

            # Predictions for primary stock
            prediction_dates, predictions, upper_bound, lower_bound = predict_stock_prices(primary_data)
            prediction_fig.add_trace(go.Scatter(x=primary_data['Date'], y=primary_data['Close'],
                                                mode='lines', name=f'{stock_name} Historical'))
            prediction_fig.add_trace(go.Scatter(x=prediction_dates, y=predictions, mode='lines',
                                                name=f'{stock_name} Predicted', line=dict(color='red')))
            prediction_fig.add_trace(go.Scatter(x=prediction_dates, y=upper_bound, mode='lines',
                                                name=f'{stock_name} Prediction Upper Bound', line=dict(dash='dot')))
            prediction_fig.add_trace(go.Scatter(x=prediction_dates, y=lower_bound, mode='lines',
                                                name=f'{stock_name} Prediction Lower Bound', line=dict(dash='dot')))

            # Predictions for comparison stock
            if comparison_data is not None:
                (comparison_prediction_dates, comparison_predictions,
                 comparison_upper_bound, comparison_lower_bound) = predict_stock_prices(comparison_data)
                prediction_fig.add_trace(go.Scatter(x=comparison_data['Date'], y=comparison_data['Close'],
                                                    mode='lines', name=f'{comparison_stock} Historical'))
                prediction_fig.add_trace(go.Scatter(x=comparison_prediction_dates, y=comparison_predictions,
                                                    mode='lines', name=f'{comparison_stock} Predicted', line=dict(color='blue')))
                prediction_fig.add_trace(go.Scatter(x=comparison_prediction_dates, y=comparison_upper_bound,
                                                    mode='lines', name=f'{comparison_stock} Prediction Upper Bound', line=dict(dash='dot')))
                prediction_fig.add_trace(go.Scatter(x=comparison_prediction_dates, y=comparison_lower_bound,
                                                    mode='lines', name=f'{comparison_stock} Prediction Lower Bound', line=dict(dash='dot')))

            historical_html = historical_fig.to_html(full_html=False)
            prediction_html = prediction_fig.to_html(full_html=False)

            return render_template(
                'index.html', 
                historical_plot=historical_html, 
                prediction_plot=prediction_html, 
                stock_name=stock_name, 
                comparison_stock=comparison_stock,
                highest_price_primary=highest_price_primary,
                lowest_price_primary=lowest_price_primary,
                highest_price_comparison=highest_price_comparison,
                lowest_price_comparison=lowest_price_comparison
            )
        else:
            return render_template('index.html', error_message="Error fetching data. Please check stock symbols and dates.")
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
