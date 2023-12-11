from flask import Flask, render_template, request, session, redirect, url_for
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from io import BytesIO
import base64
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        session['current_ticker'] = ticker
        return redirect(url_for('index'))
        
    ticker = session.get('current_ticker', None)
    if ticker is not None:    
        stock_data = get_stock_data(ticker)
        info = yf.Ticker(ticker)

        try:
            company_info = info.info
            
            if 'companyOfficers' in company_info.keys():
                for officer in company_info['companyOfficers']:
                    if 'totalPay' in officer.keys():
                        formatted_number = "{:,}".format(officer['totalPay'])  # Format number with commas
                        officer['totalPay'] = f"{formatted_number} {company_info['financialCurrency']}"
            if 'fullTimeEmployees' in company_info.keys():       
                company_info['fullTimeEmployees'] =  "{:,}".format(company_info['fullTimeEmployees'])
            if 'marketCap' in company_info.keys():       
                formatted_number = "{:,}".format(company_info['marketCap'])  # Format number with commas
                company_info['marketCap'] = f"{formatted_number} {company_info['financialCurrency']}"

            return render_template('index.html', ticker=ticker, company_info=company_info)
        except:
            error_message = f"Error fetching data for {ticker}. Please check the ticker symbol or try again later."
            return render_template('index.html', error_message=error_message)

    return render_template('index.html')

@app.route('/financials', methods=['GET', 'POST'])
def financials():
    if request.method == 'POST':
        ticker = request.form['ticker']
        session['current_ticker'] = ticker
        return redirect(url_for('financials'))
        
    ticker = session.get('current_ticker', None) 
    if ticker is not None:   
        stock_data = get_stock_data(ticker)
        info = yf.Ticker(ticker)

        if info is not None:
            income_stmt = info.income_stmt
            cashflow = info.cashflow
            
            print(income_stmt)
            print(cashflow)
            
            fcfs = plot_cash_flow(cashflow)
            cash = plot_cash_position(cashflow)
            # icfs = plot_investing_cash_flow(cashflow)
            
            return render_template('financials.html', ticker=ticker, fcfs=fcfs, cash=cash)
        else:
            error_message = f"Error fetching data for {ticker}. Please check the ticker symbol or try again later."
            return render_template('financials.html', error_message=error_message)

    return render_template('financials.html')

@app.route('/stock', methods=['GET', 'POST'])
def stock():
    if request.method == 'POST':
        ticker = request.form['ticker']
        session['current_ticker'] = ticker
        return redirect(url_for('stock'))
        
    ticker = session.get('current_ticker', None)
    if ticker is not None:
        stock_data = get_stock_data(ticker)
        info = yf.Ticker(ticker)

        try:
            company_info = info.info
            
            model, forecast = fit_prophet_model(stock_data)
            
            # Get moving averages (10, 20, 50, 100, 200)
            series = stock_data
            series['10 day Moving Average'] = calculate_ma(stock_data, 10)
            series['20 day Moving Average'] = calculate_ma(stock_data, 20)
            series['50 day Moving Average'] = calculate_ma(stock_data, 50)
            series['100 day Moving Average'] = calculate_ma(stock_data, 100)
            series['200 day Moving Average'] = calculate_ma(stock_data, 200)
            
            series = series.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
            
            together_div = plot_prophet(ticker, series, model, forecast)
            
            best_aic = float('inf')
            best_order = None

            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        current_order = (p, d, q)
                        sarima_results = fit_sarima_model(stock_data, current_order)
                        current_aic = sarima_results.aic

                        if current_aic < best_aic:
                            best_aic = current_aic
                            best_order = current_order

            print(f"Best SARIMA Order: {best_order} with AIC: {best_aic}")

            # Fit the best SARIMA model
            best_sarima_results = fit_sarima_model(stock_data, best_order)

            # Get SARIMA residuals
            sarima_residuals = best_sarima_results.resid

            # Fit GARCH model on SARIMA residuals
            garch_results = fit_garch_model(sarima_residuals)

            # Plot SARIMA and GARCH results
            plot_garch = plot_sarima(ticker, series, best_sarima_results, garch_results)

            return render_template('stock.html', together_div=together_div, plot_garch=plot_garch, ticker=ticker)
        except:
            error_message = f"Error fetching data for {ticker}. Please check the ticker symbol or try again later."
            return render_template('stock.html', error_message=error_message)

    return render_template('stock.html')


def get_stock_data(ticker):
    try:
        # Fetch historical data
        stock_data = yf.download(ticker, period="10y")
        return stock_data
    except Exception as e:
        return None

# Function to fit SARIMA model
def fit_sarima_model(series, order):
    sarima_model = sm.tsa.SARIMAX(series['Close'], order=order, enforce_stationarity=False, enforce_invertibility=False)
    sarima_results = sarima_model.fit(disp=False)
    return sarima_results

# Function to fit GARCH model
def fit_garch_model(residuals):
    garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
    garch_results = garch_model.fit(disp='off')
    return garch_results


def fit_prophet_model(stock_data):
    df = stock_data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})

    # Create a Prophet model
    model = Prophet()
    
    # Add weekly seasonality with a period of 5 days
    model.add_seasonality(name='weekly_custom', period=5, fourier_order=5)
    
    # Fit the model to the historical data
    model.fit(df)

    # Create a dataframe for future dates
    future = model.make_future_dataframe(periods=30)  # Forecasting for the next 30 sessions
    future['initial'] = df['ds'].iloc[-1]  # Set the initial date to the last day in your historical data

    # Generate forecasts
    forecast = model.predict(future)
    
    return model, forecast

def calculate_ma(series, period):
    ma = series['Close'].rolling(window=period).mean()

    return ma

def plot_sarima(ticker, series, sarima_results, garch_results):
    # SARIMA forecast
    sarima_forecast = sarima_results.get_forecast(steps=30)
    sarima_mean = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    # GARCH forecast
    garch_forecast = garch_results.forecast(horizon=30)
    garch_volatility = np.sqrt(garch_forecast.variance.values[-1, :])

    last_day = series['Date'].iloc[-1]
    
    # Extract the last date and add one day
    start_date = last_day + pd.Timedelta(days=1)

    # Create a time series for the next 30 business days excluding weekends
    next_30_days = pd.date_range(start=start_date, periods=30, freq='B')
    
    # Melt dataframe to plot multiple lines together
    df1 = series.melt(id_vars='Date', var_name=ticker)

    # Plotting with Plotly
    fig = px.line(df1, 
                  x='Date', 
                  y='value', 
                  line_shape='linear',
                  title='Stock Price over Time (in local currency)', 
                  color=ticker
                  )

    fig.add_scatter(x=next_30_days, y=sarima_mean, mode='lines', name='SARIMA Forecast', line=dict(color='green'))
    fig.add_trace(go.Scatter(x=next_30_days, y=sarima_ci.iloc[:, 0], fill=None, mode='lines', line=dict(color='green'), showlegend=False))
    fig.add_trace(go.Scatter(x=next_30_days, y=sarima_ci.iloc[:, 1], fill='tonexty', mode='lines', line=dict(color='green'), name='SARIMA Confidence Interval', fillcolor='rgba(0,100,80,0.2)'))

    # Display AIC for SARIMA and log-likelihood for GARCH
    sarima_aic = sarima_results.aic
    # garch_log_likelihood = garch_results.loglikelihood

    aic_annotation = dict(x=0.02, y=0.95, xref='paper', yref='paper', text=f'SARIMA AIC: {sarima_aic}', showarrow=False, font=dict(color='black'))
    # log_likelihood_annotation = dict(x=0.02, y=0.90, xref='paper', yref='paper', text=f'GARCH Log-Likelihood: {garch_log_likelihood}', showarrow=False, font=dict(color='black'))

    fig.update_layout(annotations=[aic_annotation])

    plot_garch = fig.to_html(full_html=False)

    return plot_garch

def plot_prophet(ticker, series, model, forecast):    
    # Melt dataframe to plot multiple lines together
    series.reset_index(inplace=True)
    forecast = forecast.rename(columns={'ds':'Date', 'yhat': 'Prophet Prediction', 'yhat_upper': "Prophet Prediction's Upper Bound", 'yhat_lower': "Prophet Prediction's Lower Bound"})
    
    series = forecast[['Date', 'Prophet Prediction', "Prophet Prediction's Upper Bound", "Prophet Prediction's Lower Bound"]].merge(series, on='Date', how='outer')
    
    df1 = series.melt(id_vars='Date', var_name=ticker)
    
    fig = px.line(df1, 
                  x='Date', 
                  y='value', 
                  line_shape='linear',
                  title='Stock Price over Time (in local currency)', 
                  color = ticker
                  )
    # color_discrete_map={
    #                   'Close': 'maroon',
    #                   '10 day Moving Average': 'aliceblue',
    #                   '20 day Moving Average': 'beige',
    #                   '50 day Moving Average': 'green',
    #                   '100 day Moving Average': 'green',
    #                   '200 day Moving Average': 'green',
    #                   'Prophet Prediction': 'blue',
    #                   "Prophet Prediction's Upper Bound": 'red',
    #                   "Prophet Prediction's Lower Bound": 'red',
    #               }
    
    plot_div = fig.to_html(full_html=False)

    return plot_div

def plot_cash_flow(df):
    df = df.transpose()
    
    # Plotting Free Cash Flow over time
    fig = px.bar(df, x=df.index, y='Free Cash Flow', title='Free Cash Flow Over Time')
    return fig.to_html(full_html=False)

def plot_cash_position(df):
    df = df.transpose()
            
    new_df = pd.DataFrame(columns=['Change in Cash'])       
    for i in range(len(df.index)):
        if i == 0:
            # Ending cash balance for the first row
            new_df.loc[df.index[i]] = df['End Cash Position'][i]
        elif i == len(df.index)-1:
            # Beginning cash balance for the last row
            new_df.loc[df.index[i] - pd.DateOffset(years=1)] = df['Beginning Cash Position'][i]
        # Change in cash for subsequent rows
        new_df.loc[df.index[i] - pd.DateOffset(months=6)] = df['End Cash Position'][i] - df['Beginning Cash Position'][i]
    
    # Sort the index
    new_df.sort_index(inplace=True)
    
    # Plotting Cash Position over time
    fig = go.Figure(go.Waterfall(
        x=new_df.index,
        y=new_df['Change in Cash'],
        measure=['absolute', 'relative', 'relative', 'relative', 'relative', 'total'],  # 'relative' to create a waterfall plot
        connector={'line': {'color': 'rgba(63, 63, 63, 0.3)'}},
        increasing={'marker': {'color': 'green'}},
        decreasing={'marker': {'color': 'red'}},
    ))
    
    fig.update_layout(
        title='Cash Position Over Time',
        xaxis_title='Time',
        yaxis_title='Change in Cash',
    )
    
    return fig.to_html(full_html=False)

# def plot_investing_cash_flow(df):
#     df = df.transpose()
    
#     # Plotting Investing Cash Flow components
#     columns = ['Net Investment Purchase And Sale', 'Net Investment Properties Purchase And Sale',
#                'Net Business Purchase And Sale', 'Net Intangibles Purchase And Sale', 'Net PPE Purchase And Sale']

#     fig = px.bar(df, x=df.index, y=columns, title='Investing Cash Flow Components')
#     return fig.to_html(full_html=False)

if __name__ == '__main__':
    app.run(debug=True)
