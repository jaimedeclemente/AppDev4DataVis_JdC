# AppDev4DataVis_JdC

This repository contains an interactive financial data visualization app developed using Flask, Plotly, and other Python libraries. The app provides insights into stock data, financials, and forecasting for a given ticker symbol.

## Contents

1. **Presentaciones**: This directory includes a PowerPoint presentation explaining the app's features and functionalities.

2. **Prueba_Intertrimestral**: This directory contains code, PDF files, and necessary data for the midterm project.

3. **Trabajo_Final**: This directory includes the main app. The app allows users to explore stock data, financials, and predictions for a given ticker symbol.

## App Functionality

The app consists of three main tabs:

### 1. Home ("/")

- Enter a ticker symbol to view information about the company.
- Displays company details, such as market cap, full-time employees, and key financial officers.

### 2. Financials ("/financials")

- Enter a ticker symbol to view financial information.
- Displays an interactive bar chart of Free Cash Flow over time.
- Presents a waterfall plot illustrating changes in cash position over selected periods.

### 3. Stock ("/stock")

- Enter a ticker symbol to view stock-related information.
- Displays a time series plot of stock prices with various moving averages.
- Utilizes the Prophet library for stock price forecasting.
- Includes SARIMA (Seasonal Autoregressive Integrated Moving Average) models for predicting future stock prices.

### General Application Information & Recommendations

 - Unless another ticker is entered, the ticker is maintained when changing tabs so that it is not necessary to enter a new one every time a tab is changed.
 - In the stock data tab, choose your preferred moving averages and get rid of the other ones by clicking on the legend entry.

## Running the App

### Prerequisites

Before running the app, ensure you have the required Python packages installed. You can install them using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```
1. Navigate to the "Trabajo_Final" directory.

2. Run the following command in the terminal: 

```bash
flask run
```
3. Open a web browser and go to http://127.0.0.1:5000/ to access the app.

***Note:*** Ensure that the versions specified in requirements.txt are compatible with your Python environment.