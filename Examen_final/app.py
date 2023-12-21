from flask import Flask, render_template, request, session, redirect, url_for

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import secrets

import sklearn.datasets 
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, mean_squared_error, mean_absolute_error,
                            silhouette_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans

import random

random.seed(10)

app = Flask(__name__)
app.secret_key = secrets.token_hex(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num = int(request.form['num'])
        session['current_num'] = num
        return redirect(url_for('index'))
        
    num = session.get('current_num', None)
    if num is not None:
        if num <= 6 and num >= 1:
            if num == 1:
                fig1 = get_fig1()
                return render_template('index.html', fig1=fig1)
            elif num == 2:
                fig1 = get_fig1()
                fig2 = get_fig2()
                return render_template('index.html', fig1=fig1, fig2=fig2)
            elif num == 3:
                fig1 = get_fig1()
                fig2 = get_fig2()
                fig3 = get_fig3()
                return render_template('index.html', fig1=fig1, fig2=fig2, fig3=fig3)
            elif num == 4:
                fig1 = get_fig1()
                fig2 = get_fig2()
                fig3 = get_fig3()
                fig4 = get_fig4()
                return render_template('index.html', fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4)
            elif num == 5:
                fig1 = get_fig1()
                fig2 = get_fig2()
                fig3 = get_fig3()
                fig4 = get_fig4()
                fig5 = get_fig5()
                return render_template('index.html', fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5)
            else:
                fig1 = get_fig1()
                fig2 = get_fig2()
                fig3 = get_fig3()
                fig4 = get_fig4()
                fig5 = get_fig5()
                fig6 = get_fig6()
                return render_template('index.html', fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6)
        else:
            error_message = f"Error fetching data for {num}. The number must be between 1 and 6."
            return render_template('index.html', error_message=error_message)

    return render_template('index.html')

df = pd.read_csv("./bank-full.csv", sep=";") 
    
def get_fig1():
   # Descriptivo 1: Distribucion de la educacion de los clientes
    # Create a Plotly figure for a pie chart
    explode = [0, 0.1, 0, 0]
    values = df.groupby('education').apply(len)/len(df)
    labels = values.index
    print(values)
    print(labels)

    fig1 = go.Figure()

    fig1.add_trace(go.Pie(
        labels=labels,
        values=values,
        textinfo='percent+label',
        hoverinfo='label+percent',
        pull=explode
    ))

    # Update layout
    fig1.update_layout(
        title='Distribution of clients by education level',
    )
    
    fig1_html = fig1.to_html(full_html=False) 
    
    return fig1_html

def get_fig2():
    # Descriptivo 2: Distribucion etaria de los clientes
    # Create histograms using Plotly
    histogram = go.Histogram(x=df['age'], nbinsx=20, opacity=0.7, name='Clients age')

    # Create a Plotly figure
    fig2 = go.Figure(data=histogram)

    # Update layout
    fig2.update_layout(
        title='Distribution of clients by age',
        xaxis=dict(title='Age'),
        yaxis=dict(title='Num. of clients'),
        showlegend=True
    )
    
    fig2_html = fig2.to_html(full_html=False) 
    
    return fig2_html

def get_fig3():
    # Descriptivo 3: Balance según trabajo
    df['id'] = range(0, len(df))

    data = []
    for job in df['job'].unique():
        df_group = df[df['job'] == job]
        trace = go.Scatter(x=df_group['id'], 
                            y=df_group['balance'],
                            mode='markers',
                            name=job)
        data.append(trace)

    # Layout of the plot
    layout = go.Layout(
        title='Balance in account by job',
        xaxis=dict(title='Client ID'),
        yaxis=dict(title='Balance in account'),
        showlegend=True
    )
    fig3 = go.Figure(data=data, layout=layout)
    
    fig3_html = fig3.to_html(full_html=False) 
    
    return fig3_html

def get_fig4():
    # Descriptivo 4: Porcentaje de clientes por trabajo
    fig4 = go.Figure(go.Bar(
        x=df['job'].unique(),
        y=df.groupby('job').apply(len)/len(df),
        marker_color='mediumseagreen'
    ))

    # Update layout
    fig4.update_layout(
        title='Share of clients by job',
        xaxis=dict(title='Type of job'),
        yaxis=dict(title='Percentage of clients'),
        bargap=0.2,
    )
    
    fig4_html = fig4.to_html(full_html=False) 
    
    return fig4_html

def get_fig5():
    # Descriptivo 5: Predicción de si cogen o no el servicio
    # Dividir train-test
    for col in df.columns:
        if df[col].dtype != 'int64':
            #for i in range(len(df[col].unique())):
            #    df.loc[df[col]==df[col].unique()[i], col] = i
            #print(df[col].dtype)
            #df = df.drop(col, axis=1)
            df[col] = pd.get_dummies(df[col])
    X = df.drop(['y', 'id'], axis=1)
    #X = df[['age', 'balance', 'day', 'duration']]
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.3, random_state = 123)
    
    # Entrenamiento del modelo
    svm = LinearSVC(C = 0.1)
    svm.fit(X_train, y_train)
    
    #Evaluamos la clasificacion
    predictions = svm.predict(X_test)
    print(classification_report(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions)

    class_names = [0, 1]

    # Create annotated heatmap
    heatmap = ff.create_annotated_heatmap(z=cm,
                                        x=class_names,
                                        y=class_names,
                                        colorscale='Viridis')

    # Update layout
    heatmap.update_layout(title='Confusion Matrix',
                        xaxis=dict(title='Predicted service'),
                        yaxis=dict(title='True service'))
    
    fig5_html = heatmap.to_html(full_html=False) 
    
    return fig5_html

def get_fig6():
    # Descriptivo 6: Matriz de correlacion
    # Create a correlation matrix
    correlation_matrix = df.corr().round(2)

    # Extract column names
    columns = correlation_matrix.columns.tolist()

    # Create a Plotly heatmap
    heatmap = ff.create_annotated_heatmap(z=correlation_matrix.values,
                                        x=columns,
                                        y=columns,
                                        colorscale='Viridis')

    # Update layout
    heatmap.update_layout(
        title='Correlation Heatmap',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Features'),
    )
    
    fig6_html = heatmap.to_html(full_html=False) 
    
    return fig6_html

if __name__ == '__main__':
    app.run(debug=True)
