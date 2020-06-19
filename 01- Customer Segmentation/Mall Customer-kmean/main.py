import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
#import warnings
import os
#warnings.filterwarnings("ignore")
#py.offline.init_notebook_mode(connected = True)
#print(os.listdir("../input"))

#EDA
df = pd.read_csv(r'Mall_Customers.csv')
def eda():
    print(df.head())
    print(df.describe())
    print(df.dtypes)
    print(df.isnull().sum())
    print(df)
#Data Visualization
def viz():
    #Histograms
    plt.style.use('fivethirtyeight')
    plt.figure(1 , figsize = (15 , 6))
    n = 0
    for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(1 , 3 , n)
        plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
        sns.distplot(df[x] , bins = 20)
        plt.title('Distplot of {}'.format(x))
    #plt.show()
    #Count Plot of Gender
    plt.figure(1 , figsize = (15 , 5))
    sns.countplot(y = 'Gender' , data = df)
    #plt.show()
    #Relation between Age , Annual Income and Spending Score
    plt.figure(1 , figsize = (15 , 7))
    n = 0
    for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
            n += 1
            plt.subplot(3 , 3 , n)
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
            sns.regplot(x = x , y = y , data = df)
            plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
    #plt.show()
    #Age vs Annual Income w.r.t Gender
    plt.figure(1 , figsize = (15 , 6))
    for gender in ['Male' , 'Female']:
        plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender] ,
                    s = 200 , alpha = 0.5 , label = gender)
    plt.xlabel('Age'), plt.ylabel('Annual Income (k$)')
    plt.title('Age vs Annual Income w.r.t Gender')
    plt.legend()
    #plt.show()
    #Annual Income vs Spending Score w.r.t Gender
    plt.figure(1 , figsize = (15 , 6))
    for gender in ['Male' , 'Female']:
        plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                    data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
    plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)')
    plt.title('Annual Income vs Spending Score w.r.t Gender')
    plt.legend()
    #plt.show()
    #Distribution / Boxplots
    plt.figure(1 , figsize = (15 , 7))
    n = 0
    for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(1 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')
        sns.swarmplot(x = cols , y = 'Gender' , data = df)
        plt.ylabel('Gender' if n == 1 else '')
        plt.title('Boxplots & Swarmplots' if n == 2 else '')
    plt.show()
#Model Training - Kmean
#Choice of K
X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300,
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(X3)
    inertia.append(algorithm.inertia_)

#Model
algorithm = (KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300,
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X3)
labels3 = algorithm.labels_
centroids3 = algorithm.cluster_centers_

#viz
df['label3'] =  labels3
trace1 = go.Scatter3d(
    x= df['Age'],
    y= df['Spending Score (1-100)'],
    z= df['Annual Income (k$)'],
    mode='markers',
     marker=dict(
        color = df['label3'],
        size= 20,
        line=dict(
            color= df['label3'],
            width= 12
        ),
        opacity=0.8
     )
)
data = [trace1]
layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
        )
)
fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig)