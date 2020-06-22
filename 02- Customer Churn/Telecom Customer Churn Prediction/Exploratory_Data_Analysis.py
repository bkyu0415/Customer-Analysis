import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
import os
import matplotlib.pyplot as plt#visualization
#from PIL import  Image
#%matplotlib inline
import pandas as pd
import seaborn as sns#visualization
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
#py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
telcom = pd.read_csv(r'WA_Fn-UseC_-Telco-Customer-Churn.csv')
#EDA
def efa():
    print(telcom.head)
    print(telcom.shape)
    print(telcom.describe)
    print(telcom.columns)
    print(telcom.isnull().sum().values.sum())
    print(telcom.nunique())


# Data Manipulation
# Replacing spaces with null values in total charges column
telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", np.nan)

# Dropping null values from total charges column which contain .15% missing data
telcom = telcom[telcom["TotalCharges"].notnull()]
telcom = telcom.reset_index()[telcom.columns]

# convert to float type
telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

# replace 'No internet service' to No for the following columns
replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in replace_cols:
    telcom[i] = telcom[i].replace({'No internet service': 'No'})

# replace values
telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1: "Yes", 0: "No"})


# Tenure to categorical column
def tenure_lab(telcom):
    if telcom["tenure"] <= 12:
        return "Tenure_0-12"
    elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24):
        return "Tenure_12-24"
    elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48):
        return "Tenure_24-48"
    elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60):
        return "Tenure_48-60"
    elif telcom["tenure"] > 60:
        return "Tenure_gt_60"


telcom["tenure_group"] = telcom.apply(lambda telcom: tenure_lab(telcom),
                                      axis=1)

# Separating churn and non churn customers
churn = telcom[telcom["Churn"] == "Yes"]
not_churn = telcom[telcom["Churn"] == "No"]

# Separating catagorical and numerical columns
Id_col = ['customerID']
target_col = ["Churn"]
cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols = [x for x in cat_cols if x not in target_col]
num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

## Data Processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# customer id col
Id_col = ['customerID']
# Target columns
target_col = ["Churn"]
# categorical columns
cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols = [x for x in cat_cols if x not in target_col]
# numerical columns
num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
# Binary columns with 2 values
bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
# Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

# Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols:
    telcom[i] = le.fit_transform(telcom[i])

# Duplicating columns for multi value columns
telcom = pd.get_dummies(data=telcom, columns=multi_cols)

# Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(telcom[num_cols])
scaled = pd.DataFrame(scaled, columns=num_cols)

# dropping original values merging scaled values for numerical columns
df_telcom_og = telcom.copy()
telcom = telcom.drop(columns=num_cols, axis=1)
telcom = telcom.merge(scaled, left_index=True, right_index=True, how="left")

## Variable Summary
summary = (df_telcom_og[[i for i in df_telcom_og.columns if i not in Id_col]].
           describe().transpose().reset_index())

summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)

val_lst = [summary['feature'], summary['count'],
           summary['mean'],summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]

trace  = go.Table(header = dict(values = summary.columns.tolist(),
                                line = dict(color = ['#506784']),
                                fill = dict(color = ['#119DFF']),
                               ),
                  cells  = dict(values = val_lst,
                                line = dict(color = ['#506784']),
                                fill = dict(color = ["lightgrey",'#F5F8FF'])
                               ),
                  columnwidth = [200,60,100,100,60,60,80,80,80])
layout = go.Layout(dict(title = "Variable Summary"))
figure = go.Figure(data=[trace],layout=layout)
py.plot(figure)

##Correlation Matrix
correlation = telcom.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)

#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = "Viridis",
                   colorbar   = dict(title = "Pearson Correlation coefficient",
                                     titleside = "right"
                                    ) ,
                  )

layout = go.Layout(dict(title = "Correlation Matrix for variables",
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )

data = [trace]
fig = go.Figure(data=data,layout=layout)
py.plot(fig)

##Visualzing PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X = telcom[[i for i in telcom.columns if i not in Id_col + target_col]]
Y = telcom[target_col + Id_col]

principal_components = pca.fit_transform(X)
pca_data = pd.DataFrame(principal_components,columns = ["PC1","PC2"])
pca_data = pca_data.merge(Y,left_index=True,right_index=True,how="left")
pca_data["Churn"] = pca_data["Churn"].replace({1:"Churn",0:"Not Churn"})

def pca_scatter(target,color) :
    tracer = go.Scatter(x = pca_data[pca_data["Churn"] == target]["PC1"] ,
                        y = pca_data[pca_data["Churn"] == target]["PC2"],
                        name = target,mode = "markers",
                        marker = dict(color = color,
                                      line = dict(width = .5),
                                      symbol =  "diamond-open"),
                        text = ("Customer Id : " +
                                pca_data[pca_data["Churn"] == target]['customerID'])
                       )
    return tracer

layout = go.Layout(dict(title = "Visualising data with principal components",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "principal component 1",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "principal component 2",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        height = 600
                       )
                  )
trace1 = pca_scatter("Churn",'red')
trace2 = pca_scatter("Not Churn",'royalblue')
data = [trace2,trace1]
fig = go.Figure(data=data,layout=layout)
py.plot(fig)


## separating binary columns
bi_cs = telcom.nunique()[telcom.nunique() == 2].keys()
dat_rad = telcom[bi_cs]


# plotting radar chart for churn and non churn customers(binary variables)
def plot_radar(df, aggregate, title):
    data_frame = df[df["Churn"] == aggregate]
    data_frame_x = data_frame[bi_cs].sum().reset_index()
    data_frame_x.columns = ["feature", "yes"]
    data_frame_x["no"] = data_frame.shape[0] - data_frame_x["yes"]
    data_frame_x = data_frame_x[data_frame_x["feature"] != "Churn"]

    # count of 1's(yes)
    trace1 = go.Scatterpolar(r=data_frame_x["yes"].values.tolist(),
                             theta=data_frame_x["feature"].tolist(),
                             fill="toself", name="count of 1's",
                             mode="markers+lines",
                             marker=dict(size=5)
                             )
    # count of 0's(No)
    trace2 = go.Scatterpolar(r=data_frame_x["no"].values.tolist(),
                             theta=data_frame_x["feature"].tolist(),
                             fill="toself", name="count of 0's",
                             mode="markers+lines",
                             marker=dict(size=5)
                             )
    layout = go.Layout(dict(polar=dict(radialaxis=dict(visible=True,
                                                       side="counterclockwise",
                                                       showline=True,
                                                       linewidth=2,
                                                       tickwidth=2,
                                                       gridcolor="white",
                                                       gridwidth=2),
                                       angularaxis=dict(tickfont=dict(size=10),
                                                        layer="below traces"
                                                        ),
                                       bgcolor="rgb(243,243,243)",
                                       ),
                            paper_bgcolor="rgb(243,243,243)",
                            title=title, height=700))

    data = [trace2, trace1]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)


# plot
plot_radar(dat_rad, 1, "Churn -  Customers")
plot_radar(dat_rad, 0, "Non Churn - Customers")

