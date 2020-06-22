from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score
from yellowbrick.classifier import DiscriminationThreshold

from Data_Manipulation import *
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
import plotly.offline as py#visualization


# splitting train and test data
train, test = train_test_split(telcom, test_size=.25, random_state=111)

##seperating dependent and independent variables
cols = [i for i in telcom.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X = test[cols]
test_Y = test[target_col]


# Function attributes
# dataframe     - processed dataframe
# Algorithm     - Algorithm used
# training_x    - predictor variables dataframe(training)
# testing_x     - predictor variables dataframe(testing)
# training_y    - target variable(training)
# training_y    - target variable(testing)
# cf - ["coefficients","features"](cooefficients for logistic
# regression,features for tree based models)

# threshold_plot - if True returns threshold plot for model

from imblearn.over_sampling import SMOTE

cols    = [i for i in telcom.columns if i not in Id_col+target_col]

smote_X = telcom[cols]
smote_Y = telcom[target_col]

#Split train and test data
smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(smote_X,smote_Y,
                                                                         test_size = .25 ,
                                                                         random_state = 111)

#oversampling minority class using smote
os = SMOTE(random_state = 0)
os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)
os_smote_X = pd.DataFrame(data = os_smote_X,columns=cols)
os_smote_Y = pd.DataFrame(data = os_smote_Y,columns=target_col)
###


def telecom_churn_prediction(algorithm, training_x, testing_x,
                             training_y, testing_y, cols, cf, threshold_plot):
    # model
    algorithm.fit(training_x, training_y)
    predictions = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    # coeffs
    if cf == "coefficients":
        coefficients = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features":
        coefficients = pd.DataFrame(algorithm.feature_importances_)

    column_df = pd.DataFrame(cols)
    coef_sumry = (pd.merge(coefficients, column_df, left_index=True,
                           right_index=True, how="left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)

    print(algorithm)
    print("\n Classification report : \n", classification_report(testing_y, predictions))
    print("Accuracy   Score : ", accuracy_score(testing_y, predictions))
    # confusion matrix
    conf_matrix = confusion_matrix(testing_y, predictions)
    # roc_auc_score
    model_roc_auc = roc_auc_score(testing_y, predictions)
    print("Area under curve : ", model_roc_auc, "\n")
    fpr, tpr, thresholds = roc_curve(testing_y, probabilities[:, 1])

    # plot confusion matrix
    trace1 = go.Heatmap(z=conf_matrix,
                        x=["Not churn", "Churn"],
                        y=["Not churn", "Churn"],
                        showscale=False, colorscale="Picnic",
                        name="matrix")

    # plot roc curve
    trace2 = go.Scatter(x=fpr, y=tpr,
                        name="Roc : " + str(model_roc_auc),
                        line=dict(color=('rgb(22, 96, 167)'), width=2))
    trace3 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color=('rgb(205, 12, 24)'), width=2,
                                  dash='dot'))

    # plot coeffs
    trace4 = go.Bar(x=coef_sumry["features"], y=coef_sumry["coefficients"],
                    name="coefficients",
                    marker=dict(color=coef_sumry["coefficients"],
                                colorscale="Picnic",
                                line=dict(width=.6, color="black")))

    # subplots
    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                            subplot_titles=('Confusion Matrix',
                                            'Receiver operating characteristic',
                                            'Feature Importances'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 2)
    fig.append_trace(trace4, 2, 1)

    fig['layout'].update(showlegend=False, title="Model performance",
                         autosize=False, height=900, width=800,
                         plot_bgcolor='rgba(240,240,240, 0.95)',
                         paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195))
    fig["layout"]["xaxis2"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid=True, tickfont=dict(size=10),
                                        tickangle=90))
    py.plot(fig)

    if threshold_plot == True:
        visualizer = DiscriminationThreshold(algorithm)
        visualizer.fit(training_x, training_y)
        visualizer.poof()


def telecom_churn_prediction_alg(algorithm, training_x, testing_x,
                                 training_y, testing_y, threshold_plot=True):
    # model
    algorithm.fit(training_x, training_y)
    predictions = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)

    print(algorithm)
    print("\n Classification report : \n", classification_report(testing_y, predictions))
    print("Accuracy Score   : ", accuracy_score(testing_y, predictions))
    # confusion matrix
    conf_matrix = confusion_matrix(testing_y, predictions)
    # roc_auc_score
    model_roc_auc = roc_auc_score(testing_y, predictions)
    print("Area under curve : ", model_roc_auc)
    fpr, tpr, thresholds = roc_curve(testing_y, probabilities[:, 1])

    # plot roc curve
    trace1 = go.Scatter(x=fpr, y=tpr,
                        name="Roc : " + str(model_roc_auc),
                        line=dict(color=('rgb(22, 96, 167)'), width=2),
                        )
    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color=('rgb(205, 12, 24)'), width=2,
                                  dash='dot'))

    # plot confusion matrix
    trace3 = go.Heatmap(z=conf_matrix, x=["Not churn", "Churn"],
                        y=["Not churn", "Churn"],
                        showscale=False, colorscale="Blues", name="matrix",
                        xaxis="x2", yaxis="y2"
                        )

    layout = go.Layout(dict(title="Model performance",
                            autosize=False, height=500, width=800,
                            showlegend=False,
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            xaxis=dict(title="false positive rate",
                                       gridcolor='rgb(255, 255, 255)',
                                       domain=[0, 0.6],
                                       ticklen=5, gridwidth=2),
                            yaxis=dict(title="true positive rate",
                                       gridcolor='rgb(255, 255, 255)',
                                       zerolinewidth=1,
                                       ticklen=5, gridwidth=2),
                            margin=dict(b=200),
                            xaxis2=dict(domain=[0.7, 1], tickangle=90,
                                        gridcolor='rgb(255, 255, 255)'),
                            yaxis2=dict(anchor='x2', gridcolor='rgb(255, 255, 255)')
                            )
                       )
    data = [trace1, trace2, trace3]
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)

    if threshold_plot == True:
        visualizer = DiscriminationThreshold(algorithm)
        visualizer.fit(training_x, training_y)
        visualizer.poof()

