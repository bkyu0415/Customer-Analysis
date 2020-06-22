
from Model_Performance.Model_Comparison import *
import itertools
import matplotlib.pyplot as plt
import seaborn as sns#visualization

import plotly.offline as py
py.init_notebook_mode(connected=True)#visualization
import warnings
warnings.filterwarnings("ignore")

# concat all models
model_performances = pd.concat([model1,model2,model3,
                                model4,model5,model6,
                                model7,model8,model9,
                                model10],axis = 0).reset_index()

model_performances = model_performances.drop(columns="index", axis=1)

##Confusion Matrice
lst    = [logit,logit_smote,decision_tree,knn,rfc,
          gnb,svc_lin,svc_rbf,xgc]

length = len(lst)

mods   = ['Logistic Regression(Baseline_model)','Logistic Regression(SMOTE)',
          'Decision Tree','KNN Classifier','Random Forest Classifier',"Naive Bayes",
          'SVM Classifier Linear','SVM Classifier RBF', 'XGBoost Classifier']

fig = plt.figure(figsize=(13,15))
fig.set_facecolor("#F3F3F3")
for i,j,k in itertools.zip_longest(lst,range(length),mods) :
    plt.subplot(4,3,j+1)
    predictions = i.predict(test_X)
    conf_matrix = confusion_matrix(predictions,test_Y)
    sns.heatmap(conf_matrix,annot=True,fmt = "d",square = True,
                xticklabels=["not churn","churn"],
                yticklabels=["not churn","churn"],
                linewidths = 2,linecolor = "w",cmap = "Set1")
    plt.title(k,color = "b")
    plt.subplots_adjust(wspace = .3,hspace = .3)
    print(conf_matrix)


##ROC Curve
plt.style.use("dark_background")
fig = plt.figure(figsize=(12,16))
fig.set_facecolor("#F3F3F3")
for i,j,k in itertools.zip_longest(lst,range(length),mods) :
    qx = plt.subplot(4,3,j+1)
    probabilities = i.predict_proba(test_X)
    predictions   = i.predict(test_X)
    fpr,tpr,thresholds = roc_curve(test_Y,probabilities[:,1])
    plt.plot(fpr,tpr,linestyle = "dotted",
             color = "royalblue",linewidth = 2,
             label = "AUC = " + str(np.around(roc_auc_score(test_Y,predictions),3)))
    plt.plot([0,1],[0,1],linestyle = "dashed",
             color = "orangered",linewidth = 1.5)
    plt.fill_between(fpr,tpr,alpha = .4)
    plt.fill_between([0,1],[0,1],color = "k")
    plt.legend(loc = "lower right",
               prop = {"size" : 12})
    qx.set_facecolor("k")
    plt.grid(True,alpha = .15)
    plt.title(k,color = "b")
    plt.xticks(np.arange(0,1,.3))
    plt.yticks(np.arange(0,1,.3))

##Precision Recall Curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

fig = plt.figure(figsize=(13, 17))
fig.set_facecolor("#F3F3F3")
for i, j, k in itertools.zip_longest(lst, range(length), mods):
    qx = plt.subplot(4, 3, j + 1)
    probabilities = i.predict_proba(test_X)
    predictions = i.predict(test_X)
    recall, precision, thresholds = precision_recall_curve(test_Y, probabilities[:, 1])
    plt.plot(recall, precision, linewidth=1.5,
             label=("avg_pcn : " +
                    str(np.around(average_precision_score(test_Y, predictions), 3))))
    plt.plot([0, 1], [0, 0], linestyle="dashed")
    plt.fill_between(recall, precision, alpha=.2)
    plt.legend(loc="lower left",
               prop={"size": 10})
    qx.set_facecolor("k")
    plt.grid(True, alpha=.15)
    plt.title(k, color="b")
    plt.xlabel("recall", fontsize=7)
    plt.ylabel("precision", fontsize=7)
    plt.xlim([0.25, 1])
    plt.yticks(np.arange(0, 1, .3))

plt.show()