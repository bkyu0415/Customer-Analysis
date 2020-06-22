from Univariate_Selection import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
from IPython.display import SVG, display

# top 3 categorical features
features_cat = score[score["feature_type"] == "Categorical"]["features"][:3].tolist()

# top 3 numerical features
features_num = score[score["feature_type"] == "Numerical"]["features"][:3].tolist()


# Function attributes
# columns        - selected columns
# maximum_depth  - depth of tree
# criterion_type - ["gini" or "entropy"]
# split_type     - ["best" or "random"]
# Model Performance - True (gives model output)

def plot_decision_tree(columns, maximum_depth, criterion_type,
                       split_type, model_performance=None):
    # separating dependent and in dependent variables
    dtc_x = df_x[columns]
    dtc_y = df_y[target_col]

    # model
    dt_classifier = DecisionTreeClassifier(max_depth=maximum_depth,
                                           splitter=split_type,
                                           criterion=criterion_type,
                                           )
    dt_classifier.fit(dtc_x, dtc_y)

    # plot decision tree
    graph = Source(tree.export_graphviz(dt_classifier, out_file=None,
                                        rounded=True, proportion=False,
                                        feature_names=columns,
                                        precision=2,
                                        class_names=["Not churn", "Churn"],
                                        filled=True
                                        )
                   )

    # model performance
    if model_performance == True:
        telecom_churn_prediction(dt_classifier,
                                 dtc_x, test_X[columns],
                                 dtc_y, test_Y,
                                 columns, "features", threshold_plot=True)
    display(graph)


#plot_decision_tree(features_num, 3, "gini", "best")
#plot_decision_tree(features_cat,3,"entropy","best",model_performance = True)

#using contract,tenure and paperless billing variables
columns = ['tenure','Contract_Month-to-month', 'PaperlessBilling',
           'Contract_One year', 'Contract_Two year']

plot_decision_tree(columns,3,"gini","best",model_performance= True)