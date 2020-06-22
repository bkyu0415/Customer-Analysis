from Univariate_Selection import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
from IPython.display import SVG, display

from sklearn.ensemble import RandomForestClassifier

# top 3 categorical features
features_cat = score[score["feature_type"] == "Categorical"]["features"][:3].tolist()

# top 3 numerical features
features_num = score[score["feature_type"] == "Numerical"]["features"][:3].tolist()

# function attributes
# columns  - column used
# nf_estimators   - The number of trees in the forest.
# estimated_tree  - tree number to be displayed
# maximum_depth   - depth of the tree
# criterion_type  - split criterion type ["gini" or "entropy"]
# Model performance - prints performance of model



def plot_tree_randomforest(columns, nf_estimators,
                           estimated_tree, maximum_depth,
                           criterion_type, model_performance=None):
    dataframe = df_telcom_og[columns + target_col].copy()

    # train and test datasets
    rf_x = dataframe[[i for i in columns if i not in target_col]]
    rf_y = dataframe[target_col]

    # random forest classifier
    rfc = RandomForestClassifier(n_estimators=nf_estimators,
                                 max_depth=maximum_depth,
                                 criterion=criterion_type,
                                 )
    rfc.fit(rf_x, rf_y)

    estimated_tree = rfc.estimators_[estimated_tree]

    graph = Source(tree.export_graphviz(estimated_tree, out_file=None,
                                        rounded=True, proportion=False,
                                        feature_names=columns,
                                        precision=2,
                                        class_names=["Not churn", "Churn"],
                                        filled=True))
    display(graph)

    # model performance
    if model_performance == True:
        telecom_churn_prediction(rfc,
                                 rf_x, test_X[columns],
                                 rf_y, test_Y,
                                 columns, "features", threshold_plot=True)


cols1 = [i for i in train_X.columns if i not in target_col + Id_col]
plot_tree_randomforest(cols1, 100, 99, 3, "entropy", True)