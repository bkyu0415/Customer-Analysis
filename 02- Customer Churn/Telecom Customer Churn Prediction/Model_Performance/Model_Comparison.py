from Model_Building import *

from sklearn.metrics import cohen_kappa_score

##Logistic Regression
logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                           verbose=0, warm_start=False)

logit_smote = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

from Model_Training.Model_Logistic_Regression.M3_RFE import *
##Classification Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(max_depth = 9,
                                       random_state = 123,
                                       splitter  = "best",
                                       criterion = "gini",
                                      )
rfc = RandomForestClassifier(n_estimators = 1000,
                             random_state = 123,
                             max_depth = 9,
                             criterion = "gini")
##KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
##Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(priors=None)
##Support Vector Machine
from sklearn.svm import SVC
svc_lin  = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)
svc_rbf  = SVC(C=1.0, kernel='rbf',
               degree= 3, gamma=1.0,
               coef0=0.0, shrinking=True,
               probability=True,tol=0.001,
               cache_size=200, class_weight=None,
               verbose=False,max_iter= -1,
               random_state=None)
##XGBoost
from xgboost import XGBClassifier

xgc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, learning_rate=0.9, max_delta_step=0,
                    max_depth = 7, min_child_weight=1, missing=None, n_estimators=100,
                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    silent=True, subsample=1)



# gives model report in dataframe
def model_report(model, training_x, testing_x, training_y, testing_y, name):
    model.fit(training_x, training_y)
    predictions = model.predict(testing_x)
    accuracy = accuracy_score(testing_y, predictions)
    recallscore = recall_score(testing_y, predictions)
    precision = precision_score(testing_y, predictions)
    roc_auc = roc_auc_score(testing_y, predictions)
    f1score = f1_score(testing_y, predictions)
    kappa_metric = cohen_kappa_score(testing_y, predictions)

    df = pd.DataFrame({"Model": [name],
                       "Accuracy_score": [accuracy],
                       "Recall_score": [recallscore],
                       "Precision": [precision],
                       "f1_score": [f1score],
                       "Area_under_curve": [roc_auc],
                       "Kappa_metric": [kappa_metric],
                       })
    return df


# outputs for every model
model1 = model_report(logit, train_X, test_X, train_Y, test_Y,
                      "Logistic Regression(Baseline_model)")
model2 = model_report(logit_smote, os_smote_X, test_X, os_smote_Y, test_Y,
                      "Logistic Regression(SMOTE)")
model3 = model_report(logit_rfe, train_rf_X, test_rf_X, train_rf_Y, test_rf_Y,
                      "Logistic Regression(RFE)")
model4 = model_report(decision_tree,train_X,test_X,train_Y,test_Y,
                      "Decision Tree")
model5 = model_report(rfc,train_X,test_X,train_Y,test_Y,
                      "Random Forest")
model6 = model_report(knn,os_smote_X,test_X,os_smote_Y,test_Y,
                      "KNN Classifier")
model7 = model_report(gnb,os_smote_X,test_X,os_smote_Y,test_Y,
                      "Naive Bayes")
model8 = model_report(svc_lin,os_smote_X,test_X,os_smote_Y,test_Y,
                      "SVM Classifier Linear")
model9 = model_report(svc_rbf,os_smote_X,test_X,os_smote_Y,test_Y,
                      "SVM Classifier RBF")

model10 = model_report(xgc,os_smote_X,test_X,os_smote_Y,test_Y,
                      "XGBoost Classifier")

#model11 = model_report(lgbm_c,os_smote_X,test_X,os_smote_Y,test_Y,"LGBM Classifier")

