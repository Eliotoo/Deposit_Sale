import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import export_graphviz


#Display settings for PyCharm
desired_width=420
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',50)

#import data
df_bank = pd.read_csv("bank-additional-full.csv")
df_bank = df_bank.drop(columns=['default','contact','day_of_week','duration','pdays','previous','nr.employed','month'])

#delete 'unkown' observations
df_bank = df_bank[df_bank.job != 'unknown']
df_bank = df_bank[df_bank.marital != 'unknown']
df_bank = df_bank[df_bank.education != 'unknown']
df_bank = df_bank[df_bank.housing != 'unknown']
df_bank = df_bank[df_bank.loan != 'unknown']
df_bank = df_bank[df_bank.education !='illiterate']

#delete emp.var.rate column
df_bank = df_bank.drop(columns=['emp.var.rate'])

#correct skewness - log+1 function
'''
age_log = np.log(df_bank.age + 1)
print(age_log.skew())
print(df_bank.age.skew())
df_bank = df_bank.assign(age_log=pd.Series(np.log(df_bank.age + 1)))
'''
#Create numerical categories
df_bank = df_bank.assign(Job_cat= np.where(df_bank['job'] == 'admin.',1,
                                           np.where(df_bank['job']=='blue-collar',2,
                                                    np.where(df_bank['job']=='technician', 3,
                                                            np.where(df_bank['job']=='services', 4,
                                                                     np.where(df_bank['job']=='management', 5,
                                                                              np.where(df_bank['job']=='retired', 6,
                                                                                       np.where(df_bank['job']=='entrepreneur', 7,
                                                                                                np.where(df_bank['job']=='self-employed', 8,
                                                                                                         np.where(df_bank['job']=='housemaid', 9,
                                                                                                                  np.where(df_bank['job']=='unemployed', 10,
                                                                                                                          np.where(df_bank['job']=='student', 11, 0))) ) ) ) ) ) ) ) ) )
df_bank = df_bank.assign(Contact_cat= np.where(df_bank['campaign'] == 1, 1,
                                               np.where(df_bank['campaign'] == 2, 2, 3)) )
df_bank = df_bank.assign(CPI_cat =np.where(df_bank['cons.price.idx'] < 93, 1,
                                           np.where(df_bank['cons.price.idx'] <= 93.5, 2,
                                                    np.where(df_bank['cons.price.idx'] <= 94, 3,
                                                             np.where(df_bank['cons.price.idx'] <= 94.5, 4, 5)))))
df_bank = df_bank.assign(CCI_cat = np.where(df_bank['cons.conf.idx'] <-45, 1,
                                            np.where(df_bank['cons.conf.idx']<= -40, 2,
                                                     np.where(df_bank['cons.conf.idx'] <= -35, 3, 4 ))) )
df_bank = df_bank.assign(marital_cat = np.where(df_bank['marital'] == 'married',1,
                                                np.where(df_bank['marital']=='single',2,
                                                         np.where(df_bank['marital'] == 'divorced', 3,0))))
df_bank = df_bank.assign(Edu_cat =np.where(df_bank['education'] == 'basic.4y', 1,
                                           np.where(df_bank['education'] == 'basic.6y', 1,
                                                    np.where(df_bank['education'] == 'basic.9y', 1,
                                                             np.where(df_bank['education'] == 'high.school', 2,
                                                                      np.where(df_bank['education'] == 'professional.course', 3,4))))) )
df_bank = df_bank.assign(Outcome_cat = np.where(df_bank['poutcome'] =='nonexistent',1,
                                                np.where(df_bank['poutcome'] == 'failure',2,
                                                       np.where(df_bank['poutcome'] == 'success', 3,0))) )
df_bank = df_bank.assign(rate_cat = np.where(df_bank['euribor3m'] <= 1, 1,
                                                   np.where(df_bank['euribor3m'] <= 2,2,
                                                            np.where(df_bank['euribor3m'] <=3,3,
                                                                     np.where(df_bank['euribor3m'] <=4,4,
                                                                              np.where(df_bank['euribor3m'] <=5,5,6))) )))
#encode to bool
df_bank = df_bank.assign(subscribed = np.where(df_bank['y'] == 'yes', 1, 0) )
df_bank = df_bank.assign(personal = np.where(df_bank['loan'] =='yes', 1, 0))
df_bank = df_bank.assign(mortgage = np.where(df_bank['housing'] == 'yes', 1, 0))

df_bank = df_bank.drop(columns=['job','marital', 'education','loan','housing','campaign','euribor3m','poutcome','cons.price.idx','cons.conf.idx','y'])

print(df_bank.sample(n=10))

#assign input variables
y = df_bank['subscribed']
X = df_bank.loc[:, df_bank.columns != 'subscribed']

#Split values to train and test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=1)

#build model
clf = DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth=3,max_features='sqrt',min_samples_leaf=10,min_samples_split=2)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Decision Tree with all features accuracy =",accuracy_score(y_test, y_pred)*100,"%")

#Find value of splitt importance
importances = clf.feature_importances_
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns)
plt.show()

#Print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(classification_report(y_test, y_pred))

#Print AUC value
DT_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree with all features (area = %0.2f)' % DT_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Export graphiz
dot_data = export_graphviz(clf,out_file='Tree1.dot',feature_names=X.columns
                           ,filled=True,rounded=True, class_names=['Not-buying','Buying'])

#Decision Tree with personal characteristics
X = df_bank[['age','Job_cat','marital_cat','Edu_cat']]

#Split values to train and test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=1)

#build model
clf = DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth=3,max_features='sqrt',min_samples_leaf=10,min_samples_split=2)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Decision Tree with personal characteristics accuracy =",accuracy_score(y_test, y_pred)*100,"%")

#Find value of splitt importance
importances = clf.feature_importances_
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns)
plt.show()

#Print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(classification_report(y_test, y_pred))

#Print AUC value
DT_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree with personal characteristics accuracy (area = %0.2f)' % DT_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Export graphiz
dot_data = export_graphviz(clf,out_file='Tree2.dot',feature_names=X.columns
                           ,filled=True,rounded=True, class_names=['Not-buying','Buying'])

#Best-important features
X = df_bank[['age','CCI_cat','Outcome_cat']]
print(X.head)

#Split values to train and test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=1)

#build model
clf = DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth=3,max_features='sqrt',min_samples_leaf=10,min_samples_split=2)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Decision Tree best features accuracy =",accuracy_score(y_test, y_pred)*100,"%")
importances = clf.feature_importances_

#Show importance
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns)
plt.show()

#Print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(classification_report(y_test, y_pred))

#Print AUC
DT_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree best features accuracy (area = %0.2f)' % DT_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Export graphiz
dot_data = export_graphviz(clf,out_file='Tree3.dot',feature_names=X.columns
                           ,filled=True,rounded=True, class_names=['Not-buying','Buying'])

#Make DT with features selected in Logistic Regression
X = df_bank[['CCI_cat','Outcome_cat','rate_cat','CPI_cat','personal','marital_cat','Job_cat','Edu_cat']]

#Split values to train and test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=1)

#build model
clf = DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth=3,max_features='sqrt',min_samples_leaf=10,min_samples_split=2)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Decision Tree Log reg features accuracy =",accuracy_score(y_test, y_pred)*100,"%")
importances = clf.feature_importances_

#Show which features are the most important
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns)
plt.show()

#Print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(classification_report(y_test, y_pred))

#Print AUC
DT_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree Log reg features (area = %0.2f)' % DT_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#Export graphiz
dot_data = export_graphviz(clf,out_file='Tree4.dot',feature_names=X.columns
                           ,filled=True,rounded=True, class_names=['Not-buying','Buying'])


#Make DT with client record data
X = df_bank[['Outcome_cat','personal','mortgage']]

#Split values to train and test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=1)

#build model
clf = DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth=3,max_features='sqrt',min_samples_leaf=10,min_samples_split=2)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Decision Tree client features accuracy =",accuracy_score(y_test, y_pred)*100,"%")
importances = clf.feature_importances_

#Show which features are the most important
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns)
plt.show()

#Print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(classification_report(y_test, y_pred))

#Print AUC
DT_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree client features (area = %0.2f)' % DT_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#Export graphiz
dot_data = export_graphviz(clf,out_file='Tree5.dot',feature_names=X.columns
                           ,filled=True,rounded=True, class_names=['Not-buying','Buying'])

#Make DT with client record data
X = df_bank[['CPI_cat','CCI_cat','rate_cat']]

#Split values to train and test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=1)

#build model
clf = DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth=3,max_features='sqrt',min_samples_leaf=10,min_samples_split=2)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Decision Tree macroeconomical features accuracy =",accuracy_score(y_test, y_pred)*100,"%")
importances = clf.feature_importances_

#Show which features are the most important
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances)
plt.xticks(range(X.shape[1]), X.columns)
plt.show()

#Print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print(classification_report(y_test, y_pred))

#Print AUC
DT_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree macroeconomical features (area = %0.2f)' % DT_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#Export graphiz
dot_data = export_graphviz(clf,out_file='Tree6.dot',feature_names=X.columns
                           ,filled=True,rounded=True, class_names=['Not-buying','Buying'])
