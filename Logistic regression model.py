import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import seaborn as sns

#Display settings for PyCharm
desired_width=420
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',50)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

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
age_log = np.log(df_bank.age + 1)
print(age_log.skew())
print(df_bank.age.skew())
df_bank = df_bank.assign(age_log=pd.Series(np.log(df_bank.age + 1)))

#create categorical features
df_bank = pd.get_dummies(df_bank, columns=['poutcome'], drop_first=True)
df_bank = df_bank.assign(eurlibor3m_above_3 = np.where(df_bank['euribor3m'] > 3, 1,0))
df_bank = df_bank.assign(Contact_cat= np.where(df_bank['campaign'] == 1, 1,
                                               np.where(df_bank['campaign'] == 2, 2, 3)) )
df_bank = df_bank.assign(CPI_cat =np.where(df_bank['cons.price.idx'] < 93, 1,
                                           np.where(df_bank['cons.price.idx'] <= 93.5, 2,
                                                    np.where(df_bank['cons.price.idx'] <= 94, 3,
                                                             np.where(df_bank['cons.price.idx'] <= 94.5, 4, 5)))))
df_bank = df_bank.assign(CCI_cat = np.where(df_bank['cons.conf.idx'] <-45, 1,
                                            np.where(df_bank['cons.conf.idx']<= -40, 2,
                                                     np.where(df_bank['cons.conf.idx'] <= -35, 3, 4 ))))


#encode to bool
df_bank = df_bank.assign(subscribed = np.where(df_bank['y'] == 'yes', 1, 0) )
df_bank = df_bank.assign(personal_loan = np.where(df_bank['loan'] =='yes', 1, 0))
df_bank = df_bank.assign(mortgage = np.where(df_bank['housing'] == 'yes', 1, 0))

#Change categorical to bool / ordinal
df_bank= pd.get_dummies(df_bank, columns=['marital','job'], drop_first=True)
df_bank = df_bank.assign(Edu_cat =np.where(df_bank['education'] == 'basic.4y', 1,
                                           np.where(df_bank['education'] == 'basic.6y', 1,
                                                    np.where(df_bank['education'] == 'basic.9y', 1,
                                                             np.where(df_bank['education'] == 'high.school', 2,
                                                                      np.where(df_bank['education'] == 'professional.course', 3,4))))) )

df_bank = df_bank.drop(columns=['age','cons.price.idx','cons.conf.idx', 'y','loan','housing','campaign','euribor3m','education'])
df_bank2= df_bank.loc[:, df_bank.columns != 'subscribed']
#assign input variables
y = df_bank['subscribed']
X = df_bank.loc[:, df_bank.columns != 'subscribed']
print(X.head(n=10))
#Split values to train and test
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=1)

#Logistic regression
LR_Model = LogisticRegression(random_state=0 , class_weight='balanced')
LR_Model.fit(X_train,y_train)
LR_Pred = LR_Model.predict(X_test)
print("Logistic regression accuracy =",accuracy_score(y_test, LR_Pred)*100,"%")
print(X.columns)
print(LR_Model.coef_)
print(LR_Model.intercept_)

# ---------------------


def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se
    p = (1 - norm.cdf(abs(t))) * 2
    return p

# test p-values
model = LR_Model
np.set_printoptions(suppress=True)
df1 = pd.DataFrame(logit_pvalue(model, X_train),columns=['P_value']).astype(float)
df2 = pd.DataFrame(X.columns)
df3 = pd.DataFrame(np.array(["Const"]))
df_valid = df3.append(df2 ,ignore_index=True, sort=False)
df_valid = pd.concat([df_valid, df1],axis=1)
print(df_valid)
print(df_valid[df_valid.P_value < 0.05])


#Re-build model

df_bank = df_bank.drop(columns=['age_log', 'mortgage','marital_married','job_entrepreneur','job_management','job_self-employed','job_technician','job_unemployed'])

#verify model after rebuilding
y = df_bank['subscribed']
X = df_bank.loc[:, df_bank.columns != 'subscribed']
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.25, random_state=1)
LR_Model = LogisticRegression(random_state=0 , class_weight='balanced')
LR_Model.fit(X_train,y_train)
LR_Pred = LR_Model.predict(X_test)
print("Logistic regression accuracy =",accuracy_score(y_test, LR_Pred)*100,"%")
print(X.columns)
print(LR_Model.coef_)
print(LR_Model.intercept_)
model = LR_Model
np.set_printoptions(suppress=True)
df1 = pd.DataFrame(logit_pvalue(model, X_train),columns=['P_value']).astype(float)
df2 = pd.DataFrame(X.columns)
df3 = pd.DataFrame(np.array(["Const"]))
df_valid = df3.append(df2 ,ignore_index=True, sort=False)
df_valid = pd.concat([df_valid, df1],axis=1)
print(df_valid[df_valid.P_value < 0.05])

cnf_matrix = confusion_matrix(y_test, LR_Pred)
print(cnf_matrix)

print(classification_report(y_test, LR_Pred))

logit_roc_auc = roc_auc_score(y_test, LR_Model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, LR_Model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

