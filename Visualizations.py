import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

#Display settings for PyCharm
desired_width=420
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',21)

#import data
df_bank = pd.read_csv("bank-additional-full.csv")
df_bank = df_bank.drop(columns=['default','contact','day_of_week','duration','pdays','previous','nr.employed'])
#print(df_bank.head())

#Create df for contignency analysis
target_s = df_bank['y'] == "yes"
target_f = df_bank['y'] == "no"

#Categorical variable analysis
'''
print(df_bank['poutcome'].value_counts())
jobs = df_bank.groupby('poutcome').size()
jobs_f = df_bank[target_f].groupby('poutcome').size()
jobs_s = df_bank[target_s].groupby('poutcome').size()
headers = list(jobs_f.keys())

p1 = plt.bar(height=jobs_f, x=headers)
p2 = plt.bar(height=jobs_s, x=headers, bottom=jobs_f)

for xpos, ypos, yval in zip(headers, jobs_f/2, jobs_f / jobs * 100):
    plt.text(xpos, ypos, "%.1f"%yval+'%', ha="center", va="center")
for xpos, ypos, yval in zip(headers,jobs_f+jobs_s/2, jobs_s / jobs * 100):
    plt.text(xpos, ypos, "%.1f"%yval+'%', ha="center", va="center")

plt.title("Outcome of previous marketing campaign")
plt.ylabel("Frequency")
plt.xlabel("Category")
plt.legend((p2[0],p1[0]), ('Success', 'Failure'))

plt.show()
'''
x = df_bank['age']
#Numerical variable analysis
print(x.describe())
print(x.skew())
print(x.kurt())
print(x.mode())

x = df_bank['age']
plt.figure(figsize=(9, 8))
#sns.distplot(df_bank['age'], color='g', bins=100, hist_kws={'alpha': 0.4});
sns.distplot(x,fit=norm, hist_kws={'alpha': 0.4})
plt.show()
sns.boxplot(x=df_bank['y'], y=x)
plt.show()

corr = df_bank.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,cmap=cmap, vmax=1,annot=True, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
