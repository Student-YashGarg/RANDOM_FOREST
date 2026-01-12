# LOAN APPROVAL CLASSIFIER MODEL
# USING RANDOM FOREST

import pandas as pd

df=pd.read_csv("E:\Desktop\ML\RANDOM_FOREST\loan_data.csv")
print(df)

print(df.isnull().sum())

# #---------REPLACE STRING WITH VALUE---------------

# df.replace({'Gender':{'Male':1,'Female':0}},inplace=True)
# df.replace({'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
# df.replace({'Married':{'Yes':1,'No':0}},inplace=True)
# df.replace({'Loan_Status':{'Y':1,'N':0}},inplace=True)

# via Label Encoder...use SAME COLUMN
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

df['Gender']=lb.fit_transform(df['Gender'])
df['Education']=lb.fit_transform(df['Education'])
df['Married']=lb.fit_transform(df['Married'])
df['Loan_Status']=lb.fit_transform(df['Loan_Status'])

df.drop(['Unnamed: 6'],axis='columns',inplace=True)

print(df)

#-----------FRATURES-------------
x=df.drop(['Loan_Status'],axis='columns')
y=df['Loan_Status']

#----------SPLIT_DATA-----------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(len(x_train))
print(len(x_test))

#------LOAD_MODEL------------------
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)

#----------TRAN_MODEL------------------
model.fit(x_train,y_train)
print("Model is Trained Successfully!")

#---------------------------------------------------------------
# FEATURE IMPORTANCE
feature_importance=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
print(feature_importance)
# LoanAmount         0.476774  highest
# ApplicantIncome    0.281519
# Married            0.119348
# Gender             0.117735
# Education          0.004624

# plot feature importance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
ax=sns.barplot(y=feature_importance,x=feature_importance.index,palette='viridis')
for container in ax.containers:
    ax.bar_label(container)
plt.xlabel("Feature_Importance")
plt.ylabel("Features")
plt.title("FEATURE IMPORTANCE PLOT")
plt.grid(True)
plt.tight_layout()
plt.show()
#---------------------------------------------------------------------
#-----TEST_DATA_PREDICTION------------
y_test_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_test_pred)
print("TEST Model Score (Accuracy):", accuracy) # 1.0

#------MANUAL_PREDICTION-----------
print(model.predict([[1,1,0,5000,200]])) # 1 ..approved

gen=int(input("Enter Gender(Male-1 ,Female-0) :"))
mar=int(input("Enter Married Status (1-Yes 0-No) :"))
edu=int(input("Enter Your Education:(Graduate-0,Not Graduate 1) :"))
tc=int(input("Enter ApplicantIncome:"))
amt=float(input("Enter LoanAmount :"))
result=model.predict([[gen,mar,edu,tc,amt]])
if(result==1):
    print("********Loan Approved*******")
else:
    print(".......Loan is Cancel .....")