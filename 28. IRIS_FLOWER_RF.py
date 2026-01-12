# IRIS FLOWER CLASSIFICATION
# using RANDOM FOREST

import pandas as pd

#--------LOAD DATASET------------------------
df=pd.read_csv("E:\Desktop\ML\DECISION_TREE\Iris.csv")
print(df)
# # print(df.isnull().sum()) # 0
# print(df.shape) # (150, 6)

#--------REPLACE STRING VALUE on DEPENDENT (Y) COLUMN---------
# FOR DEPENDENT NEVER USE OHE...always use map/repale/labelEncoder
df.replace({'Species':{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}},inplace=True)

# DROP ID ...IT NOT REQIUES
df.drop(['Id'],axis='columns',inplace=True)

print(df)
#------------FEATURES-------------------------------------------
x=df.drop(['Species'],axis='columns') # Independent
y=df['Species'] # Dependent

#----SPLIT DATA------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#------IMPORT MODEL----------------------
# by Decision Tree
# from sklearn import tree
# model=tree.DecisionTreeClassifier()

# by RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)

#---------TRAINED MODEL----------------------------------------------
model.fit(x_train,y_train)
print("Model is Trained Successfully!")
print("TRAIN MODEL SCORE:", model.score(x_train,y_train)) # 1.0

#-----------FEATURE IMPORTANCE---------------
feature_importance=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
print(feature_importance)

#--------------PREDICTION----------------------------------
# TEST PREDICTION 
y_test_pred=model.predict(x_test)
print("TEST MODEL SCORE:", model.score(x_test,y_test)) # 0.97

# MANUAL PREDICTION
# results be like ...{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

print(model.predict([[6.3,2.3,4.4,1.3]])) # [1].....Iris-versicolor

# spl=float(input("Enter value of SepalLength in Cm: "))
# spw=float(input("Enter value of SepalWidth in Cm: "))  
# ppl=float(input("Enter value of PetalLength in Cm: "))
# ppw=float(input("Enter value of PetalWidth in Cm: "))
# result=model.predict([[spl,spw,ppl,ppw]])
# if(result==0):
#     print("Type of Flower is Iris-setosa")
# elif(result==1):
#     print("Type of Flower is Iris-versicolor")
# else:
#     print("Type of Flower is Iris-virginica")

################################################################3
#--------GUI-------------------------------------
import tkinter as tk 
from tkinter import messagebox

app = tk.Tk()
app.title("IRIS FLOWER Detection")
app.geometry("500x500")
app.configure(bg="#f0f0f0")

# Entry labels and fields
fields = {
    "SepalLength (cm)": None,
    "SepalWidth (cm)": None,
    "PetalLength (cm)": None,
    "PetalWidth (cm)": None,
}

tk.Label(app, text="IRIS FLOWER DETECTION", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)
frame = tk.Frame(app, bg="#f0f0f0")
frame.pack()

for i, label in enumerate(fields):
    tk.Label(frame, text=label, font=("Arial", 12), bg="#f0f0f0").grid(row=i, column=0, pady=8, padx=10, sticky="w")
    entry = tk.Entry(frame, font=("Arial", 12), width=20)
    entry.grid(row=i, column=1, pady=8, padx=10)
    fields[label] = entry

result=tk.Label(app, text="", font=("ariel", 22), bg="#f0f0f0")
result.pack(pady=10)
# Prediction function
def predict_loan():
    try:
        sl = float(fields["SepalLength (cm)"].get())
        sw = float(fields["SepalWidth (cm)"].get())
        pl = float(fields["PetalLength (cm)"].get())
        pw = float(fields["PetalWidth (cm)"].get())
        features = [[sl,sw,pl,pw]]

        prediction = model.predict(features)[0]
    
        label_map = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"}

        msg = f"Prediction:{label_map[prediction]}\n" \
              f"\nModel Accuracy: {model.score(x_test,y_test)*100:.2f}%"
        
        # messagebox.showinfo("Result", msg)
        result.config(text=msg,fg='blue')
    except ValueError:
        # messagebox.showerror("Invalid Input", "Please enter valid numerical values.")
        result.config(text="Please enter valid numerical values.",fg='red')

# Plot function
def plot_fi():
        
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

# Button
tk.Button(app, text="Predict", command=predict_loan,
          font=("Arial", 12), bg="#4caf50", fg="white", padx=10, pady=5).pack(pady=20)
tk.Button(app, text="Feature_Importance", command=plot_fi,
          font=("Arial", 12), bg="#4c98af", fg="white", padx=10, pady=5).pack(pady=20)

app.mainloop()



