# CREDIT CARD FRAUD CLASSIFICATION

import pandas as pd

#-------LOAD DATASET--------------------------------
df=pd.read_csv("E:\Desktop\ML\RANDOM_FOREST\credit_card_fraud_large.csv")
print(df)
print(df.isnull().sum())

#------FRATURES------------------------------
x=df.drop(['IsFraud'],axis='columns')
y=df['IsFraud']

#-----SPLIT_DATA-------------------------
from sklearn.model_selection import train_test_split
x_trian,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#---------LOAD_MODEL-------------------------
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)

#-------TRAIN_MODEL-------------------------
model.fit(x_trian,y_train)
print("Model Trained Successfully!")
print("Train Model Score:", model.score(x_trian,y_train)) # 1.0

#-----------FEATURE IMPORTANCE---------------
feature_importance=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
print(feature_importance)

#---------PREDICTION-------------------------
# TEST PREDICTION
y_test_pred=model.predict(x_test)
print("Test Model Score:", model.score(x_test,y_test)) # 0.75

# via GUI
################################################################3
#--------GUI-------------------------------------
import tkinter as tk 
from tkinter import messagebox

app = tk.Tk()
app.title("CREDIT_CARD_FRAUD_DETECTION")
app.geometry("500x500")
app.configure(bg="#f0f0f0")

# Entry labels and fields
fields = {
    "Transaction Amount": None,
    "Transaction Time": None,
    "Location Risk (0=Safe, 1=Risky)": None,
    "Card Type (0=Debit, 1=Credit)": None,
}

tk.Label(app, text="CREDIT_CARD_FRAUD_DETECTION", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)
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
        sl = float(fields["Transaction Amount"].get())
        sw = float(fields["Transaction Time"].get())
        pl = float(fields["Location Risk (0=Safe, 1=Risky)"].get())
        pw = float(fields["Card Type (0=Debit, 1=Credit)"].get())
        features = [[sl,sw,pl,pw]]

        pred = model.predict(features)[0]

        msg = f"Prediction:{" Fraudulent Transaction!" if pred == 1 else " Transaction Safe."}\n" 
        # messagebox.showinfo("Result", msg)
        result.config(text=msg,fg='red' if pred==1 else 'green')

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



