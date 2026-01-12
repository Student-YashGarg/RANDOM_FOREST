import pandas as pd

# Load Datadet
df=pd.read_csv(r"E:\Desktop\ML\RANDOM_FOREST\tourist_recommendation_rf.csv")
print(df)

# Features
x=df.drop(['Name','Recommended'],axis='columns') #Independent
y=df['Recommended'] #Dependent

# SPLIT DATA
from sklearn.model_selection import train_test_split
x_trian,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# load model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)

# train model
model.fit(x_trian,y_train)
print("Model Trained Successfully!")
print("Train Model Score:", model.score(x_trian,y_train)) # 1.0

# FEATURE IMPORTANCE
feature_importance=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
print(feature_importance)
# Nature            0.586317  highest
# Culture           0.187527
# Adventure         0.129173
# FamilyFriendly    0.044786
# Luxury            0.036561
# Budget            0.015636

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

# PREDICTION
y_test_pred=model.predict(x_test)
print("Test Model Score:", model.score(x_test,y_test)) # 1.0

# Predication
# User input as preference dictionary
na=int(input("Nature Friendly(Yes-1 , No-0): "))
cu=int(input("Culture Friendly(Yes-1 , No-0): "))
ad=int(input("Adventure Friendly(Yes-1 , No-0): "))
lx=int(input("Luxury Friendly(Yes-1 , No-0): "))
bd=int(input("Budget Friendly(Yes-1 , No-0): "))
fm=int(input("Family Friendly(Yes-1 , No-0): "))

user_preferences = {
    'Nature': na,
    'Culture': cu,
    'Adventure': ad,
    'Luxury': lx,
    'Budget': bd,
    'FamilyFriendly': fm
}
df_input=pd.DataFrame([user_preferences])
result=model.predict(df_input)
# Show result
if result == 1:
    print("\n Recommended Destination Based on Your Preferences")
else:
    print("\n No Suitable Recommendation Found for Your Preferences")
