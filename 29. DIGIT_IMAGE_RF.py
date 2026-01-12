# DIGIT_IMAGE_CLASSIFIRE
# Using RANDOM FOREST
# use sklearn digits dataset

# DIGIT PREDICTION MODEL
# “A multiclass classification project using the Digits dataset to train a RANDOM FOREST model 
# that predicts handwritten digits from 8×8 pixel images.” 

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns

#---------LOAD_DATASET------------------------------
digits=load_digits() # Dictionary Format

print(digits.keys()) # dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
print(len(digits.images)) # 1797...total images
print(len(digits.data)) # 1797

print(digits.images[9])
# [[ 0.  0. 11. 12.  0.  0.  0.  0.]
#  [ 0.  2. 16. 16. 16. 13.  0.  0.]
#  [ 0.  3. 16. 12. 10. 14.  0.  0.]
#  [ 0.  1. 16.  1. 12. 15.  0.  0.]
#  [ 0.  0. 13. 16.  9. 15.  2.  0.]
#  [ 0.  0.  0.  3.  0.  9. 11.  0.]
#  [ 0.  0.  0.  0.  9. 15.  4.  0.]
#  [ 0.  0.  9. 12. 13.  3.  0.  0.]]

# plt.gray()
plt.matshow(digits.images[9],cmap='hot')
plt.show()

# for i in range(5):
#     plt.matshow(digits.images[i],cmap='hot')
#     plt.show()

#--------------FEATURES--------------
x=digits.data # independent...1797
y=digits.target # dependent...1797

#--------SPLIT DATASET--------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#---------------TRAIN MODEL----------------------------
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)
model.fit(x_train,y_train)
print("Model is Trained Successfully!")
print("TRAINED Model_Score:", model.score(x_train,y_train)) # 1.0

#-------------PREDICTION ON TEST--------------------
y_test_pred=model.predict(x_test)
print("Testing Data Prediction")
print(y_test_pred[:10])
print("Actual Prediction")
print(y_test[:10])

print("TEST Model_Score:",model.score(x_test,y_test)) # 0.96

#-----------CONFUSION MATRIX--------------
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_test_pred)
print("Confusion Matrix")
print(cm)


sns.heatmap(cm,annot=True)
plt.xlabel('Predication')
plt.ylabel('Truth')
plt.title("Confusion Matrix - Digits")
plt.show()







