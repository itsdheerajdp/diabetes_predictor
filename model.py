import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
dataset=pd.read_csv('diabetes.csv')
x=dataset[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]].values
y=dataset["Outcome"].values
classifier=RandomForestClassifier()
classifier.fit(x,y)
# making the pickle object of our machine learning model
pickle.dump(classifier, open("model.pkl","wb"))
