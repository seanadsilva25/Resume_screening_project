import pandas as pd

data = pd.read_csv("../data/UpdatedResumeDataSet.csv")

print(data.head())

data = data[['Resume_str','Category']]# Keep only required columns
data = data.dropna()#dropna() removes empty values

print(data.head())

#To clean the data
import re
def cleanResume(text):
    text = re.sub(r'http\S+',' ',text)
    text = re.sub(r'<.*?>',' ',text)
    text = re.sub(r'[^a-zA-z]',' ',text)
    text = text.lower()
    return text
data['Resume_str'] = data['Resume_str'].astype(str)
data['cleaned_resume'] = data['Resume_str'].apply(cleanResume)


#To convert text into numbers (TF-IDF feature)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english',max_features=10000,ngram_range=(1,2))#ngram_range allows model to learn more resume keywords

X = tfidf.fit_transform(data['cleaned_resume'])
y = data['Category']

#split training and testing the data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression #works well for text classification problems
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)#fit() trains the model

print("Model trained successfully")

y_pred = model.predict(X_test)#model predicts job category for unseen resumes

from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test,y_pred))

#resume prediction

resume_text = input("Enter Resume Text:")
resume_clean = cleanResume(resume_text)
resume_vector = tfidf.transform([resume_clean])
prediction = model.predict(resume_vector)
print("Predicted Job Category:",prediction[0])

#saving the trained model

import pickle
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(tfidf,open("tfidf.pkl","wb"))