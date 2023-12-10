from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score


categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train=fetch_20newsgroups(subset="train",categories=categories,shuffle=True,random_state=42)

vectorizer=TfidfVectorizer()
x_train_tfidf=vectorizer.fit_transform(twenty_train.data)
y_train=twenty_train.target

x_train,x_test,y_train,y_test=train_test_split(x_train_tfidf,y_train,test_size=0.3,random_state=42)
svm_classifier=SVC(kernel='linear',random_state=42)

svm_classifier.fit(x_train,y_train)

pred=svm_classifier.predict(x_test)
accuracy_score=accuracy_score(y_test,pred)
class_report=classification_report(y_test,pred,target_names=twenty_train.target_names)

print("Accuracy Score",accuracy_score)
print("Classification Report\n",class_report)

new_data = [
    "Computer graphics"
]
x_new_tfidf=vectorizer.transform(new_data)

new_pred=svm_classifier.predict(x_new_tfidf)

for i,text in enumerate(new_data):
  predicted_category=twenty_train.target_names[new_pred[i]]
  print("Predicted Category:",predicted_category)
