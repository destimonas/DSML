from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

digits=load_digits()
x=digits.data
y=digits.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
k=KNeighborsClassifier(n_neighbors=10)
k.fit(x_train,y_train)
print(k.predict(x_test))
v=k.predict(x_test)
r=accuracy_score(y_test,v)
print("accuracy:",r)
