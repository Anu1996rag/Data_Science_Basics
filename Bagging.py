#this code demonstrates the implementation of bagging concept

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

#importing dataset
data = load_iris()
X,y = data.data,data.target

#splitting out the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#creating object of Bagging Classifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_samples=50,verbose=2,n_jobs=-1)
bag_clf.fit(X_train,y_train)
pred = bag_clf.predict(X_test)

print(bag_clf.__class__.__name__ , accuracy_score(y_test,pred))
