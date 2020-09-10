#this file gives a demonstration of voting classifier working
# importing relevant libraries


from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


#importing dataset
data = load_iris()
X,y = data.data,data.target

#splitting out the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#creating objects for every model
lr = LogisticRegression()
svc = SVC()
rnd = RandomForestClassifier()

#creating Voting Classifier Object
vtg_clf = VotingClassifier(estimators=[('lr',lr),('rf',rnd),('svc',svc)])
vtg_clf.fit(X_train,y_train)
print(vtg_clf.score(X_train,y_train))

#Printing out the accuracy score of each model
for clf in (lr,svc,rnd,vtg_clf):
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(y_test,pred))
