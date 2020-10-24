import random

# Enum Class for Sentiments
class Sentiment:
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'
    POSITIVE = 'POSITIVE'


# defining a class for defining the sentiment

class Review:
    def __init__(self,text,score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment() # this calls the function while initializing the object

    def get_sentiment(self):
        '''this function derives the sentiment based on the overall ratings'''
        if self.score <=2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE


""" defining a class which separates out the two variables 
one is review text and other one is the sentiment score
and finally it evenly distributes the positive sentences and the
negative sentences  """
class ReviewContainer:
    def __init__(self,reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
    
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE,self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE,self.reviews))
        positive_reduced = positive[:len(negative)] # this is performed as we have more positive values than negative ones 
        #total dataset overwriting with the above calculated
        self.reviews = positive_reduced + negative
        #lastly to shuffle the dataset 
        random.shuffle(self.reviews)

# Splitting out and  Preparing the dataset
from sklearn.model_selection import train_test_split

'''This class splits the dataset into the training and
test dataset and returns with the dataset having features and labels split out'''
class SplitDataset :
    def __init__(self,reviews):
        self.reviews = reviews

    def split_it(self):

        training , test = train_test_split(self.reviews, 
                                            test_size = 0.33, random_state = 42)

        # passig the above split data to the ReviewContainer class to prepare the data
        train_container = ReviewContainer(training)
        test_container = ReviewContainer(test)

        # making the dataset evenly distributed by passing the values to the method
        train_container.evenly_distribute()
        train_X = train_container.get_text()
        train_y = train_container.get_sentiment()

        test_container.evenly_distribute()
        test_X = test_container.get_text()
        test_y = test_container.get_sentiment()

        return train_X,train_y,test_X,test_y



# loading the dataset
import json

fileName = './Books_small.json'
reviews =[]
with open(fileName) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'],review['overall'])) #creates object for the Review class and loads into the list



# calling out the Split dataset object 
dataSplit = SplitDataset(reviews)
train_X,train_y,test_X,test_y = dataSplit.split_it()

# BAG OF WORDS vectorization 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

train_X_vectorized = vectorizer.fit_transform(train_X)
test_X_vectorized = vectorizer.transform(test_X)

# using the Voting Classifier Approach to determine which algorithm is best suited for sentiment classification 
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# for model evaluation
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score, accuracy_score

def classificationMetrics(modelName,predictions):
    precision, recall, fbeta_score, support = precision_recall_fscore_support(test_y, predictions, average='weighted')
    f1_ = f1_score(test_y, predictions, average='weighted')
    acc_score = accuracy_score(test_y, predictions)
    print(f'{modelName}\n\t Precision :{round(precision,3)}  Recall : {round(recall,3)} f1_score : {round(f1_,3)}  Accuracy Score : {round(acc_score,3)}')


def votingClassifier():
    # creating all the objects
    log_clf = LogisticRegression()
    svc_clf = SVC()
    nb_clf = GaussianNB()
    dt_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier()

    voting_clf = VotingClassifier(estimators = [('lr',log_clf),('svc',svc_clf),('nb',nb_clf),
                                        ('dt',dt_clf),('rf',rf_clf)])
    
    print("******** Accuracies of various classifiers ********")
    
    for classifier in (log_clf,svc_clf,nb_clf,dt_clf,rf_clf,voting_clf):
        classifier.fit(train_X_vectorized.toarray(),train_y)
        pred = classifier.predict(test_X_vectorized.toarray())
        classificationMetrics(classifier.__class__.__name__,pred)

        # print(f"Classifier : {classifier.__class__.__name__}\nAccuracy Score : {accuracy_score(test_y,pred)}")
    
"""     #fitting the data
    voting_clf.fit(train_X_vectorized.toarray(),train_y)

    # predicting 
    pred = voting_clf.predict(test_X_vectorized.toarray())

    print("********** VOTING CLASSIFIER ACCURACY ***********")
    print(accuracy_score(test_y,pred)) """
    





# calling the function 
votingClassifier()

