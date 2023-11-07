
import pandas as pd 
import nltk


df = pd.read_csv(r'/home/sandeep/disk_C/csv_file/sms_spam_detection/spam.csv')
df['sms'] = df.apply(lambda x : str(x[1])+str(x[2])+str(x[3])+str(x[4]), axis=1)
df = df.iloc[:,[0,5]]

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus =[]

for i in range(5572):
    sms = re.sub('[^A-Za-z]' ,' ', df['sms'][i])
    sms = sms.lower()
    sms = sms.split()
    ps = PorterStemmer()
    sms = [ps.stem(word) for word in sms if not word in set(stopwords.words('english'))]
    sms = ' '.join(sms)
    corpus.append(sms)

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features= 500)

X = tf.fit_transform(corpus).toarray()
y = df.iloc[:,[0]]
vocub = tf.vocabulary_

from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split(X,y, test_size = .25 , random_state = 42)

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

        

name = ['GaussianNB','LogisticRegression','SGDClassifier','RandomForest','DTC','KNN' ,'SVC']
classifiers =  [
        GaussianNB(),
        LogisticRegression(),
        SGDClassifier(max_iter = 100),
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        SVC(kernel= 'linear')        
        ]

metrics = ['Classifier','accuracy']


for model in classifiers:
    model.fit(X_train ,y_train)
    name = model.__class__.__name__
    y_pred = model.predict(X_test)    
    ac = accuracy_score(y_test , y_pred)

    print("="*30)
    print(name)
    print('******Result*******')
    print('accuracy : {}'.format(ac))
