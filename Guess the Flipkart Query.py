import numpy as np 
import re

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def pre_process_words(word):
    porter = PorterStemmer()
    word=porter.stem(word)
    return re.findall(r'(?:[a-zA-Z]+[a-zA-Z\'\-]?[a-zA-Z]|[a-zA-Z] +)',word)

def pre_process(txt):
    txt=txt.lower()
    tokens = txt.split()
    new_txt=""
    for i in tokens:
        temp=pre_process_words(i)
        if(temp):
            new_txt=new_txt+" "+temp[0]     
        
    return new_txt


def generate_dataset(path):
    x_train=[]
    y_train=[]
    c=0
    with open(path, 'r') as file:
        for line in file:
            c+=1
            if c==1:
                continue
            x=[a for a in line.rstrip().split("\t")]
            sen=pre_process(x[0])
            x_train.append(sen)
            y_train.append(x[1])
            
    return x_train,y_train


x_train,y_train=generate_dataset("training.txt")
x=np.array(x_train)
y=np.array(y_train)


text_clf=Pipeline([('vect',CountVectorizer()),   #convert text to vectors
                   ('tfidf',TfidfTransformer()), #normalizing Data
                   ('clf', LinearSVC())])        #using Support Vector Machine With Linear Kernel 
text_clf.fit(x,y)

test=[]
for i in range(int(input())): 
    x=input()
    sen=" ".join(word for word in pre_process_words(x))
    test.append(x)
predicted=text_clf.predict(np.array(test))
for i in predicted:
    print(i)


