import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import random
from nltk.corpus import movie_reviews
from textblob.classifiers import NaiveBayesClassifier
random.seed(1)
import pandas as pd
data= pd.read_csv('boo.csv')
data=data[['Message','Label']]
print data

#print data
data=data.values
print data
ol=[]
for d in data:
    ol.append(d.tolist())
print ol
train=ol[:60]
test=ol[29900:]



cl = NaiveBayesClassifier(train)


accuracy = cl.accuracy(test)
print("Accuracy: {0}".format(accuracy))



res= pd.read_csv('foo.csv')


res=res.values
print res
pl=[]
for r in res:
    pl.append(r[1])
print pl
pred=cl.prob_classify(pl)
print pred.max()

# Show 5 most informative features
#cl.show_informative_features(5)
