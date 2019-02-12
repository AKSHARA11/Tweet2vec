import csv
import re
import fileinput
import random
from random import randint
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


# OPENING AND READING CSV FILE
fhand = open('dataset.csv','r',encoding="ISO-8859-1")
csv_f = csv.reader(fhand)



# CONVERTING ALL THE ROWS OF CSV FILE IN LOWERCASE PERMANENTLY
"""for line in fileinput.input("dataset.csv",inplace=1):
    print(line.lower(),end='')"""
#not req. everytime, ran once to convert to lower case




# SAMPLING 6 RANDOM TWEETS OR FIRST 10 TWEETS    
word = []
for i in range(random.randint(1,6)):
#for i in range(1,10):
    for row in csv_f:
        row = re.sub(r"[^a-zA-Z0-9-]+", ' ', str(row[5:]))	# REGEX TO REMOVE ALL SPECIAL CHARACTERS IN THE REQUIRED SENTENCES/TWEETS ([5:])
        row = str(row).split(" ")				# SPLITTING EACH TWEET INTO ITS WORDS
        word.append(row)					# APPENDING THE LIST OF WORDS OF EVERY TWEET IN AN EMPTY LIST
        break
    i+=1

print(word)
print('\n')




# COUNTING THE FREQUENCY OF EVERY WORD
counts={}
fst=[]

for tweet in word:
    for wrd in tweet:
        counts[wrd] = counts.get(wrd,0)+1

for (k,v) in counts.items():
    tup = (k,v)			
    fst.append(tup)						# TUPLE INSIDE THE LIST FST

print(fst)
print('\n')



# train model
model = gensim.models.Word2Vec(word, min_count = 1, size=100,window = 5)



# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
print(model)
print('\n')



# create a scatter plot of the projection
pyplot.scatter(result[:,0],result[:,1])
words = list(model.wv.vocab)
print(words)

for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i,0],result[i,1]))		# PLOTTING THE VECTORS
pyplot.show()

#print(model["@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"])
#print('\n')


model.save('model.bin')						# SAVING THE MODEL IN BIN FILE
new_model = Word2Vec.load('model.bin')
    
        
