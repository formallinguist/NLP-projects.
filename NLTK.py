import nltk
nltk.download()
from nltk.book import *

#Question1
def lexical_diversity(text):
    diversity = len(text)/len(set(text))
    return diversity


print(lexical_diversity(text1))


#Question2
texts = (text1,text2,text3,text4,text5,text6,text7,text8,text9)

for word in texts:
    print(lexical_diversity(word))


#Question3
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
def frequent_30 (text):
    words = []
    for word in text:
        word = word.lower()
        if word.isalpha() and word not in stop_words:
            words.append(word)
    frequentdist = FreqDist(words)
    return (frequentdist).most_common(30)

print(frequent_30(text1))

#Question4
Data =(text1, text2, text3, text4,text5, text6,text7, text8, text9)
for word in Data:
    print (frequent_30(word))


#Question5
from nltk.corpus import brown
import nltk
nltk.download('brown')
brown.categories()
romance = brown.words(categories='romance')
fdist = nltk.FreqDist([w.lower() for w in romance])
Wh = ['what', 'when', 'where', 'who', 'why']
for i in Wh:
  print(i+ ':', fdist[i], end= ' ')


#Question6
import nltk
from nltk.corpus import gutenberg
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from urllib.request import urlopen

url= ("http://www.gutenberg.org/files/2554/2554-0.txt")
raw_data = urlopen(url).read().decode('utf-8')
data = word_tokenize(raw_data)
lemmatizer = WordNetLemmatizer()
for word in data:
    print(word + " ---> " + lemmatizer.lemmatize(word))

#Question 7
def percentage_of_word(text, word):
    word_fdist = nltk.FreqDist(w.lower() for w in text)
    freq_word = word_fdist [word]
    totalword = len(text)
    percentage = str (freq_word/totalword * 100)
    answer = (f"The percentage of the word:{word} is {percentage}")
    return answer
 
print (percentage_of_word(text4, "a"))


#Questin8
import requests
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
from urllib import request
from bs4 import BeautifulSoup

url1= "https://www.theguardian.com/environment/series/the-nature-of"
url2= "https://www.theguardian.com/uk/culture"

read_url1 = request.urlopen(url1)
read_url2 = request.urlopen (url2)

website_1= read_url1.read().decode('utf-8')
website_2= read_url2.read().decode('utf-8')

data1=BeautifulSoup(website_1, 'html.parser').get_text()
data2=BeautifulSoup(website_2, 'html.parser').get_text()

tokens1= word_tokenize(data1)
tokens2= word_tokenize(data2)

words1=[]
for word in tokens1:
    word = word.lower()
    if word.isalpha() and word not in stop_words:
        words1.append(word)
        
words2=[]
for word in tokens2:
    word=word.lower()
    if word.isalpha() and word not in stop_words:
        words2.append(word)
        
print ( percentage_of_word(words1, "environment"))
print ( percentage_of_word(words2, "movies"))