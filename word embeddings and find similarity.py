#!/usr/bin/env python
# coding: utf-8

# In[29]:


pip install gensim


# In[30]:


import gensim


# In[31]:


import gensim.downloader as api


# In[ ]:


model = api.load('word2vec-google-news-300')


# In[ ]:


from gensim.models import word2vec


# In[6]:


model['Apple']


# In[7]:


wv = api.load('word2vec-google-news-300')


# In[8]:


print(wv.most_similar(positive=['woman','king'],negative=['man']))


# In[9]:


print(wv.similarity('boy','man'))


# In[10]:


print(wv.doesnt_match("breakfast cereal dinner lunch".split()))


# In[11]:


from gensim.models import Word2Vec
sentences = [["cat","say","meow"],["dog","say","woof"]]
model = Word2Vec(sentences,min_count=1)


# In[12]:


model.build_vocab(sentences)


# In[13]:


model.train(sentences,total_examples=model.corpus_count,epochs=model.epochs)


# In[14]:


import nltk


# In[15]:


text = 'Ravi is eating an apple'
sent = nltk.sent_tokenize(text)
print(sent[0])


# In[16]:


from nltk import word_tokenize
tokens = word_tokenize(sent[0])
print(tokens)


# In[17]:


nltk.download('averaged_perceptron_tagger')


# In[18]:


nltk.pos_tag(tokens)


# In[22]:


text =   '''Historians write in the context of their own time, and with due regard to the current dominant ideas of how to interpret the past, and sometimes write to provide lessons for their own society. In the words of Benedetto Croce, "All history is contemporary history". History is facilitated by the formation of a "true discourse of past" through the production of narrative and analysis of past events relating to the human race.[21] The modern discipline of history is dedicated to the institutional production of this discourse.

All events that are remembered and preserved in some authentic form constitute the historical record.[22] The task of historical discourse is to identify the sources which can most usefully contribute to the production of accurate accounts of past. Therefore, the constitution of the historian's archive is a result of circumscribing a more general archive by invalidating the usage of certain texts and documents (by falsifying their claims to represent the "true past"). Part of the historian's role is to skillfully and objectively utilize the vast amount of sources from the past, most often found in the archives. The process of creating a narrative inevitably generates a silence as historians remember or emphasize different events of the past.[23][clarification needed]

The study of history has sometimes been classified as part of the humanities and at other times as part of the social sciences.[24] It can also be seen as a bridge between those two broad areas, incorporating methodologies from both. Some individual historians strongly support one or the other classification.[25] In the 20th century, French historian Fernand Braudel revolutionized the study of history, by using such outside disciplines as economics, anthropology, and geography in the study of global history.

Traditionally, historians have recorded events of the past, either in writing or by passing on an oral tradition, and have attempted to answer historical questions through the study of written documents and oral accounts. From the beginning, historians have also used such sources as monuments, inscriptions, and pictures. In general, the sources of historical knowledge can be separated into three categories: what is written, what is said, and what is physically preserved, and historians often consult all three.[26] But writing is the marker that separates history from what comes before.

Archaeology is especially helpful in unearthing buried sites and objects, which contribute to the study of history. Archaeological finds rarely stand alone, with narrative sources complementing its discoveries. Archaeology's methodologies and approaches are independent from the field of history. "Historical archaeology" is a specific branch of archaeology which often contrasts its conclusions against those of contemporary textual sources. For example, Mark Leone, the excavator and interpreter of historical Annapolis, Maryland, USA, has sought to understand the contradiction between textual documents idealizing "liberty" and the material record, demonstrating the possession of slaves and the inequalities of wealth made apparent by the study of the total historical environment.

There are varieties of ways in which history can be organized, including chronologically, culturally, territorially, and thematically. These divisions are not mutually exclusive, and significant intersections are often present. It is possible for historians to concern themselves with both the very specific and the very general, although the modern trend has been toward specialization. The area called Big History resists this specialization, and searches for universal patterns or trends. History has often been studied with some practical or theoretical aim, but also may be studied out of simple intellectual curiosity.[2 '''


# In[23]:


text


# In[24]:


from wordcloud import WordCloud


# In[28]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
wordcloud = WordCloud().generate(text)
plt.figure(figsize=(10,8))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')


# # 
