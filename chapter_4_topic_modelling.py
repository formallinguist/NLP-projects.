Python 3.9.7 (default, Sep 16 2021, 13:09:58) 
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license()" for more information.
>>> topic = {}
>>> import numpy as np
>>> import pandas as pd
>>> 
>>> tfidf = dict(list(zip('cat dog apple lion NYC love'.split(), np.random.rand(6))))
>>> 
>>> topic['petness'] = (.3 * tfidf['cat'] +\
		        .3 * tfidf['dog'] +\
		         0 * tfidf['apple'] +\
		         0 * tfidf['lion'] -\
		        .2 * tfidf['NYC'] +\
		        .2 * tfidf['love'])
>>> 
>>> topic['animalness'] = (.1 * tfidf['cat'] +\
		           .1 * tfidf['dog'] -\
		           .1 * tfidf['apple'] +\
		           .5 * tfidf['lion'] +\
		           .1 * tfidf['NYC'] -\
		           .1 * tfidf['love'])
>>> 
>>> topic['cityness'] = ( 0 * tfidf['cat'] -\
		           .1 * tfidf['dog'] +\
		           .2 * tfidf['apple'] -\
		           .1 * tfidf['lion'] +\
		           .5 * tfidf['NYC'] +\
		           .1 * tfidf['love'])
>>> 
>>> 
>>> print(topic)
{'petness': 0.2699084219012048, 'animalness': 0.15796671998892792, 'cityness': 0.36850085303315927}
>>> 
>>> 
>>> #Vectors for the six words.
>>> 
>>> word_vector = {}
>>> 
>>> word_vector['cat'] = .3 * topic['petness'] +\
		         .1 * topic['animalness'] +\
		          0 * topic['cityness']
>>> 
>>> 
>>> word_vector['dog'] = .3 * topic['petness'] +\
		         .1 * topic['animalness'] -\
		         .1 * topic['cityness']
>>> 
>>> word_vector['apple'] = 0 * topic['petness'] -\
		           .1 * topic['animalness'] +\
		           .2 * topic['cityness']
>>> 
>>> word_vector['lion'] = 0 * topic['petness'] +\
		           .5 * topic['animalness'] -\
		           .1 * topic['cityness']
>>> 
>>> word_vector['NYC'] = .2 * topic['petness'] +\
		           .1 * topic['animalness'] +\
		           .5 * topic['cityness']
>>> 
>>> 
>>> word_vector['love'] = .2 * topic['petness'] -\
		           .1 * topic['animalness'] +\
		           .1 * topic['cityness']
>>> 
>>> 
>>> print(word_vector)
{'cat': 0.09676919856925423, 'dog': 0.0599191132659383, 'apple': 0.05790349860773906, 'lion': 0.04213327469114803, 'NYC': 0.2540287828957134, 'love': 0.0750350976846641}
>>> 
>>> 
>>> #Here, I trained an LDA model to classify the SMS messages as spam or non-spam.
>>> 
>>> from nlpia.data.loaders import get_data

Warning (from warnings module):
  File "/home/bluebird/anaconda3/lib/python3.9/site-packages/pugnlp/constants.py", line 136
    [datetime.datetime, pd.datetime, pd.Timestamp])
FutureWarning: The pandas.datetime class is deprecated and will be removed from pandas in a future version. Import from datetime module instead.

Warning (from warnings module):
  File "/home/bluebird/anaconda3/lib/python3.9/site-packages/pugnlp/constants.py", line 158
    MIN_TIMESTAMP = pd.Timestamp(pd.datetime(1677, 9, 22, 0, 12, 44), tz='utc')
FutureWarning: The pandas.datetime class is deprecated and will be removed from pandas in a future version. Import from datetime module instead.

Warning (from warnings module):
  File "/home/bluebird/anaconda3/lib/python3.9/site-packages/pugnlp/tutil.py", line 100
    np = pd.np
FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead

Warning (from warnings module):
  File "/home/bluebird/anaconda3/lib/python3.9/site-packages/pugnlp/util.py", line 80
    np = pd.np
FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead

Warning (from warnings module):
  File "/home/bluebird/nlpia/src/nlpia/futil.py", line 30
    np = pd.np
FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead

Warning (from warnings module):
  File "/home/bluebird/nlpia/src/nlpia/loaders.py", line 79
    np = pd.np
FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead
>>> pd.options.display.width = 120
>>> sms = get_data('sms-spam')
Loading file with name: sms-spam
>>> index = ['sms{} {}'.format(i, '!'*j) for (i,j) in\
	 zip(range(len(sms)), sms.spam)]
>>> sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
>>> sms['spam'] = sms.spam.astype(int)
>>> len(sms)
4837
>>> sms.spam.sum()
638
>>> sms.head(6)
        spam                                               text
sms0       0  Go until jurong point, crazy.. Available only ...
sms1       0                      Ok lar... Joking wif u oni...
sms2 !     1  Free entry in 2 a wkly comp to win FA Cup fina...
sms3       0  U dun say so early hor... U c already then say...
sms4       0  Nah I don't think he goes to usf, he lives aro...
sms5 !     1  FreeMsg Hey there darling it's been 3 week's n...
>>> 
>>> 
>>> #Performing Tokenization and TF-IDF vectorization.
>>> 
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
>>> tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
>>> tfidf_docs = tfidf_model.fit_transform(\
	raw_documents=sms.text).toarray()
>>> tfidf_docs.shape
(4837, 9232)
>>> sms.spam.sum()
638
>>> 
>>> 
>>>  #We can use "Mask" to select only spam rows.
>>> mask = sms.spam.astype(bool).values
>>> spam_centroid = tfidf_docs[mask].mean(axis=0)
>>> ham_centroid = tfidf_docs[~mask].mean(axis=0)
>>> 
>>> spam_centroid.round(2)
array([0.06, 0.  , 0.  , ..., 0.  , 0.  , 0.  ])
>>> ham_centroid.round(2)
array([0.02, 0.01, 0.  , ..., 0.  , 0.  , 0.  ])
>>> 
>>> spamminess_score = tfidf_docs.dot(spam_centroid - \
				  ham_centroid)
>>> spamminess_score.round(2)
array([-0.01, -0.02,  0.04, ..., -0.01, -0.  ,  0.  ])
>>> 
>>> 
>>> # Using "sklearnMinMaxscaler" to get a score to range between 0 and 1
>>> 
>>> from sklearn.preprocessing import MinMaxScaler
>>> sms['lda_score'] = MinMaxScaler().fit_transform(\
	spamminess_score.reshape(-1,1))
>>> sms['lda_predict'] = (sms.lda_score > .5).astype(int)
>>> sms['spam lda_predict lda_score'.split()].round(2).head(6)
        spam  lda_predict  lda_score
sms0       0            0       0.23
sms1       0            0       0.18
sms2 !     1            1       0.72
sms3       0            0       0.18
sms4       0            0       0.29
sms5 !     1            1       0.55
>>> 
>>> 
>>>  #Let's look at the rest of the training set.
>>> (1. - (sms.spam - sms.lda_predict).abs().sum() / len(sms)).round(3)
0.977
>>> 
>>>  #Confusion matrix of the training set.
>>> from pugnlp.stats import Confusion

Warning (from warnings module):
  File "/home/bluebird/anaconda3/lib/python3.9/site-packages/pugnlp/stats.py", line 25
    np = pd.np
FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead
>>> Confusion(sms['spam lda_predict'.split()])

Warning (from warnings module):
  File "/home/bluebird/anaconda3/lib/python3.9/site-packages/pugnlp/stats.py", line 504
    self.__setattr__('_hist_labels', self.sum().astype(int))
UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access

Warning (from warnings module):
  File "/home/bluebird/anaconda3/lib/python3.9/site-packages/pugnlp/stats.py", line 510
    setattr(self, '_hist_classes', self.T.sum())
UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
lda_predict     0    1
spam                  
0            4135   64
1              45  593
>>> 
>>> 
>>> #Topic word matrix for LSA on 16 short sentences about Cats, Dogs and NYC.
>>> 
>>> from nlpia.book.examples.ch04_catdog_lsa_3x6x16\
     import word_topic_vectors
Loading file with name: cats_and_dogs_sorted
  0%|          | 0/263 [00:00<?, ?it/s]100%|██████████| 263/263 [00:00<00:00, 52684.21it/s]
>>> word_topic_vectors.T.round(1)
      cat  dog  apple  lion  nyc  love
top0 -0.6 -0.4    0.5  -0.3  0.4  -0.1
top1 -0.1 -0.3   -0.4  -0.1  0.1   0.8
top2 -0.3  0.8   -0.1  -0.5  0.0   0.1
>>> 
>>> #Singular value decomposition.
>>> 
>>> from nlpia.book.examples.ch04_catdog_lsa_sorted\
     import lsa_models, prettify_tdm
Loading file with name: cats_and_dogs_sorted
  0%|          | 0/263 [00:00<?, ?it/s]100%|██████████| 263/263 [00:00<00:00, 104767.97it/s]
>>> bow_svd, tfidf_svd = lsa_models()
Loading file with name: cats_and_dogs_sorted
  0%|          | 0/263 [00:00<?, ?it/s]100%|██████████| 263/263 [00:00<00:00, 85631.26it/s]
>>> prettify_tdm(**bow_svd)
   cat dog apple lion nyc love                                             text
0              1        1                                 NYC is the Big Apple.
1              1        1                        NYC is known as the Big Apple.
2                       1    1                                      I love NYC!
3              1        1           I wore a hat to the Big Apple party in NYC.
4              1        1                       Come to NYC. See the Big Apple!
5              1                             Manhattan is called the Big Apple.
6    1                                  New York is a big city for a small cat.
7    1              1           The lion, a big cat, is the king of the jungle.
8    1                       1                               I love my pet cat.
9                       1    1                      I love New York City (NYC).
10   1   1                                              Your dog chased my cat.
>>> 
>>> tdm = bow_svd['tdm']
>>> tdm
       0   1   2   3   4   5   6   7   8   9   10
cat     0   0   0   0   0   0   1   1   1   0   1
dog     0   0   0   0   0   0   0   0   0   0   1
apple   1   1   0   1   1   1   0   0   0   0   0
lion    0   0   0   0   0   0   0   1   0   0   0
nyc     1   1   1   1   1   0   0   0   0   1   0
love    0   0   1   0   0   0   0   0   1   1   0
>>> 
>>> 
>>> #U left singular vectors.
>>> 
>>> import numpy as np
>>> U,s, Vt = np.linalg.svd(tdm)
>>> pd.DataFrame(U, index=tdm.index).round(2)
          0     1     2     3     4     5
cat   -0.04  0.83 -0.38 -0.00  0.11 -0.38
dog   -0.00  0.21 -0.18 -0.71 -0.39  0.52
apple -0.62 -0.21 -0.51  0.00  0.49  0.27
lion  -0.00  0.21 -0.18  0.71 -0.39  0.52
nyc   -0.75  0.00  0.24 -0.00 -0.52 -0.32
love  -0.22  0.42  0.69  0.00  0.41  0.37
>>> 
>>> #S singular values.
>>> s.round(1)
array([3.1, 2.2, 1.8, 1. , 0.8, 0.5])
>>> S = np.zeros((len(U), len(Vt)))
>>> np.fill_diagonal(S,s)
>>> pd.DataFrame(S).round(1)
    0    1    2    3    4    5    6    7    8    9    10
0  3.1  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
1  0.0  2.2  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
2  0.0  0.0  1.8  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
3  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
4  0.0  0.0  0.0  0.0  0.8  0.0  0.0  0.0  0.0  0.0  0.0
5  0.0  0.0  0.0  0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0
>>> 
>>> 
>>> #V right singular vectors.
>>> pd.DataFrame(Vt).round(2)
      0     1     2     3     4     5     6     7     8     9     10
0  -0.44 -0.44 -0.31 -0.44 -0.44 -0.20 -0.01 -0.01 -0.08 -0.31 -0.01
1  -0.09 -0.09  0.19 -0.09 -0.09 -0.09  0.37  0.47  0.56  0.19  0.47
2  -0.16 -0.16  0.52 -0.16 -0.16 -0.29 -0.22 -0.32  0.17  0.52 -0.32
3  -0.00 -0.00 -0.00 -0.00 -0.00  0.00 -0.00  0.71  0.00 -0.00 -0.71
4  -0.04 -0.04 -0.14 -0.04 -0.04  0.58  0.13 -0.33  0.62 -0.14 -0.33
5  -0.09 -0.09  0.10 -0.09 -0.09  0.51 -0.73  0.27 -0.01  0.10  0.27
6  -0.57  0.21  0.11  0.33 -0.31  0.34  0.34 -0.00 -0.34  0.23  0.00
7  -0.32  0.47  0.25 -0.63  0.41  0.07  0.07  0.00 -0.07 -0.18  0.00
8  -0.50  0.29 -0.20  0.41  0.16 -0.37 -0.37 -0.00  0.37 -0.17  0.00
9  -0.15 -0.15 -0.59 -0.15  0.42  0.04  0.04 -0.00 -0.04  0.63 -0.00
10 -0.26 -0.62  0.33  0.24  0.54  0.09  0.09  0.00 -0.09 -0.23 -0.00
>>> 
>>> 
>>> #Term-document matrix reconstruction error.
>>> err = []
>>> for numdim in range (len(s), 0, -1):
	S[numdim - 1, numdim -1] = 0
	reconstructed_tdm = U.dot(S).dot(Vt)
	err.append(np.sqrt(((\
		reconstructed_tdm - tdm).values.flatten() ** 2).sum()
		/ np.product(tdm.shape)))

	
>>> np.array(err).round(2)
array([0.06, 0.12, 0.17, 0.28, 0.39, 0.55])
>>> 
>>> 
>>> #Principal component analysis on 3d vectors.
>>> 
>>> import pandas as pd
>>> pd.set_option('display.max_columns',6)
>>> from sklearn.decomposition import PCA
>>> import seaborn
>>> from matplotlib import pyplot as plt
>>> from nlpia.data.loaders import get_data
>>> 
>>> df = get_data('pointcloud').sample(1000)
Loading file with name: pointcloud
>>> pca = PCA(n_components=2)
>>> df2d = pd.DataFrame(pca.fit_transform(df), columns=list('xy'))
>>> df2d.plot(kind='scatter', x = 'x', y ='y')
<AxesSubplot:xlabel='x', ylabel='y'>
>>> plt.show()
>>> 
>>> 
>>> #Finding the principal components using SVD on the 5000 SMS messages.
>>> from nlpia.data.loaders import get_data
>>> pd.options.display.width = 120
>>> sms = get_data('sms-spam')
Loading file with name: sms-spam
>>> 
>>> index = ['sms{} {}'.format(i, '!'*j)
	 for (i,j) in zip(range(len(sms)), sms.spam)]
>>> 
>>> sms.index = index
>>> sms.head(6)
        spam                                                    text
sms0       0  Go until jurong point, crazy.. Available only in bu...
sms1       0                           Ok lar... Joking wif u oni...
sms2 !     1  Free entry in 2 a wkly comp to win FA Cup final tkt...
sms3       0       U dun say so early hor... U c already then say...
sms4       0  Nah I don't think he goes to usf, he lives around h...
sms5 !     1  FreeMsg Hey there darling it's been 3 week's now an...
>>> 
>>> 
>>> #Now you can calculate the TF-IDF vectors for each of these messages.
>>> 
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
>>> 
>>> tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
>>> tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
>>> len(tfidf.vocabulary_)
9232
>>> 
>>> tfidf_docs = pd.DataFrame(tfidf_docs)
>>> tfidf_docs = tfidf_docs - tfidf_docs.mean()
>>> tfidf_docs.shape
(4837, 9232)
>>> sms.spam.sum()
638
>>> 
>>> 
>>> #Using PCA for SMS message semantic analysis.
>>> 
>>> from sklearn.decomposition import PCA
>>> pca = PCA(n_components=16)
>>> pca = pca.fit(tfidf_docs)
>>> pca_topic_vectors = pca.transform(tfidf_docs)
>>> columns = ['topic{}'.format(i) for i in range (pca.n_components)]
>>> pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns,\
				 index=index)
>>> 
KeyboardInterrupt
>>> pca_topic_vectors.round(3).head(6)
        topic0  topic1  topic2  ...  topic13  topic14  topic15
sms0     0.201   0.003   0.037  ...   -0.032   -0.011   -0.036
sms1     0.404  -0.094  -0.078  ...   -0.022    0.052    0.040
sms2 !  -0.030  -0.048   0.090  ...   -0.016   -0.052   -0.044
sms3     0.329  -0.033  -0.035  ...   -0.037    0.023    0.050
sms4     0.002   0.031   0.038  ...    0.045   -0.073    0.033
sms5 !  -0.016   0.059   0.014  ...    0.076    0.023   -0.048

[6 rows x 16 columns]
>>> tfidf.vocabulary_

>>> 
>>> 
>>> column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(),\
				     tfidf.vocabulary_.keys())))
>>> 
>>> terms

>>> 
>>>  #Creating a DataFrame of Pandas conataining the weights.
>>> weights = pd.DataFrame(pca.components_, columns=terms,index=['topic{}'.format(i) for i in range(16)])
>>> pd.options.display.max_columns = 8
>>> weights.head(4).round(3)
            !      "      #   #150  ...      …      ┾    〨ud      鈥
topic0 -0.071  0.008 -0.001 -0.000  ... -0.002  0.001  0.001  0.001
topic1  0.064  0.008  0.000 -0.000  ...  0.003  0.001  0.001  0.001
topic2  0.071  0.027  0.000  0.001  ...  0.002 -0.001 -0.001 -0.001
topic3 -0.059 -0.032 -0.001 -0.000  ...  0.001  0.001  0.001  0.001

[4 rows x 9232 columns]
>>> 
>>> pd.options.display.max_columns = 12
>>> deals = weights['! ;) :) half off free crazy deal only $ 80 %'.split()].round(3) * 100
>>> deals
            !   ;)    :)  half  off  free  crazy  deal  only    $   80    %
topic0   -7.1  0.1  -0.5  -0.0 -0.4  -2.0   -0.0  -0.1  -2.2  0.3 -0.0 -0.0
topic1    6.4  0.0   7.4   0.1  0.4  -2.3   -0.2  -0.1  -3.8 -0.1 -0.0 -0.2
topic2    7.1  0.2  -0.1   0.0  0.3   4.4    0.1  -0.1   0.7  0.0  0.0  0.1
topic3   -5.9 -0.3  -7.1   0.2  0.3  -0.2    0.0   0.1  -2.3  0.1 -0.1 -0.3
topic4   38.1 -0.1 -12.5  -0.1 -0.2   9.9    0.1  -0.2   3.0  0.3  0.1 -0.1
topic5  -26.5  0.1  -1.6  -0.3 -0.7  -1.4   -0.6  -0.2  -1.8 -0.9  0.0  0.0
topic6  -10.9 -0.5  19.9  -0.4 -0.9  -0.6   -0.2  -0.1  -1.4 -0.0 -0.0 -0.1
topic7   15.7  0.1 -17.9   0.8  0.8  -3.0    0.0   0.1  -1.8 -0.3  0.0 -0.1
topic8   34.5  0.1   5.0  -0.5 -0.5  -0.1   -0.4  -0.4   3.3 -0.6 -0.0 -0.2
topic9    7.4 -0.3  16.8   1.4 -0.9   6.0   -0.5  -0.4   3.4 -0.5 -0.0 -0.0
topic10 -32.2 -0.2  -9.8   0.2  0.1  12.3    0.1   0.0   0.5 -0.0 -0.1 -0.2
topic11 -22.8 -0.4 -29.9  -0.4 -1.5   4.0   -0.1  -0.1  -0.9  0.4  0.0  0.3
topic12 -24.5 -0.2  32.5  -0.2  0.2  -4.3   -0.5   0.1   2.9  0.4 -0.0  0.3
topic13  13.4 -0.2  32.3  -0.3  0.9   4.8    0.4   0.1  -2.4 -0.4  0.0 -0.1
topic14   1.0 -0.1  23.2  -0.2 -0.7   5.6    0.3  -0.1   3.3 -0.1  0.1 -0.3
topic15  -9.7  0.6  -5.3   0.7  1.5   3.4    0.6  -0.4  -1.3  0.7 -0.1  0.2
>>> 
>>> 
>>> deals.T.sum()
topic0    -11.9
topic1      7.6
topic2     12.7
topic3    -15.5
topic4     38.3
topic5    -33.9
topic6      4.8
topic7     -5.6
topic8     40.2
topic9     32.4
topic10   -29.3
topic11   -51.4
topic12     6.7
topic13    48.5
topic14    32.0
topic15    -9.1
dtype: float64
>>> 
>>> 
>>> #using the truncated SVD  to retain only the 16 most interesting topics.
>>>  from sklearn.decomposition import TruncatedSVD
 
SyntaxError: unexpected indent
>>> from sklearn.decomposition import TruncatedSVD
>>> svd = TruncatedSVD(n_components=16, n_iter=100)
>>> svd_topic_vectors = svd.fit_transform(tfidf_docs.values)
>>> svd_topic_vectors = pd.DataFrame(svd_topic_vectors, columns=columns,\
				 index=index)
>>> svd_topic_vectors.round(3).head(6)
        topic0  topic1  topic2  topic3  topic4  topic5  ...  topic10  topic11  topic12  topic13  topic14  topic15
sms0     0.201   0.003   0.037   0.011  -0.019  -0.053  ...    0.007   -0.007    0.002   -0.036   -0.014    0.037
sms1     0.404  -0.094  -0.078   0.051   0.100   0.047  ...   -0.004    0.036    0.043   -0.021    0.051   -0.042
sms2 !  -0.030  -0.048   0.090  -0.067   0.091  -0.043  ...    0.125    0.023    0.026   -0.020   -0.042    0.052
sms3     0.329  -0.033  -0.035  -0.016   0.052   0.056  ...    0.022    0.023    0.073   -0.046    0.022   -0.070
sms4     0.002   0.031   0.038   0.034  -0.075  -0.093  ...    0.028   -0.009    0.027    0.034   -0.083   -0.021
sms5 !  -0.016   0.059   0.014  -0.006   0.122  -0.040  ...    0.041    0.055   -0.037    0.075   -0.001    0.020

[6 rows x 16 columns]
>>> 
>>> 
>>> #First six topics for the first six sms.
>>> 
>>> svd_topic_vectors = (svd_topic_vectors.T / np.linalg.norm(\
	svd_topic_vectors, axis=1)).T
>>> svd_topic_vectors.iloc[:10].dot(svd_topic_vectors.iloc[:10].T).round(1)
        sms0   sms1   sms2 !  sms3   sms4   sms5 !  sms6   sms7   sms8 !  sms9 !
sms0      1.0    0.6    -0.1    0.6   -0.0    -0.3   -0.3   -0.1    -0.3    -0.3
sms1      0.6    1.0    -0.2    0.8   -0.2     0.0   -0.2   -0.2    -0.1    -0.1
sms2 !   -0.1   -0.2     1.0   -0.2    0.1     0.4    0.0    0.3     0.5     0.4
sms3      0.6    0.8    -0.2    1.0   -0.2    -0.3   -0.1   -0.3    -0.2    -0.1
sms4     -0.0   -0.2     0.1   -0.2    1.0     0.2    0.0    0.1    -0.4    -0.2
sms5 !   -0.3    0.0     0.4   -0.3    0.2     1.0   -0.1    0.1     0.3     0.4
sms6     -0.3   -0.2     0.0   -0.1    0.0    -0.1    1.0    0.1    -0.2    -0.2
sms7     -0.1   -0.2     0.3   -0.3    0.1     0.1    0.1    1.0     0.1     0.4
sms8 !   -0.3   -0.1     0.5   -0.2   -0.4     0.3   -0.2    0.1     1.0     0.3
sms9 !   -0.3   -0.1     0.4   -0.1   -0.2     0.4   -0.2    0.4     0.3     1.0
>>> 
>>> 
>>> #LDA works with raw BOW count vectors rather than normalized TF-IDF vectors.
>>> 
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from nltk.tokenize import casual_tokenize
>>> np.random.seed(42)
>>> 
>>> counter = CountVectorizer(tokenizer=casual_tokenize)
>>> bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text)\
			.toarray(), index=index)
>>> 
>>> column_nums, terms = zip(*sorted(zip(counter.vocabulary_.values(),\
				     counter.vocabulary_.keys())))
>>> 
>>> bow_docs.columns = terms
>>> 
>>> 
>>> #let's check the first sms message labelled as "sms0"
>>> 
>>> sms.iloc[0]
spam                                                         0
text    Go until jurong point, crazy.. Available only in bu...
Name: sms0 , dtype: object
>>> 
>>> bow_docs.loc['sms0 '][bow_docs.loc['sms0 '] > 0].head()
,            1
..           1
...          2
amore        1
available    1
Name: sms0 , dtype: int64
>>> 
>>> 
>>> 
>>> # Using LDiA to create topic vectors for sms corpus.
>>> from sklearn.decomposition import LatentDirichletAllocation as LDiA
>>> ldia = LDiA(n_components=16, learning_method='batch')
>>> ldia = ldia.fit(bow_docs)
>>> 
KeyboardInterrupt
>>> ldia.components_.shape
(16, 9232)
>>> 
>>> pd.set_option('display.width', 75)
>>> omponents = pd.DataFrame(ldia.components_.T, index=terms,\
			  columns=columns)
>>> components.round(2).head(3)
Traceback (most recent call last):
  File "<pyshell#249>", line 1, in <module>
    components.round(2).head(3)
NameError: name 'components' is not defined
>>> components = pd.DataFrame(ldia.components_.T, index=terms,\
			  columns=columns)
>>> components.round(2).head(3)
   topic0  topic1  topic2  topic3  topic4  topic5  ...  topic10  topic11  \
!  184.03   15.00   72.22  394.95   45.48   36.14  ...    37.42    44.18   
"    0.68    4.22    2.41    0.06  152.35    0.06  ...     8.42    11.42   
#    0.06    0.06    0.06    0.06    0.06    2.07  ...     0.06     0.06   

   topic12  topic13  topic14  topic15  
!    64.40   297.29    41.16    11.70  
"     0.07    62.72    12.27     0.06  
#     1.07     4.05     0.06     0.06  

[3 rows x 16 columns]
>>> 
>>> components.topic3.sort_values(ascending=False)[:10]
!       394.952246
.       218.049724
to      119.533134
u       118.857546
call    111.948541
£       107.358914
,        96.954384
*        90.314783
your     90.215961
is       75.750037
Name: topic3, dtype: float64
>>> 
>>> 
>>> #Lets see how topic vectors are different that are produced by SVD and PCA
>>> 
>>> ldia16_topic_vectors = ldia.transform(bow_docs)
>>> ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, \
				    index=index, columns=columns)
>>> ldia16_topic_vectors.round(2).head()
        topic0  topic1  topic2  topic3  topic4  topic5  ...  topic10  \
sms0      0.00    0.62    0.00    0.00    0.00    0.00  ...     0.00   
sms1      0.01    0.01    0.01    0.01    0.01    0.01  ...     0.01   
sms2 !    0.00    0.00    0.00    0.00    0.00    0.00  ...     0.00   
sms3      0.00    0.00    0.00    0.00    0.09    0.00  ...     0.00   
sms4      0.39    0.00    0.33    0.00    0.00    0.00  ...     0.00   

        topic11  topic12  topic13  topic14  topic15  
sms0       0.00     0.00     0.00     0.00     0.00  
sms1       0.12     0.01     0.01     0.01     0.01  
sms2 !     0.00     0.00     0.00     0.00     0.00  
sms3       0.00     0.00     0.00     0.00     0.00  
sms4       0.00     0.09     0.00     0.00     0.00  

[5 rows x 16 columns]
>>> 
>>> 
>>> #Let's see how good are these LDIA topics are at predicting.
>>> 
>>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
>>> from sklearn.model_selection import train_test_split
>>> X_train, X_test, Y_train, Y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size=0.5, random_state=271828)
>>> lda = LDA(n_components=1)
>>> lda = lda.fit(X_train, Y_train)
>>> sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
>>> round(float(lda.score(X_test, Y_test)), 2)
0.94
>>> 
>>> 
>>>  #Let's see how LDiA model compares to a much higher-dimensional model based on the TF-IDF vectors.
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> from nltk.tokenize.casual import casual_tokenize
>>> tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
>>> tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
>>> tfidf_docs = tfidf_docs - tfidf_docs.mean(axis=0)
>>> 
>>> X_train,X_test,Y_train,Y_test = train_test_split(tfidf_docs,\
						 sms.spam.values, test_size=0.5, random_state=271828)
>>> 
>>> lda = LDA(n_components=1)
>>> lda = lda.fit(X_train, Y_train)
>>> round(float(lda.score(X_train, Y_train)), 3)
1.0
>>> 
>>> round(float(lda.score(X_test, Y_test)), 3)
0.748
>>> 
>>> 
>>> #A comparison of 32 topics.
>>> ldia32 = LDiA(n_components=32, learning_method='batch')
>>> ldia32 = ldia32.fit(bow_docs)
>>> ldia32.components_.shape
(32, 9232)
>>> 
>>> 
>>> #compute 32-d topic vectors for all the sms messages.
>>> ldia32_topic_vectors = ldia32.transform(bow_docs)
>>> columns32 = ['topic{}'.format(i) for i in range(ldia32.n_components)]
>>> ldia32_topic_vectors = pd.DataFrame(ldia32_topic_vectors, index=index,\
				    columns=columns32)
>>> ldia32_topic_vectors.round(2).head()
        topic0  topic1  topic2  topic3  topic4  topic5  ...  topic26  \
sms0       0.0    0.00     0.0    0.06    0.14    0.00  ...      0.0   
sms1       0.0    0.00     0.0    0.00    0.53    0.00  ...      0.0   
sms2 !     0.0    0.00     0.0    0.00    0.00    0.65  ...      0.0   
sms3       0.0    0.11     0.0    0.00    0.39    0.00  ...      0.0   
sms4       0.0    0.00     0.0    0.00    0.00    0.00  ...      0.0   

        topic27  topic28  topic29  topic30  topic31  
sms0       0.00      0.0     0.00      0.0      0.0  
sms1       0.00      0.0     0.14      0.0      0.0  
sms2 !     0.00      0.0     0.00      0.0      0.0  
sms3       0.00      0.0     0.00      0.0      0.0  
sms4       0.47      0.0     0.00      0.0      0.0  

[5 rows x 32 columns]
>>> 
>>> #Now training the LdiA model classifier using 32 topic vectors.
>>> 
>>> X_train, X_test, Y_train, Y_test = train_test_split(ldia32_topic_vectors, sms.spam, test_size=0.5,random_state=271828)
>>> lda = LDA(n_components=1)
>>> lda = lda.fit(X_train, Y_train)
>>> sms['ldia32_spam'] = lda.predict(ldia32_topic_vectors)
>>> X_train.shape
(2418, 32)
>>> 
>>> 
>>> round(float(lda.score(X_train, Y_train)), 3)
0.933
>>> round(float(lda.score(X_test, Y_test)), 3)
0.936
>>> 
>>> 
>>> #LDA, Let's see how accurate the model can be at classifying the spam sms messages.
>>> 
>>> lda = LDA(n_components = 1)
>>> lda = lda.fit(tfidf_docs,sms.spam)
>>> sms['lda_spaminess'] = lda.predict(tfidf_docs)
>>> ((sms.spam - sms.lda_spaminess)** 2.).sum() ** .5
0.0
>>> 
>>> (sms.spam == sms.lda_spaminess).sum()
4837
>>> 
>>> len(sms)
4837
>>> 
>>> from sklearn.model_selection import cross_val_score
>>> lda = LDA(n_components=1)
>>> scores = cross_val_score(lda, tfidf_docs, sms.spam, cv=5)
>>> "Accuracy: {:.2f} (+/-{:.2f})".format(scores.mean(),scores.std()*2)
'Accuracy: 0.76 (+/-0.02)'
>>> 