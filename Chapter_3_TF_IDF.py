Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
>>> #Tokenizing the sentence.
>>> 
>>> import nltk
>>> from nltk.tokenize import TreebankWordTokenizer
>>> sentence = """The faster Harry got to the store, the faster Harry, the faster, would get home."""
>>> tokenizer = TreebankWordTokenizer()
>>> tokens = tokenizer.tokenize(sentence.lower())
>>> tokens
['the', 'faster', 'harry', 'got', 'to', 'the', 'store', ',', 'the', 'faster', 'harry', ',', 'the', 'faster', ',', 'would', 'get', 'home', '.']
>>> 
>>> 
>>> #using counter to count the number of occurances of words.
>>> from collections import Counter
>>> bag_of_words = Counter(tokens)
>>> bag_of_words
Counter({'the': 4, 'faster': 3, ',': 3, 'harry': 2, 'got': 1, 'to': 1, 'store': 1, 'would': 1, 'get': 1, 'home': 1, '.': 1})
>>> 
>>> 
>>> #Most common words.
>>> bag_of_words.most_common(4)
[('the', 4), ('faster', 3), (',', 3), ('harry', 2)]
>>> 
>>> 
>>> #Calculating the TermFrequency of the word Harry.
>>> times_harry_appears = bag_of_words['harry']
>>> num_unique_words = len(bag_of_words)
>>> tf = times_harry_appears / num_unique_words
>>> round(tf, 4)
0.1818
>>> 
>>> 
>>> #Using the "Kite" intro text from wikipedia.
>>> kite_text = """A kite is a tethered heavier-than-air or lighter-than-air craft with wing surfaces that react against the air to create lift and drag forces.[2] A kite consists of wings, tethers and anchors. Kites often have a bridle and tail to guide the face of the kite so the wind can lift it.[3] Some kite designs do not need a bridle; box kites can have a single attachment point. A kite may have fixed or moving anchors that can balance the kite. The name is derived from the kite, the hovering bird of prey.[4]
... 
... The lift that sustains the kite in flight is generated when air moves around the kite's surface, producing low pressure above and high pressure below the wings.[5] The interaction with the wind also generates horizontal drag along the direction of the wind. The resultant force vector from the lift and drag force components is opposed by the tension of one or more of the lines or tethers to which the kite is attached.[6] The anchor point of the kite line may be static or moving (e.g., the towing of a kite by a running person, boat, free-falling anchors as in paragliders and fugitive parakites[7][8] or vehicle).[9][10]

The same principles of fluid flow apply in liquids, so kites can be used in underwater currents.[11][12] Paravanes and otter boards operate underwater on an analogous principle.

Man-lifting kites were made for reconnaissance, entertainment and during development of the first practical aircraft, the biplane.

Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses. Sport kites can be flown in aerial ballet, sometimes as part of a competition. Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding, kite buggying and snow kiting."""


from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)
token_counts
Counter({'the': 27, 'kite': 14, ',': 13, 'and': 12, 'a': 11, '[': 11, ']': 11, 'of': 11, 'kites': 9, 'or': 7, 'can': 6, 'is': 5, 'to': 5, 'in': 5, 'be': 5, 'lift': 4, 'have': 4, 'that': 3, 'drag': 3, 'may': 3, 'as': 3, 'flown': 3, 'with': 2, 'air': 2, 'tethers': 2, 'bridle': 2, 'so': 2, 'wind': 2, 'moving': 2, 'anchors': 2, 'from': 2, 'pressure': 2, 'force': 2, 'by': 2, 'which': 2, '.': 2, 'used': 2, 'underwater': 2, 'for': 2, 'practical': 2, 'are': 2, 'power': 2, 'tethered': 1, 'heavier-than-air': 1, 'lighter-than-air': 1, 'craft': 1, 'wing': 1, 'surfaces': 1, 'react': 1, 'against': 1, 'create': 1, 'forces.': 1, '2': 1, 'consists': 1, 'wings': 1, 'anchors.': 1, 'often': 1, 'tail': 1, 'guide': 1, 'face': 1, 'it.': 1, '3': 1, 'some': 1, 'designs': 1, 'do': 1, 'not': 1, 'need': 1, ';': 1, 'box': 1, 'single': 1, 'attachment': 1, 'point.': 1, 'fixed': 1, 'balance': 1, 'kite.': 1, 'name': 1, 'derived': 1, 'hovering': 1, 'bird': 1, 'prey.': 1, '4': 1, 'sustains': 1, 'flight': 1, 'generated': 1, 'when': 1, 'moves': 1, 'around': 1, "'s": 1, 'surface': 1, 'producing': 1, 'low': 1, 'above': 1, 'high': 1, 'below': 1, 'wings.': 1, '5': 1, 'interaction': 1, 'also': 1, 'generates': 1, 'horizontal': 1, 'along': 1, 'direction': 1, 'wind.': 1, 'resultant': 1, 'vector': 1, 'components': 1, 'opposed': 1, 'tension': 1, 'one': 1, 'more': 1, 'lines': 1, 'attached.': 1, '6': 1, 'anchor': 1, 'point': 1, 'line': 1, 'static': 1, '(': 1, 'e.g.': 1, 'towing': 1, 'running': 1, 'person': 1, 'boat': 1, 'free-falling': 1, 'paragliders': 1, 'fugitive': 1, 'parakites': 1, '7': 1, '8': 1, 'vehicle': 1, ')': 1, '9': 1, '10': 1, 'same': 1, 'principles': 1, 'fluid': 1, 'flow': 1, 'apply': 1, 'liquids': 1, 'currents.': 1, '11': 1, '12': 1, 'paravanes': 1, 'otter': 1, 'boards': 1, 'operate': 1, 'on': 1, 'an': 1, 'analogous': 1, 'principle.': 1, 'man-lifting': 1, 'were': 1, 'made': 1, 'reconnaissance': 1, 'entertainment': 1, 'during': 1, 'development': 1, 'first': 1, 'aircraft': 1, 'biplane.': 1, 'long': 1, 'varied': 1, 'history': 1, 'many': 1, 'different': 1, 'types': 1, 'individually': 1, 'at': 1, 'festivals': 1, 'worldwide.': 1, 'recreation': 1, 'art': 1, 'other': 1, 'uses.': 1, 'sport': 1, 'aerial': 1, 'ballet': 1, 'sometimes': 1, 'part': 1, 'competition.': 1, 'multi-line': 1, 'steerable': 1, 'designed': 1, 'generate': 1, 'large': 1, 'forces': 1, 'activities': 1, 'such': 1, 'surfing': 1, 'landboarding': 1, 'buggying': 1, 'snow': 1, 'kiting': 1})


#Removing the stopwords from the kite intro text.

import nltk
nltk.download('stopwords', quiet=True)
True

stopwords = nltk.corpus.stopwords.words('english')
tokens = [ x for x in tokens if x not in stopwords]
kite_counts = Counter(tokens)
kite_counts
Counter({'kite': 14, ',': 13, '[': 11, ']': 11, 'kites': 9, 'lift': 4, 'drag': 3, 'may': 3, 'flown': 3, 'air': 2, 'tethers': 2, 'bridle': 2, 'wind': 2, 'moving': 2, 'anchors': 2, 'pressure': 2, 'force': 2, '.': 2, 'used': 2, 'underwater': 2, 'practical': 2, 'power': 2, 'tethered': 1, 'heavier-than-air': 1, 'lighter-than-air': 1, 'craft': 1, 'wing': 1, 'surfaces': 1, 'react': 1, 'create': 1, 'forces.': 1, '2': 1, 'consists': 1, 'wings': 1, 'anchors.': 1, 'often': 1, 'tail': 1, 'guide': 1, 'face': 1, 'it.': 1, '3': 1, 'designs': 1, 'need': 1, ';': 1, 'box': 1, 'single': 1, 'attachment': 1, 'point.': 1, 'fixed': 1, 'balance': 1, 'kite.': 1, 'name': 1, 'derived': 1, 'hovering': 1, 'bird': 1, 'prey.': 1, '4': 1, 'sustains': 1, 'flight': 1, 'generated': 1, 'moves': 1, 'around': 1, "'s": 1, 'surface': 1, 'producing': 1, 'low': 1, 'high': 1, 'wings.': 1, '5': 1, 'interaction': 1, 'also': 1, 'generates': 1, 'horizontal': 1, 'along': 1, 'direction': 1, 'wind.': 1, 'resultant': 1, 'vector': 1, 'components': 1, 'opposed': 1, 'tension': 1, 'one': 1, 'lines': 1, 'attached.': 1, '6': 1, 'anchor': 1, 'point': 1, 'line': 1, 'static': 1, '(': 1, 'e.g.': 1, 'towing': 1, 'running': 1, 'person': 1, 'boat': 1, 'free-falling': 1, 'paragliders': 1, 'fugitive': 1, 'parakites': 1, '7': 1, '8': 1, 'vehicle': 1, ')': 1, '9': 1, '10': 1, 'principles': 1, 'fluid': 1, 'flow': 1, 'apply': 1, 'liquids': 1, 'currents.': 1, '11': 1, '12': 1, 'paravanes': 1, 'otter': 1, 'boards': 1, 'operate': 1, 'analogous': 1, 'principle.': 1, 'man-lifting': 1, 'made': 1, 'reconnaissance': 1, 'entertainment': 1, 'development': 1, 'first': 1, 'aircraft': 1, 'biplane.': 1, 'long': 1, 'varied': 1, 'history': 1, 'many': 1, 'different': 1, 'types': 1, 'individually': 1, 'festivals': 1, 'worldwide.': 1, 'recreation': 1, 'art': 1, 'uses.': 1, 'sport': 1, 'aerial': 1, 'ballet': 1, 'sometimes': 1, 'part': 1, 'competition.': 1, 'multi-line': 1, 'steerable': 1, 'designed': 1, 'generate': 1, 'large': 1, 'forces': 1, 'activities': 1, 'surfing': 1, 'landboarding': 1, 'buggying': 1, 'snow': 1, 'kiting': 1})


#Vectorizing the text.
document_vector = []
doc_length = len(tokens)
for key, value in kite_counts.most_common():
    document_vector.append(value/ doc_length)

    
document_vector
[0.0603448275862069, 0.05603448275862069, 0.04741379310344827, 0.04741379310344827, 0.03879310344827586, 0.017241379310344827, 0.01293103448275862, 0.01293103448275862, 0.01293103448275862, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207, 0.004310344827586207]

docs = ["The faster Harry got to the store, the faster and faster Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry")

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]

    

len(doc_tokens[0])
17
print(docs)
['The faster Harry got to the store, the faster and faster Harry would get home.', 'Harry is hairy and faster than Jill.', 'Jill is not as hairy as Harry']

all_doc_tokens = sum(doc_tokens,[])
len(all_doc_tokens)
32

lexicon = sorted(set(all_doc_tokens))
len(lexicon)
18

print(lexicon)
[',', '.', 'and', 'as', 'faster', 'get', 'got', 'hairy', 'harry', 'home', 'is', 'jill', 'not', 'store', 'than', 'the', 'to', 'would']


from collections import OrderedDict
Zero_vector = OrderedDict((token, 0) for token in lexicon)
Zero_vector
OrderedDict([(',', 0), ('.', 0), ('and', 0), ('as', 0), ('faster', 0), ('get', 0), ('got', 0), ('hairy', 0), ('harry', 0), ('home', 0), ('is', 0), ('jill', 0), ('not', 0), ('store', 0), ('than', 0), ('the', 0), ('to', 0), ('would', 0)])


#Now we need to make copies of that base vector and update the values of the the vector for each document.

import copy
doc_vectors = []
for doc in docs:
    vec = copy.copy(Zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    doc_vectors.append(vec)

    

print(doc_vectors)
[OrderedDict([(',', 0.05555555555555555), ('.', 0.05555555555555555), ('and', 0.05555555555555555), ('as', 0), ('faster', 0.16666666666666666), ('get', 0.05555555555555555), ('got', 0.05555555555555555), ('hairy', 0), ('harry', 0.1111111111111111), ('home', 0.05555555555555555), ('is', 0), ('jill', 0), ('not', 0), ('store', 0.05555555555555555), ('than', 0), ('the', 0.16666666666666666), ('to', 0.05555555555555555), ('would', 0.05555555555555555)]), OrderedDict([(',', 0), ('.', 0.05555555555555555), ('and', 0.05555555555555555), ('as', 0), ('faster', 0.05555555555555555), ('get', 0), ('got', 0), ('hairy', 0.05555555555555555), ('harry', 0.05555555555555555), ('home', 0), ('is', 0.05555555555555555), ('jill', 0.05555555555555555), ('not', 0), ('store', 0), ('than', 0.05555555555555555), ('the', 0), ('to', 0), ('would', 0)]), OrderedDict([(',', 0), ('.', 0), ('and', 0), ('as', 0.1111111111111111), ('faster', 0), ('get', 0), ('got', 0), ('hairy', 0.05555555555555555), ('harry', 0.05555555555555555), ('home', 0), ('is', 0.05555555555555555), ('jill', 0.05555555555555555), ('not', 0.05555555555555555), ('store', 0), ('than', 0), ('the', 0), ('to', 0), ('would', 0)])]


#Zipf's law using Brown Corpus.

nltk.download('brown')
[nltk_data] Downloading package brown to C:\Users\RAVI
[nltk_data]     KIRAN\AppData\Roaming\nltk_data...
[nltk_data]   Package brown is already up-to-date!
True

from nltk.corpus import brown
brown.words()[:10]
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of']
brown.tagged_words()[:5]
[('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'), ('Jury', 'NN-TL')]

from collections import Counter
puncs = set((',','.','--','-','!','?',':',';',"''",'(',')','[',']'))
word_list = (x.lower() for x in brown.words() if x not in puncs)
token_counts = Counter(word_list)
token_counts.most_common(20)
[('the', 69971), ('of', 36412), ('and', 28853), ('to', 26158), ('a', 23195), ('in', 21337), ('that', 10594), ('is', 10109), ('was', 9815), ('he', 9548), ('for', 9489), ('``', 8837), ('it', 8760), ('with', 7289), ('as', 7253), ('his', 6996), ('on', 6741), ('be', 6377), ('at', 5372), ('by', 5306)]


###Topic Modelling.

#getting the total wordcount of each document.


kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)
print(intro_tokens)
['a', 'kite', 'is', 'a', 'tethered', 'heavier-than-air', 'or', 'lighter-than-air', 'craft', 'with', 'wing', 'surfaces', 'that', 'react', 'against', 'the', 'air', 'to', 'create', 'lift', 'and', 'drag', 'forces.', '[', '2', ']', 'a', 'kite', 'consists', 'of', 'wings', ',', 'tethers', 'and', 'anchors.', 'kites', 'often', 'have', 'a', 'bridle', 'and', 'tail', 'to', 'guide', 'the', 'face', 'of', 'the', 'kite', 'so', 'the', 'wind', 'can', 'lift', 'it.', '[', '3', ']', 'some', 'kite', 'designs', 'do', 'not', 'need', 'a', 'bridle', ';', 'box', 'kites', 'can', 'have', 'a', 'single', 'attachment', 'point.', 'a', 'kite', 'may', 'have', 'fixed', 'or', 'moving', 'anchors', 'that', 'can', 'balance', 'the', 'kite.', 'the', 'name', 'is', 'derived', 'from', 'the', 'kite', ',', 'the', 'hovering', 'bird', 'of', 'prey.', '[', '4', ']', 'the', 'lift', 'that', 'sustains', 'the', 'kite', 'in', 'flight', 'is', 'generated', 'when', 'air', 'moves', 'around', 'the', 'kite', "'s", 'surface', ',', 'producing', 'low', 'pressure', 'above', 'and', 'high', 'pressure', 'below', 'the', 'wings.', '[', '5', ']', 'the', 'interaction', 'with', 'the', 'wind', 'also', 'generates', 'horizontal', 'drag', 'along', 'the', 'direction', 'of', 'the', 'wind.', 'the', 'resultant', 'force', 'vector', 'from', 'the', 'lift', 'and', 'drag', 'force', 'components', 'is', 'opposed', 'by', 'the', 'tension', 'of', 'one', 'or', 'more', 'of', 'the', 'lines', 'or', 'tethers', 'to', 'which', 'the', 'kite', 'is', 'attached.', '[', '6', ']', 'the', 'anchor', 'point', 'of', 'the', 'kite', 'line', 'may', 'be', 'static', 'or', 'moving', '(', 'e.g.', ',', 'the', 'towing', 'of', 'a', 'kite', 'by', 'a', 'running', 'person', ',', 'boat', ',', 'free-falling', 'anchors', 'as', 'in', 'paragliders', 'and', 'fugitive', 'parakites', '[', '7', ']', '[', '8', ']', 'or', 'vehicle', ')', '.', '[', '9', ']', '[', '10', ']', 'the', 'same', 'principles', 'of', 'fluid', 'flow', 'apply', 'in', 'liquids', ',', 'so', 'kites', 'can', 'be', 'used', 'in', 'underwater', 'currents.', '[', '11', ']', '[', '12', ']', 'paravanes', 'and', 'otter', 'boards', 'operate', 'underwater', 'on', 'an', 'analogous', 'principle.', 'man-lifting', 'kites', 'were', 'made', 'for', 'reconnaissance', ',', 'entertainment', 'and', 'during', 'development', 'of', 'the', 'first', 'practical', 'aircraft', ',', 'the', 'biplane.', 'kites', 'have', 'a', 'long', 'and', 'varied', 'history', 'and', 'many', 'different', 'types', 'are', 'flown', 'individually', 'and', 'at', 'festivals', 'worldwide.', 'kites', 'may', 'be', 'flown', 'for', 'recreation', ',', 'art', 'or', 'other', 'practical', 'uses.', 'sport', 'kites', 'can', 'be', 'flown', 'in', 'aerial', 'ballet', ',', 'sometimes', 'as', 'part', 'of', 'a', 'competition.', 'power', 'kites', 'are', 'multi-line', 'steerable', 'kites', 'designed', 'to', 'generate', 'large', 'forces', 'which', 'can', 'be', 'used', 'to', 'power', 'activities', 'such', 'as', 'kite', 'surfing', ',', 'kite', 'landboarding', ',', 'kite', 'buggying', 'and', 'snow', 'kiting', '.']


kite_history = """Kites were known throughout Polynesia, as far as New Zealand, with the assumption being that the knowledge diffused from China along with the people. Anthropomorphic kites made from cloth and wood were used in religious ceremonies to send prayers to the gods.[17] Polynesian kite traditions are used by anthropologists to get an idea of early "primitive" Asian traditions that are believed to have at one time existed in Asia.[18]

Kites were late to arrive in Europe, although windsock-like banners were known and used by the Romans. Stories of kites were first brought to Europe by Marco Polo towards the end of the 13th century, and kites were brought back by sailors from Japan and Malaysia in the 16th and 17th centuries.[19][20] Konrad Kyeser described dragon kites in Bellifortis about 1400 AD.[21] Although kites were initially regarded as mere curiosities, by the 18th and 19th centuries they were being used as vehicles for scientific research.[19] In China, the kite has been claimed as the invention of the 5th-century BC Chinese philosophers Mozi (also Mo Di, or Mo Ti) and Lu Ban (also Gongshu Ban, or Kungshu Phan). Materials ideal for kite building were readily available including silk fabric for sail material; fine, high-tensile-strength silk for flying line; and resilient bamboo for a strong, lightweight framework. By 549 AD paper kites were certainly being flown, as it was recorded that in that year a paper kite was used as a message for a rescue mission. Ancient and medieval Chinese sources describe kites being used for measuring distances, testing the wind, lifting men, signaling, and communication for military operations. The earliest known Chinese kites were flat (not bowed) and often rectangular. Later, tailless kites"""

kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)
print(history_tokens)
['kites', 'were', 'known', 'throughout', 'polynesia', ',', 'as', 'far', 'as', 'new', 'zealand', ',', 'with', 'the', 'assumption', 'being', 'that', 'the', 'knowledge', 'diffused', 'from', 'china', 'along', 'with', 'the', 'people.', 'anthropomorphic', 'kites', 'made', 'from', 'cloth', 'and', 'wood', 'were', 'used', 'in', 'religious', 'ceremonies', 'to', 'send', 'prayers', 'to', 'the', 'gods.', '[', '17', ']', 'polynesian', 'kite', 'traditions', 'are', 'used', 'by', 'anthropologists', 'to', 'get', 'an', 'idea', 'of', 'early', '``', 'primitive', "''", 'asian', 'traditions', 'that', 'are', 'believed', 'to', 'have', 'at', 'one', 'time', 'existed', 'in', 'asia.', '[', '18', ']', 'kites', 'were', 'late', 'to', 'arrive', 'in', 'europe', ',', 'although', 'windsock-like', 'banners', 'were', 'known', 'and', 'used', 'by', 'the', 'romans.', 'stories', 'of', 'kites', 'were', 'first', 'brought', 'to', 'europe', 'by', 'marco', 'polo', 'towards', 'the', 'end', 'of', 'the', '13th', 'century', ',', 'and', 'kites', 'were', 'brought', 'back', 'by', 'sailors', 'from', 'japan', 'and', 'malaysia', 'in', 'the', '16th', 'and', '17th', 'centuries.', '[', '19', ']', '[', '20', ']', 'konrad', 'kyeser', 'described', 'dragon', 'kites', 'in', 'bellifortis', 'about', '1400', 'ad.', '[', '21', ']', 'although', 'kites', 'were', 'initially', 'regarded', 'as', 'mere', 'curiosities', ',', 'by', 'the', '18th', 'and', '19th', 'centuries', 'they', 'were', 'being', 'used', 'as', 'vehicles', 'for', 'scientific', 'research.', '[', '19', ']', 'in', 'china', ',', 'the', 'kite', 'has', 'been', 'claimed', 'as', 'the', 'invention', 'of', 'the', '5th-century', 'bc', 'chinese', 'philosophers', 'mozi', '(', 'also', 'mo', 'di', ',', 'or', 'mo', 'ti', ')', 'and', 'lu', 'ban', '(', 'also', 'gongshu', 'ban', ',', 'or', 'kungshu', 'phan', ')', '.', 'materials', 'ideal', 'for', 'kite', 'building', 'were', 'readily', 'available', 'including', 'silk', 'fabric', 'for', 'sail', 'material', ';', 'fine', ',', 'high-tensile-strength', 'silk', 'for', 'flying', 'line', ';', 'and', 'resilient', 'bamboo', 'for', 'a', 'strong', ',', 'lightweight', 'framework.', 'by', '549', 'ad', 'paper', 'kites', 'were', 'certainly', 'being', 'flown', ',', 'as', 'it', 'was', 'recorded', 'that', 'in', 'that', 'year', 'a', 'paper', 'kite', 'was', 'used', 'as', 'a', 'message', 'for', 'a', 'rescue', 'mission.', 'ancient', 'and', 'medieval', 'chinese', 'sources', 'describe', 'kites', 'being', 'used', 'for', 'measuring', 'distances', ',', 'testing', 'the', 'wind', ',', 'lifting', 'men', ',', 'signaling', ',', 'and', 'communication', 'for', 'military', 'operations.', 'the', 'earliest', 'known', 'chinese', 'kites', 'were', 'flat', '(', 'not', 'bowed', ')', 'and', 'often', 'rectangular.', 'later', ',', 'tailless', 'kites']


intro_total = len(intro_tokens)
intro_total
366

history_total = len(history_tokens)
history_total
326

#TF is number of times a word exists in the text divided by the total length of the text.

#IDF is the ratio of the total number of dcuments to the number of documents the term appears in that is "Total documents divided by documents containing a  word.

#Let's look the TF of 'Kite' in each document.

intro_tf = {}
history_tf = {}
intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total

history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite'] / history_total

'Term Frequency of "kite" in intro is: {:.4f}'.format(intro_tf['kite'])
'Term Frequency of "kite" in intro is: 0.0383'
'Term Frequency of "kite" in history is: {:.4f}'.format(history_tf['kite'])
'Term Frequency of "kite" in history is: 0.0123'


#Let's look at the TF of 'and' in each document.

intro_tf['and'] = intro_counts['and'] / intro_total
history_tf['and'] = history_counts['and'] / history_total

'Term Frequency of "and" in intro is: {:.4f}'.format(intro_tf['and'])
'Term Frequency of "and" in intro is: 0.0328'
'Term Frequency of "and" in intro is: {:.4f}'.format(history_tf['and'])
'Term Frequency of "and" in intro is: 0.0337'


#Let's look at the Tf of "China" in each document.

intro_tf['china'] = intro_counts['china'] / intro_total
history_tf['china'] = history_counts['china'] / history_total

'Term Frequency of "china" in intro is: {:.4f}'.format(intro_tf['china'])
'Term Frequency of "china" in intro is: 0.0000'
'Term Frequency of "china" in intro is: {:.4f}'.format(history_tf['china'])
'Term Frequency of "china" in intro is: 0.0061'


#Rarity measure to weight the term frequency.

num_docs_containing_and = 0
for doc in [intro_tokens, history_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1

        

num_docs_containing_kite = 0
for doc in [intro_tokens, history_tokens]:
    if 'kite' in doc:
        num_docs_containing_kite += 1

        

num_docs_containing_china = 0
for doc in [intro_tokens, history_tokens]:
    if 'china' in doc:
        num_docs_containing_china += 1

        
#IDF of the words "kite", "and" and "china"
        
num_docs = 2
intro_idf = {}
history_idf = {}

intro_idf['and'] = num_docs / num_docs_containing_and
history_idf['and'] = num_docs / num_docs_containing_and

intro_idf['kite'] = num_docs / num_docs_containing_kite
history_idf['kite'] = num_docs / num_docs_containing_kite

intro_idf['china'] = num_docs / num_docs_containing_china
history_idf['china'] = num_docs / num_docs_containing_china

intro_idf
{'and': 1.0, 'kite': 1.0, 'china': 2.0}

history_idf
{'and': 1.0, 'kite': 1.0, 'china': 2.0}


#TF-IDF of the intro document and history document for the three words "and", "kite" and "china".

intro_tfidf = {}
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
intro_tfidf['china'] = intro_tf['china'] * intro_idf['china']


history_tfidf = {}
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
history_tfidf['china'] = history_tf['and'] * history_idf['and']


intro_tfidf
{'and': 0.03278688524590164, 'kite': 0.03825136612021858, 'china': 0.0}

history_tfidf
{'and': 0.03374233128834356, 'kite': 0.012269938650306749, 'china': 0.03374233128834356}


#Relevance Ranking.

query = "How long does it take to get to the store?"
query_vec = copy.copy(Zero_vector)
query_vec = copy.copy(Zero_vector)

tokens = tokenizer.tokenize(query.lower())
token_counts = Counter(tokens)

documents = (kite_text, kite_history)


import math

def cosine_sim(vec1,vec2):
    "Convert the dictionaries to the lists for easier matching"
    vec1 = [val for val in vec1.values()]
    #print(vec1)
    vec2 = [val for val in vec2.values()]
    #print(vec2)
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    #print(mag_1)
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    #print(mag_2)
    return dot_prod / (mag_1 * mag_2)


document_tfidf_vectors = []

for doc in docs:
    vec = copy.copy(Zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)



for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in  documents:
        if key in _doc.lower():
            docs_containing_key += 1
    if docs_containing_key == 0:
        continue
    tf = value / len(tokens)
    idf = len(documents) / docs_containing_key
    query_vec[key] = tf * idf

    

for doc in docs:
    vec = copy.copy(Zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        docs_containing_key = 0
        for doc in docs:
            if key in _doc:
                docs_containing_key += 1
        tf = value /len(lexicon)
        if docs_containing_key:
            idf = len(docs) / docs_containing_key
        else:
            idf = 0
        vec[key] = tf * idf
    document_tfidf_vectors.append(vec)

    

cosine_sim(query_vec, document_tfidf_vectors[0])
0.9723055853282466
cosine_sim(query_vec, document_tfidf_vectors[1])
0.28005601680560194


#Using sk-learn to build TF-IDF vectorizer.

from sklearn.feature_extraction.text import TfidfVectorizer
print(docs)
['The faster Harry got to the store, the faster and faster Harry would get home.', 'Harry is hairy and faster than Jill.', 'Jill is not as hairy as Harry']

corpus = docs
vectorizer = TfidfVectorizer(min_df = 1)
model = vectorizer.fit_transform(corpus)
print(model.todense().round(2))
[[0.16 0.   0.48 0.21 0.21 0.   0.25 0.21 0.   0.   0.   0.21 0.   0.64
  0.21 0.21]
 [0.37 0.   0.37 0.   0.   0.37 0.29 0.   0.37 0.37 0.   0.   0.49 0.
  0.   0.  ]
 [0.   0.75 0.   0.   0.   0.29 0.22 0.   0.29 0.29 0.38 0.   0.   0.
  0.   0.  ]]
