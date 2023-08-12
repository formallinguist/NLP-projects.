Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import re
r = "(hi|hello|hey)[ ]*([a-z]*)" # "|" means "or" and "\*" means that the preceding character can occur zero or more times.
re.match(r,'Hello Rosa', flags=re.IGNORECASE)
<re.Match object; span=(0, 10), match='Hello Rosa'>
re.match(r, "hi ho, hi ho, it's off to work ...",flags=re.IGNORECASE)
<re.Match object; span=(0, 5), match='hi ho'>
#Ignoring the case of the text is common to keep the regular expressions simpler.
re.match(r, "hey, what's up", flags=re.IGNORECASE)
<re.Match object; span=(0, 3), match='hey'>
# Improving the above regular expression to match more greetings.
r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|"
r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|"\
    r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
re_greeting = re.compile(r, flags=re.IGNORECASE)
re_greeting.match('Hello Rosa')
<re.Match object; span=(0, 10), match='Hello Rosa'>
re_greeting.match('Hello Ravi').groups()
('Hello', None, None, 'Ravi')
re_greeting.match("Good morning Rosa")
<re.Match object; span=(0, 17), match='Good morning Rosa'>
>>> re_greeting.match("Good Manning Rosa")
>>> re_greeting.match("Good evening Rosa Parks").groups()
('Good evening', 'Good ', 'evening', 'Rosa')
>>> re_greeting.match("Good Morn'n Rosa")
<re.Match object; span=(0, 16), match="Good Morn'n Rosa">
>>> re_greeting.match("yo Rosa")
<re.Match object; span=(0, 7), match='yo Rosa'>
>>> #Adding output generation.
>>> my_names = set(['rosa','rose','chatty','chatbot','bot','chatterbot'])
>>> curt_names = set(['hal','you','u'])
>>> greeter_name = ''
>>> match = re_greeting.match(input())

>>> if match:
...     at_name = match.groups()[-1]
...     if at_name in curt_names:
...         print("Good one.")
...     elif at_name.lower() in my_names:
...         print("Hi {}, How are you?".format(greeter_name))
... 
...         
>>> #Using "Counter" in python to count strings.
...         
>>> from collections import Counter
>>> Counter("Guten Morgen Rosa".split())
Counter({'Guten': 1, 'Morgen': 1, 'Rosa': 1})
>>> Counter("Good morning, Rosa!".split())
Counter({'Good': 1, 'morning,': 1, 'Rosa!': 1})
>>> #all possible word orderings.
>>> from itertools import permutations
>>> [" ".join(combo) for combo in\
...  permutations("Good morning Rosa!".split(), 3)]
['Good morning Rosa!', 'Good Rosa! morning', 'morning Good Rosa!', 'morning Rosa! Good', 'Rosa! Good morning', 'Rosa! morning Good']
>>> s = """Find textbooks with titles  containing 'NLP',
... or 'natural' and 'language' , or
... 'computational' and 'linguistics'."""
>>> len(set(s.split()))
13
>>> import numpy as np
>>> np.arange(1,12 + 1).prod()
479001600
