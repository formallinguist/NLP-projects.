#1.8 Has vowel
import re

def has_vowel(word):
    pattern = '[aeiou]'
    re.search(pattern,word)
    for i in pattern:
        if i in word:
            return True
    else:
        return False       
word = input("Enter word:")
print(has_vowel(word))