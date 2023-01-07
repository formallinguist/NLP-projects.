#1.9 Is Integer
import re
def is_integer(word):
    pattern='[0-9]'
    r1=re.match(pattern,word)
    if r1:
        return "True"
    else:
        return "False"
word = input("Enter word:")
print(is_integer(word))
