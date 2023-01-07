#1.10 Is fraction
import re

def is_fraction(number):
    pattern= '[-{0,1}0-9+/1-9+]'
    r1= re.match(pattern,number)
    if r1:
        return "True"
    else:
        return "False"
number = input("Enternumber:")
print(is_fraction(number))