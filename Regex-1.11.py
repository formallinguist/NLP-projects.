#1.11 Numbers in a string

import re

def function(string):
    pattern = '[^0-9]' 
    r1 = re.split(pattern,string)
    return r1
string = input("Enterstring:")
print(function(string))