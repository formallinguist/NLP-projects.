#1.13 Normalize Whitespace
import re
def function(string):
    pattern = '\s+' 
    re1 = re.sub(pattern,' ', string)
    return re1
string = input("string:")
print(function(string))
