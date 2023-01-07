#1.14 Remove non-alphanumeric
import re
def function(string):
    pattern = '[^a-zA-Z0-9]'
    re1 = re.sub(pattern, ' ', string)
    return re1
string = input("Enter string:")
print(function(string))