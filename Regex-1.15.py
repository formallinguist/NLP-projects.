#1.15 Valid e-mail
import re
def function(valid_mail):
    pattern = '[a-zA-Z0-9]+@[a-zA-Z0-9\.]+\.{1}[edu|com|ord|net]'
    re1 = re.match(pattern,valid_mail)
    if re1:
        return "Correct"
    else: 
        return "Incorrect"

valid_mail = input("Entermail:")      
print(function(valid_mail))