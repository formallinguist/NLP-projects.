#1.4 An a followed by three b
import re
pattern ='ab{3}'
print(re.match(pattern,"abbbcabba"))