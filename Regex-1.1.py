#1.1 An a followed by zero or more b

import re
pattern = r'ab*'
print(re.match(pattern,"abcd"))
print(re.findall(pattern,"abc acbbb bc abbc"))
