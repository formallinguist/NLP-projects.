#1.3 An a followed by zero or one b
import re
pattern='ab{0,1}'
print(re.match(pattern,"abb"))