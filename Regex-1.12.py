#1.12 Valid Date
import re

regex = re.compile("[0-9]{4}\-[0-9]{2}\-[0-9]{2}")

def check_date_format(date):
    match = re.match(regex, date)

    if (match):
        return "True"

    else: 
       return check_date_format(input("Invalid date, try again: "))

date = input("date:")
print(check_date_format(date))