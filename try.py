import urllib.request as urllib2
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import re


def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

parser = HTMLParser()
#html_page = urllib2.urlopen("https://www.nytimes.com/")
new_page = []
with open("./Data/Online/f_00711.html", encoding = "ut") as page:
    lines = page.readlines()
    for line in lines:
        new_line = cleanhtml(line)
        if new_line != "" and new_line != "\n" and new_line != "\t\n":
            new_page.append(new_line)
            print("line:" , line)
            print("new line:", new_line)

#parser.feed(str(html_page.read()))
#print("Start tags", parser.lsStartTags)
#print("End tags", parser.lsEndTags)
#print("Start End tags", parser.lsStartEndTags)
#print("Comments", parser.lsComments)
#start = parser.get_starttag_text()
#print(start)
#a="אעהגכ"