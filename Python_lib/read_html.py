import urllib.request as urllib2
from bs4 import BeautifulSoup
import re

def read_url_2_text(url):
     response = urllib2.urlopen(url)
     #file:///C:/Users/rzanzuri/Desktop/reference_project/Data/Online/f_00711.html
     html_doc = response.read()

     # Parse the html file
     soup = BeautifulSoup(html_doc, 'html.parser')
     return soup

def clean_html_page(html):
     text = html.text # find_all('div'):
     text = text.replace("\n","")
     text = text.replace("\t","")
     #split_text = text.split('ADDITIONAL_STYLE')
     split_text = text.split('  ')
     size = len(split_text)
     text = max(split_text, key=len)

     #if size > 1:
     #     text = "".join(split_text[size - 1:])
     #else:
     #     text = "".join(split_text)
     cleanr = re.compile('<.*?>')
     text = re.sub(cleanr, '', text)
     # Format the parsed html file
     #strhtm = soup.prettify()   
     return text


html = read_url_2_text("file:///C:/Users/rzanzuri/Desktop/reference_project/Data/Online/f_00711.html")
text = clean_html_page(html)
with open("./stam_html.txt", "w", encoding="utf-8") as f:
     f.write(text)