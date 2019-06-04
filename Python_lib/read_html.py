import urllib.request as urllib2
from bs4 import BeautifulSoup
from threading import Lock, RLock
import re

def read_url_2_text(url):
     try:
          response = urllib2.urlopen(url)
     except:
          print(url, "aren't found - 404")
          return ""
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

def parse_html_to_text(html, html_part = ["p"]):
     all_text = []
     for element in html_part:
          try:
               all_text += html.find_all(element)
          except:
               continue

     filter_text = []
     for line in all_text:
          if not (line.string == None or len(line.string) < 10):
               filter_text.append(line.string)
     return (" ".join(filter_text)).replace("  ", " ")


def parse_maariv_html_to_text(html):
     title = str(html.title.string)

     if title.find("ספורט 1") >= 0:
          all_text = html.find_all('div', class_="article-inner-content")
     else:
          return ""
          all_text = html.find_all('p')
          all_text += html.find_all('div')          

     all_text = html.find_all('p')
     all_text += html.find_all('div', class_="article-inner-content")

     filter_text = []
     for line in all_text:
          if line.string is not None and len(line.string) >= 10:
               filter_text.append(line.string)
          elif line.text is not None and len(line.text) >= 10:
               filter_text.append(line.text)
     return (" ".join(filter_text)).replace("  ", " ")     

html = read_url_2_text("https://www.maariv.co.il/news/israel/Article-500001")
text = parse_maariv_html_to_text(html)

with open(f"./Data/download_pages/Article-500003.txt", "w", encoding="utf-8") as f:
     f.write(text)