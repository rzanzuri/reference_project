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

def parse_html_to_text(html):
     all_text = html.find_all('p')
     all_text += html.find_all('div')

     filter_text = []
     for line in all_text:
          if(line.string == None or len(line.string) < 10):
               all_text.remove(line)
          else:
               filter_text.append(line.string)
     return "".join(filter_text)


#html = read_url_2_text("file:///C:/Users/rzanzuri/Desktop/reference_project/Data/Online/f_00711.html")
#for i in range(2500000, 2600000):
     #html = read_url_2_text(f"https://news.walla.co.il/item/{i}")
for i in range(500001, 600000):
     html = read_url_2_text(f"https://www.maariv.co.il/news/israel/Article-{i}")
     text = parse_html_to_text(html)
     if len(text) > 10:
          with open(f"./Data/walla_pages/{i}.txt", "w", encoding="utf-8") as f:
               f.write(text)