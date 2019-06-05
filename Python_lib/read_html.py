import urllib.request as urllib2
from bs4 import BeautifulSoup
import re

def read_url_2_text(url):
     try:
          response = urllib2.urlopen(url)
     except:
          print(url, "aren't found - 404")
          return ""
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

def parse_html_to_text(html, html_part = ["p"], string=1):
     all_text = []
     for element in html_part:
          try:
               all_text += html.find_all(element)
          except:
               continue

     filter_text = []
     for line in all_text:
          if string:
               if not (line.string == None or len(line.string) < 10):
                    filter_text.append(line.string)
          else:
               if not (line.text == None or len(line.text) < 10):
                    filter_text.append(line.text)
     return (" ".join(filter_text)).replace("  ", " ")

def remove_nikud_from_dir(my_dir):
     import unicodedata
     from os import listdir,mkdir,rmdir
     from os.path import isfile, join,isdir

     # nikkud-test.txt is the file you save your text in.
     #my_dir = r"C:/Usersrzanzu/ri/Desktop/reference_project/Data/online_text"
     files = [f for f in listdir(my_dir) if isfile(join(my_dir, f))]
     for n_file in files:
          f= open(join(my_dir, n_file),'r', encoding='utf-8') 
          content = f.read()
          no_nikkud = remove_nikud(content)
          f.close()
          f = open(join(my_dir, n_file),'w',encoding='utf-8')
          f.write(no_nikkud)
          f.close()
def remove_nikud(text):
     import unicodedata
     normalized = unicodedata.normalize('NFKD', text)
     no_nikkud =''.join([c for c in normalized if not unicodedata.combining(c)])
     return no_nikkud

#html = read_url_2_text("file:///C:/Users/rzanzuri/Desktop/reference_project/Data/Online/f_00711.html")
#for i in range(2500000, 2600000):
#     html = read_url_2_text(f"https://news.walla.co.il/item/{i}")
#for i in range(500001, 600000):
     #html = read_url_2_text(f"https://www.maariv.co.il/news/israel/Article-{i}")
#     text = parse_html_to_text(html)
#     if len(text) > 10:
#          with open(f"./Data/download_pages/{i}.txt", "w", encoding="utf-8") as f:
#               f.write(text)