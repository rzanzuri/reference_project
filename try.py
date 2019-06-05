import unicodedata
from os import listdir,mkdir,rmdir
from os.path import isfile, join,isdir
# nikkud-test.txt is the file you save your text in.
my_dir = r"C:/Users/rzanzuri/Desktop/reference_project/Data/online_text"
files = [f for f in listdir(my_dir) if isfile(join(my_dir, f))]
for n_file in files:
  f= open(join(my_dir, n_file),'r', encoding='utf-8') 
  content = f.read()
  normalized=unicodedata.normalize('NFKD', content)
  no_nikkud=''.join([c for c in normalized if not unicodedata.combining(c)])
  no_nikkud
  f.close()
  f = open(join(my_dir, n_file),'w',encoding='utf-8')
  fw = f.write(no_nikkud)
  f.close()