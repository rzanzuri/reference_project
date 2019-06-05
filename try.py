import unicodedata
from os import listdir,mkdir,rmdir
from os.path import isfile, join,isdir
import re
import datetime


start = datetime.datetime.now()
print("start:", start)

# nikkud-test.txt is the file you save your text in.
my_dir = r"C:/Users/rzanzuri/Desktop/reference_project/Data/"
with open(join(my_dir,"all.txt"), encoding = "utf-8-sig") as f:
  content = f.read()
  content = re.sub(r"\n+"," ", content)
  lines = content.split(".")
  finish = datetime.datetime.now()
  print("middle:", finish)
  print("total:", finish - start)
  count = 0
  for line in lines:
    if count == 1000:
      exit()
    if line.find("כתב") != -1:
      print(line)
      count += 1
      with open("./with_reference.txt", "a",encoding="utf-8") as ff:
        ff.write(line + ".\n")

