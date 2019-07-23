from os import listdir,mkdir,rmdir
from os.path import isfile, join,isdir
from textHandler import get_text_file

def clean_wiki_file(wiki_file):

    new_lines = []
    with open(wiki_file, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("<doc id=") or line.startswith("</doc") or line == "\n" or line == "":
                continue
            new_lines.append(line)

    with open(wiki_file, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line)
            
    print(f"The file {wiki_file} was cleaned.")

def create_one_text_from_dir(my_dir, size):
    files = [f for f in listdir(my_dir) if isfile(join(my_dir,f))]
    all_text = []
    for f in files:
        print(f"Loading file: {f}.")
        lines = get_text_file(join(my_dir, f))
        for line in lines:
            if line.startswith("<doc id=") or line.startswith("</doc") or line == "\n" or line == "":
                continue
            all_text.append(line)
    print("Write the final file.")
    with open(join(my_dir, "final_eng_text.txt"), "w", encoding="utf-8") as final:
        for line in all_text:
            final.write(line)

create_one_text_from_dir(r"C:\Users\rzanzuri\Desktop\outputs\merged", 3*1024*1024*1024)