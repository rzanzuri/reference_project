
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