import re
def read_file_to_tuple(file = "./Outputs/HebrewText_3000_Sen_NER.txt", key = "655360"):
    with open(file, encoding="utf-8") as f:
        all_sentences = []
        sentence = ""
        space1 = ""
        space2 = None
        isNer = 0
        left = True # to know if we are in left "/' or right
        for line in f:
            if line == "" or line == "\n" or line == "\t":
                continue
            if len(all_sentences) == 29:
                print("here")
            (word,key_word) = line.rstrip().split("\t")
            #if re.search("\W",word):
            if word == ")" or word == "]" or word == "}" or word == "," or word == ":":
                space1 = ""
                #space2 = " "
            if word == "(" or word == "[" or word == "{":
                space1 = " "
                space2 = ""
            if word == "-":
                space1 = ""
                space2 = ""
            if word == "'" or word == '"':
                if left:
                    left = False
                    space1 = " "
                    space2 = ""
                else:
                    left = True
                    space1 = ""
                    #space2 = " "
            if word == ".":
                all_sentences.append(sentence + word)
                all_sentences.append(str(isNer))
                space1 = ""
                space2 = None
                sentence = ""
                isNer = 0
                left = True
                continue
            if (space1 == None) and (not sentence == ""):
                space1 = " "
            if key_word == key:
                word = "<" + word + ">"
                isNer = 1
            if space2 == None:
                sentence += (space1 + word)
                space1 = None
            else:
                sentence += (space1 + word + space2)
                space1 = ""
                space2 = None

    for sentence in all_sentences:
        print(sentence)
    with open("./Outputs/HebrewText_3000_Sen_NER_final.txt.txt","w", encoding="utf-8") as f:
        for sentence in all_sentences:
            f.write(sentence + "\n")


    return 0

read_file_to_tuple()