import datetime


dict_words = {}
dict_words["a"] = ("a","a","a",3)
dict_words["b"] = ("b","ba","bba")
dict_words["c"] = ("cc","ca","ccccca",4)

word = "a"

print(1,"=",dict_words[word][3])

if len(dict_words[word]) == 3:
    dict_words[word] = dict_words[word] + (1,)
else:
    dict_words[word] = dict_words[word][:2] + (dict_words[word][3] + 1,)

print(2,"=",dict_words[word][3])


exit





with open(r'.\EnglishWords.txt') as word_file:
    words = list(word.strip().lower() for word in word_file)
    dic = {}  
    for word in words:
        dic[word] = None
    start = datetime.datetime.now()

    if "work" in words:
        print("work")
    else:
        print("doesn't work")

    end = datetime.datetime.now()
    print("take:", end-start)
