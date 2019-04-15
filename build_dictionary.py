#pip install --upgrade google-cloud-translate
import re
import datetime
import Tokenizer
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize
dict_words = {}

def word_average(fname, lang):
    name_file = fname + "_" + lang
    dict_file = open(name_file + "_dictionary.csv", "w", encoding='utf8')
    with open(name_file, encoding='utf8') as f:
        for line in f:
            if(line == "\n" or line == ' '):
                continue
            line = line.rstrip()
            words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+\.\w+|\w+|\S',line)
            for word in words:
                #print(word)
                word = word.lower()
                if word in dict_words:
                    dict_words[word] += 1
                else:
                    dict_words[word] = 1

    print("word", "count", file=dict_file, sep=',')
    words_count = 0
    for word in dict_words:
        try:
            #dict_file.write('"' + word + '","' + dict_words[word] + '"')
            words_count += dict_words[word]
        except:
            print(f"{word} is not wrote to tokens.")
    print("average is:", words_count/len(dict_words), file=dict_file)
    print("average is:", words_count/len(dict_words))
    dict_file.close()  

def check_ave_length_word(fname):
    with open(fname, encoding='utf8') as f:
        for line in f:
            if(line == "\n" or line == ' '):
                continue
            line = line.rstrip()
            words = line.split()
            for word in words:
                #print(word)
                word = word.lower()
                if word in dict_words:
                    dict_words[word] += 1
                else:
                    dict_words[word] = 1
    total_words = 0
    total_length_words = 0
    for word in dict_words:
        total_words += dict_words[word]
        total_length_words += dict_words[word] * len(word)
    print("Total words:", total_words)
    print("Total length words:", total_length_words)
    print("Avarege length word:", total_length_words/total_words)



def checking_tokenizer_of_all_text (fname, dict_file):
    length = 10000
    count = 0

    Tokenizer.load_tokens(dict_file, dict_words)
    with open(fname,encoding='utf8') as f:
        for line in f:
            count += 1
            if count > length:
                Tokenizer.save_tokens(fname + "_tokenizer.csv", dict_words)
                count = 0
            if(line == "\n" or line == ' '):
                continue
            line = line.rstrip()
            words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+\.\w+|\w+|\S',line)
            for word in words:
                #print(word)
                word = word.lower()
                if word in dict_words:
                    if len(dict_words[word]) > 3:
                        dict_words[word] = dict_words[word][:3] + (dict_words[word][3] + 1,)
                    else:
                        dict_words[word] = dict_words[word] + (1,)
                else:
                    dict_words[word] = Tokenizer.split_to_tokens(word) + (1,)
        
def check_ave_of_token (fname):
    Tokenizer.load_tokens(fname, dict_words,with_count=1)
    word_count_regular = 0
    word_count_tokens = 0
    for word in dict_words:
        if len(dict_words[word]) > 3:
            word_count_regular += int(dict_words[word][3])
            token_length = 1
            if len(dict_words[word][0]) > 0:
                token_length += 1
            if len(dict_words[word][3]) > 0:
                token_length += 1
            word_count_tokens += token_length * int(dict_words[word][3])
        else:
            print(dict_words[word])
    print(word_count_regular)
    print(word_count_tokens)
    print("total:", word_count_tokens/word_count_regular)


if __name__ == "__main__":
    start = datetime.datetime.now()
    print("start:", start)
    lang = "he"
    fname = './wiki_10'
    #check_ave_of_token(fname + "_tokenizer.csv")
    check_ave_length_word(fname)
    #checking_tokenizer_of_all_text(fname, fname + "_tokenizer.csv")
    #Tokenizer.save_tokens(fname + "_tokenizer.csv",dict_words)
    #word_average(fname, lang)

    finish = datetime.datetime.now()
    print("end:", finish)
    print("total:", finish - start)