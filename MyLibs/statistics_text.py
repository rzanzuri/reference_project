#pip install --upgrade google-cloud-translate
import re
import datetime
import MyLibs.Tokenizer as Tokenizer
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize

# check how many times any word show in the fname in average
# input: fname - the file
# fname should be csv file
# output: {fname}_dictionary.csv
def word_average(fname):
    dict_words = {}
    dict_file = open(fname + "_dictionary.csv", "w", encoding='utf8')
    with open(fname, encoding='utf8') as f:
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

# check how many letter have in any word in average
# input: fname - the file with the text
# output: the average
def check_ave_length_word(fname):
    dict_words = {}
    with open(fname, encoding='utf8') as f:
        for line in f:
            if(line == "\n" or line == ' '):
                continue
            line = line.rstrip()
#            words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+[\"\']\w+|\w+|\S',line)
            words = re.findall(r'\d+[.,]\d+[.,]\d+|\d+[.,]\d+|\d+|\w+[\"\'\.]\w+|\w+|\S',line)
            for word in words:
                #print(word)
                if word in dict_words:
                    dict_words[word] += 1
                else:
                    dict_words[word] = 1
    #sorted(dict_words, key= dict_words.get)
    # print(*dict_words.keys()[-20:])
    total_words = 0
    total_length_words = 0
    for word in dict_words:
        total_words += dict_words[word]
        total_length_words += dict_words[word] * len(word)
    print("Total words:", total_words)
    print("Total length words:", total_length_words)
    print("Avarege length word:", total_length_words/total_words)
    return total_length_words/total_words

# check how many tokenizer have in average
# input: fname - the file to read
#        dict_file - file with old tokens
# output: {fname}_tokenizer.csv - new token files
#         dict_words - dictionary with the tokens (pre,root,suff,count) as value and the word as a key
def checking_tokenizer_of_all_text (fname, dict_file):
    length = 761060
    count = 0
    missing_words = 0
    total_words = 0
    percent = 0

    dict_words = Tokenizer.load_tokens(dict_file, 0)
    with open(fname,encoding='utf8') as f:
        lines = f.readlines()
        length = len(lines)
        for line in lines:
            count += 1
            if count / length > percent: 
                print(f"{(count/length)*100}% was done.",f"\tmissing words: {missing_words}.", f"\tTotal words: {total_words}")
                percent += 0.1
            if(line == "\n" or line == ' '):
                continue
            line = line.rstrip()
            #words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+\.\w+|\w+|\S',line) # For English
            words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+[\"\']\w+|\w+|\S',line) # For Hebrew
            total_words += len(words)
            for word in words:
                #print(word)
                if word in dict_words:
                    if len(dict_words[word]) > 3:
                        dict_words[word] = dict_words[word][:3] + (dict_words[word][3] + 1,)
                    else:
                        dict_words[word] = dict_words[word] + (1,)
                else:
                    #dict_words[word] = Tokenizer.split_to_tokens(word) + (1,)
                    dict_words[word] = ("",word,"",1)
                    #if '"' not in word and "'" not in word and "." not in word:
                    missing_words += 1
                        #print(f"The word: {word} wasn't exists in the dic.")
    print(f"missing words: {missing_words}.")
    Tokenizer.save_tokens(fname + "_tokenizer.csv", dict_words, 1)
    return dict_words

# loading token file and return the average of the tokens per word
def check_ave_of_token (fname, dic = None):
    missing_words = {}
    if dic == None:
        dict_words = Tokenizer.load_tokens(fname, with_count=1)
    else:
        dict_words = dic
    word_count_regular = 0
    word_count_tokens = 0
    count_word_with_partial = 0
    for word in dict_words:
        if len(dict_words[word]) > 3:
            word_count_regular += int(dict_words[word][3])
            token_length = 1
            if len(dict_words[word][0]) > 0:
                token_length += 1
            if len(dict_words[word][2]) > 0:
                token_length += 1
            if token_length > 1:
                count_word_with_partial += int(dict_words[word][3])
            word_count_tokens += token_length * int(dict_words[word][3])
        else:
            #print(f"missing word from dictt: {dict_words[word]}.")
            if (word not in missing_words):
                missing_words[word] = 1
            else:
                missing_words[word] += 1
    for word in missing_words:
        print(word + ":",missing_words[word])
    print(f"The count of regular words: {word_count_regular}.")
    print(f"The count of partial words: {word_count_tokens}.")
    print(f"The count of words with partial: {count_word_with_partial}.")
    print("total:", word_count_tokens/word_count_regular)
    print("total2 (how many word with partial):", count_word_with_partial/word_count_regular)

#check_ave_length_word(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\full_hebrew.txt")