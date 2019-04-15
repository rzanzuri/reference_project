import re
import csv
import random
import datetime

import Python_lib.Tokenizer as Tokenizer
import Python_lib.Statistics_text as Statistics


source_dir = r"./Data/"

def tokenizer_main():
    token_words = {}
    options = [1,2,3]
    name_file = './wiki_10'
    dict_token_file = "./wiki_10_tokens_lines_0-5000.csv"
    Tokenizer.load_tokens(dict_token_file)
    count = 5000
    num = 0
    #token_file = name_file + "_tokens_lines_" + str(num) + "-" + str(num + count) + ".csv"
    
    #triple_file = open(name_file + "_triple", "w", encoding='utf-8')
    for j in [1,2,3]:
        with open(name_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                #line = "As a result, the St. Gallen Grand Council agreed on 17 May 1887 to a request for a government loan of Swiss francs (CHF) 7000 for preparatory work for a rail link from the Linth area via the Toggenburg to St. Gallen. The first expert opinion recommended a gap between Ebnat and Uznach, but this would still have required a detour via Wil to reach St. Gallen. An initiative committee ('Initiativkomitee') for a St. Gallen–Herisau–Degersheim–Neckertal–Wattwil–Rapperwil railway link was formed in Degersheim in 1889. The leader was the Degersheim embroidery manufacturer Isidor Grauer-Frey, who also campaigned for an extension of the line beyond Rapperswil to Zug in order to make a connection to the Gotthard Railway. The maximum grade of 5.0% planned for the Zürichsee–Gotthardbahn—the later Schweizerische Südostbahn (SOB)—seemed to him unsuitable for main-line traffic. In 1889, the Grand Council granted the initiative committee a contribution of CHF 5,000 to submit an application for a concession to St. Gallen–Zug. This concession was granted by the Federal Assembly on 27 June 1890."
                if num > count:
                    print(f"Total lines is:{i}")
                    break
                if(line == "\n" or line == ' ' or len(line.split(" ")) <= 3):
                    continue
                line = line.rstrip()
                new_line = ""
                words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\'s|\w+\.\w+|\w+|\S',line)
                #words = ["unlike","playing", "rerun", "remember", "doing", "does", "refael","feet","ate", "opened", "inside", "united", "385","386.9","987$"]
                for word in words:
                    word = word.lower()
                    choice = random.choice(options)
                    if choice == 1: # take the word as is
                        new_line += word + " "
                    elif choice == 2: # split to tokens
                        if word in token_words:
                            new_line += Tokenizer.get_token_as_word(token_words[word]) + " "
                            print(token_words[word])
                        else:
                            new_line += Tokenizer.get_token_as_word(Tokenizer.split_to_tokens(word)) + " "
                            print(Tokenizer.get_token_as_word(Tokenizer.split_to_tokens(word)))
                    elif choice == 3:
                        new_line += " ".join(c for c in list(word)) + " "
                new_line = new_line.rstrip()
                print("line:", line)
                print("new_line:", new_line)
                num += 1

    #save_tokens(token_file)
    return 0

def statistics_main():
    fname = source_dir + 'wiki_10'
    #Statistics.check_ave_of_token(fname + "_tokenizer.csv")
    Statistics.check_ave_length_word(fname)
    #Statistics.checking_tokenizer_of_all_text(fname, fname + "_tokenizer.csv")
    #Tokenizer.save_tokens(fname + "_tokenizer.csv",dict_words)
    #Statistics.word_average(fname, lang)

if __name__ == "__main__":
    start = datetime.datetime.now()
    print("start:", start)

    #tokenizer_main()
    #statistics_main()    
    Tokenizer.tokenizer("this sets a minimum bound on the number of times a bigram needs to appear before it can be considered a collocation, in addition to log likelihood statistics", {}, 0)
    #Tokenizer.tokenizer("collocation", {}, 0)

    finish = datetime.datetime.now()
    print("end:", finish)
    print("total:", finish - start)

