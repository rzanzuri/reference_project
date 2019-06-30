import re
import csv
import random
import datetime
import sys
import _thread
from threading import Thread
import subprocess
import time
import os
import asyncio
#os.environ['http_proxy'] = 'http://proxy-chain.intel.com:911'
#os.environ['HTTP_PROXY'] = 'http://proxy-chain.intel.com:911'
#os.environ['https_proxy'] = 'https://proxy-chain.intel.com:912'
#os.environ['HTTPS_PROXY'] = 'https://proxy-chain.intel.com:912'

from os import listdir,mkdir,rmdir
from os.path import isfile, join,isdir

import Python_lib.Tokenizer as Tokenizer
# import Python_lib.Statistics_text as Statistics
import Python_lib.read_html as read_html
import Python_lib.textHandler as textHandler
import Python_lib.named_entity_recognotion as ner


source_dir = r"./Data/"
mutex = asyncio.Lock()

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

def parse_html_files():
    my_dir = r"C:/Users/rzanzuri/Desktop/reference_project/Data/online"
    outputs_dir = my_dir + "_text"
    #rmdir(outputs_dir)
    #mkdir(outputs_dir)
    files = [f for f in listdir(my_dir) if isfile(join(my_dir, f))]
    num = 0
    for html_file in files:
        if html_file.endswith(".html") and  html_file.startswith("f_0"):
            #if num == 100:
            #    exit()
            num += 1
            #read_html.read_url_2_text()
            html = read_html.read_url_2_text("file:///" + join(my_dir, html_file))
            text = read_html.clean_html_page(html)
            with open(join(outputs_dir, html_file.replace("html", "txt")), "w", encoding="utf-8") as f:
                f.write(text)

def split_text_to_sentence(fname, delimiter = "."):
     sentences = []
     with open(fname, encoding="utf-8") as f:
          text = f.read()
          sentences = text.split(delimiter)
     with open(fname.replace(".txt","_sentences.txt"),"w", encoding="utf-8") as f_sen:
          for sentence in sentences:
               f_sen.write(sentence + delimiter + "\n")

def write_text_to_file_from_url(url, start, amount, website_name, dir_name, html_part = ["p"], string=1):
    if(not isdir(f"{dir_name}")):
        try:
            mkdir(f"{dir_name}")
        except:
            print("failed to create dir:", dir_name)
    text_10 = ""
    for i in range(start, start + amount - 1):
        try:
            #if isfile(f"./Data/download_pages/{website_name}_{i}.txt"): continue
            html = read_html.read_url_2_text(f"{url}{i}")
            if html == "":
                continue
            text = read_html.parse_html_to_text(html, html_part, string) + "\n"
            if len(text) > 10:
                text_10 += text
            if i % 10:
                with open(f"{dir_name}/{website_name}_from_{start}_to_{start+amount-1}.txt", "a", encoding="utf-8") as f:
                    f.write(text_10)
                text_10 = ""
        except:
            print("Unexpected error:", sys.exc_info()[0])
    with open(f"{dir_name}/{website_name}_from_{start}_to_{start+amount-1}.txt", "a", encoding="utf-8") as f:
        f.write(text_10)
    with open("./output.txt", "a") as f:
        print(f"finish thread: {start}", file=f)

def run_threads(thread_count, start, count, url, name,tag=["p"], string=1):

    threads = []
    try:
        dir_name = f"./Data/download_pages/{name}/from_{start}_to_{start+thread_count*count-1}"
        for i in range(0, thread_count-1):
            threads.append(Thread(target=write_text_to_file_from_url, args=(url, start + i * count, count, name, dir_name, tag,string)))

        [t.start() for t in threads]
        [t.join() for t in threads]

    except:
        print ("Error: unable to start thread")

def reorg_files(my_dir = "", path_to_save = None, lan = None):
    files = [f for f in listdir(my_dir)]
    for my_file in files:
        if isdir(join(my_dir,my_file)):
            path = path_to_save
            if path_to_save == None:
                path = (my_dir, my_file + ".txt")
            if my_file == "Hebrew" or my_file == "English":
                reorg_files( join(my_dir, my_file), path_to_save = path, lan = my_file)
            else:
                reorg_files( join(my_dir, my_file), path_to_save = path)
        elif isfile(join(my_dir,my_file)) and lan != None and path_to_save != None and my_file == "merged.txt":
            source = open(join(my_dir, my_file),'r', encoding='utf-8') 
            content = source.read()
            if lan == "Hebrew":
                content = read_html.remove_nikud(content)
            source.close()

            if not isdir(join(path_to_save[0], lan)):
                mkdir(join(path_to_save[0], lan))
            dest = open(join(path_to_save[0], lan, path_to_save[1]), 'a',encoding='utf-8')
            dest.write(re.sub(r"\n\n+", "\n", content) + "\n\n")
            dest.close()
def build_stem_text_that_contain_all_words(my_file):
    all_words = set()
    content = []
    persent = 0
    #thread_count = 100
    #with open(my_file.replace(".txt", "2_dict_full.txt"), encoding = "utf-8") as dic:
    #       words = dic.read().split("\n")
    #all_words.update(words)
    # print("finish to read the source and create dict.\n", datetime.datetime.now())

    # with open(my_file, encoding="utf-8") as f:
    #    lines = f.read().split(".")
    #    lines_per_task = int(len(lines) / thread_count)
    #    threads = []
    #    try:
    #        for i in range(thread_count):
    #            if i == 99:
    #                threads.append(Thread(target=aaa, args=(my_file.replace(".txt", f"_stem_{i}.txt"),lines[lines_per_task*i : ] , all_words, i)))
    #            else:
    #                threads.append(Thread(target=aaa, args=(my_file.replace(".txt", f"_stem_{i}.txt"),lines[lines_per_task*i : lines_per_task * (i+1)] , all_words, i)))

    #        [t.start() for t in threads]
    #        [t.join() for t in threads]
    #    except:
    #        print ("Error: unable to start thread")
    out = open(my_file.replace(".txt", "_stem2.txt"),"w", encoding="utf-8")
    with open(my_file, encoding="utf-8") as f:
        lines = f.read().split(".")
        for i,line in enumerate(lines):
            added_line = 0
            words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+\.\w+|\w+|\S',line + ".")
            for word in words:
                if word not in all_words:
                    all_words.update([word])
                    if added_line == 0:
                        content.append(line)
                        added_line = 1
                        print(line, file = out)
                        if persent < int(i*100/len(lines)): 
                            persent = int(i*100/len(lines))
                            print("words count is:", len(all_words))
                            print(f"completed {int(i*100/len(lines))} %, total lines :", len(all_words))
    out.close()

def run_in_parallel_go_command(start , count):
    #threads = []
    base = r"C:/Users/rzanzuri/Desktop/reference_project/"
    try:
        for i in range(start, start + count):
            #threads.append(Thread(target=run_thread, args=(f"full_hebrew_stem_{i}", base + r"Data/hebrew_data/in", base + r"Data/hebrew_data/out" , base + r"yap/src/yap" )))
            run_thread(f"full_hebrew_stem_{i}", base + r"Data/hebrew_data/in", base + r"Data/hebrew_data/out" , base + r"yap/src/yap" )
            #threads.append(Thread(target=run_java, args=(f"full_hebrew_stem_{i}","")))
            #run_java(f"full_hebrew_stem_{i}","")
        #[t.start() for t in threads]
        #[t.join() for t in threads]
    except:
        print ("Error:", sys.exc_info())

def run_thread(file_name, in_path, out_path, yap_path,i):
    print("run_thread.")
    yap_command = join(yap_path,"yap.exe")
    raw = " -raw " + join(in_path, file_name + ".txt")
    out = " -out " + join(out_path, file_name + ".lattice")
    inn = " -in " + join(out_path, file_name + ".lattice")
    #osr = " -os " + join(out_path, file_name + ".segmentation")
    om = " -om " + join(out_path, file_name + ".mapping")
    #oc = " -oc " + join(out_path, file_name + ".conll")

    subprocess.run(yap_command + " hebma" + raw + out)
    subprocess.run(yap_command + " md" + inn + om )
    #subprocess.run(yap_command + " joint" + inn + osr + om + oc )
    os.remove(join(out_path, file_name + ".lattice"))
    #os.remove(join(out_path, file_name + ".conll"))
    

def run_java(my_file,sss):
    base = r"C:\Users\rzanzuri\Desktop\reference_project"
    tagger = join(base, "Tagger\\")
    command = f"java -Xmx2G -XX:MaxPermSize=256m -cp trove-2.0.2.jar;{tagger}morphAnalyzer.jar;{tagger}opennlp.jar;{tagger}gnu.jar;{tagger}chunker.jar;{tagger}splitsvm.jar;{tagger}duck1.jar;{tagger}tagger.jar;. NewDemo {tagger} "
    in_file = join(base, r"Data\hebrew_data\in_adler2", my_file + ".txt")
    out = join(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\out2", my_file + "_adler.txt")
    subprocess.run(command + " " + in_file + " " + out )
    print(command + " " + in_file + " " + out)


def split_file(my_dir, my_file, to_size):
    with open(join(my_dir, my_file), encoding = "utf-8") as f:
        lines = f.readlines()
        num_file = 0
        split_file = open(join(my_dir, "in", my_file.replace(".txt", f"_{num_file}.txt")), "w", encoding = "utf-8")
        for line in lines:
            if int(os.path.getsize(split_file.name)) >= to_size:
                print("\n", sep="\n" ,file = split_file)
                split_file.close()
                num_file += 1
                #if num_file > 5:
                #    exit()
                split_file = open(join(my_dir, "in", my_file.replace(".txt", f"_{num_file}.txt")), "w", encoding = "utf-8")
            else:
                if line == "\n":
                    continue
                #For many adler uncomment thile follow line.
                #print(line.rstrip(), file = split_file)
                #For Ruth remove the comment from the follow 3 lines.
                words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+[\"\'\.]\w+|\w+|\S',line + ".")
                print(*words, sep="\n", end="\n\n" ,file = split_file)
        print("\n", sep="\n" ,file = split_file)
        split_file.close()

def parse_tzarfati_tagger(my_dir, test_file):
    dictionary = {}
    # with open(test_file, encoding="utf-8") as s:
    #     print_with_time("Reading the test file.")
    #     text = s.read()
    #     print_with_time("Finish to read the file\t now splitting.")
    #     words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+[\"\'\.]\w+|\w+|\S',text + ".")
    #     print_with_time("Finish to split,\t now set them.")
    #     uniq_words = set(words)
    #     print_with_time("Finish to set the test.")
    # return the name of files without the format 
    files = [f for f in listdir(my_dir) if isfile(join(my_dir,f))]
    files = [f.split(".")[0] for f in files]
    files = list(set(files)) # return uniq names
    sorted(files)
    for f in files:
        if not isfile(join(my_dir,f+".mapping")):
            print(f"Missing file: {f}.")
            continue
        else:
            parse_one_file_2(join(my_dir, f), dictionary)
    write_dictionary_to_file(dictionary, f"{my_dir}\\heb_dictionaty.csv")

def parse_one_file(my_file, dic):
    print(f"Parsing file: {my_file}")
    mapping = open(my_file + ".mapping", encoding="utf-8").readlines()
    mapping = [re.split(r"\s", a) for a in mapping]
    segmentation = open(my_file + ".segmentation", encoding="utf-8").readlines()
    segmentation = [re.split(r":", re.sub(r"\s", "", a)) for a in segmentation]
    j = 0
    for i in range(len(segmentation)):
        root = prefix = suffix = rest = ""
        mapp = mapping[j]
        seg  = segmentation[i]
        word = "".join(segmentation[i])

        if word in dic or word == "":
            j += 1
            continue
        prefix = ""
        if len(seg) > 1:
            prefix = "".join(seg[:len(seg) -1])
        rest = seg[len(seg)-1]
        old_j = j
        flag = False
        for k in range(10):
            if rest in mapp:
                flag = True
                break
            j += 1
            mapp = mapping[j]
        if not flag:
            print(f"line: {i}, j: {j}.")
            j = old_j + 1
            continue
        #print(mapp[3], rest)
        if "yy" in mapp[4]:
            root, suffix = mapp[2], ""
        else:
            root, suffix = get_root_and_suff(mapp[3], rest )
        dic[word] = (prefix, root, suffix)
        j += 1

def parse_one_file_2(my_file, dic):
    print(f"Parsing file: {my_file}")
    mapping = open(my_file + ".mapping", encoding="utf-8").readlines()
    mapping = [re.split(r"\s+", a) for a in mapping]
    prefixes = ["DEF", "PREPOSITION", "CONJ", "REL","ADVERB", "TEMP"]
    prefix = ""
    for i in range(len(mapping)):
        #if i > 9700: print("ppp")
        word = word2 = root = suffix = ""
        mapp = mapping[i]
        if len(mapp) < 4 :
            prefix = ""
            continue
        if mapp[4] in prefixes or (mapp[4]=="IN" and len(mapp[3]) == 1):
            prefix += mapp[3]
            continue
        if "yy" in mapp[4]:
            word = mapp[2]
        else:
            word = prefix + mapp[2]
            if "ה" in prefix:
                word2 = prefix.replace("ה", "", 1) + mapp[2]

        if word in dic or word == "":
            prefix = ""
            continue
        if word == "להבאבלסהרשות":
            print("להבאבלסהרשות")
        if "yy" in mapp[4]:
            prefix, root, suffix = "", mapp[2], ""
        else:
            root, suffix = get_root_and_suff(mapp[3], mapp[2] )

        dic[word] = (prefix, root, suffix)
        if word2 != "":
            dic[word2] = (prefix.replace("ה","",1), root, suffix)
        prefix = ""

def get_root_and_suff(root, word):
    if len(root) > len(word):
        return word, ""
    try:
        if word == root: return root, ""
        for i in range(len(root)):
            if word[i] != root[i] and not_sofit(word[i], root[i]):
                if i < 2:
                    return word, ""
                return root, word[i:]
        return root, word[len(root):]
    except:
        print(f"Exception word: {word}, root: {root}.")
        return word, ""

def write_dictionary_to_file(dic, f):
    with open(f, "w", encoding ="utf-8") as dic_file:
        print("word", "prefix", "root", "suffix", file=dic_file, sep=',')
        for key in dic:
            if '"' in key:
                print('"' + str(key).replace('"','&#&'), dic[key][0].replace('"','&#&'), dic[key][1].replace('"','&#&'),  dic[key][2].replace('"','&#&') + '"', file=dic_file , sep='","')
            else:
                print('"' + str(key), dic[key][0], dic[key][1],  dic[key][2] + '"', file=dic_file , sep='","')

def not_sofit(a,b):
    if (a == "כ" and b == "ך") or (a == "ך" and b == "כ") or (a == "מ" and b == "ם") or (a == "ם" and b == "מ") or (a == "נ" and b == "ן") or (a == "ן" and b == "נ") or (a == "פ" and b == "ף") or (a == "ף" and b == "פ") or (a == "צ" and b == "ץ") or (a == "ץ" and b == "צ"):
        return False
    return True

def triangle_text(my_file, token_file, vector):
    dic = Tokenizer.load_tokens(token_file)
    new_lines = []
    missing_words = 0
    persent = 0
    with open(my_file, encoding="utf-8") as f:
        lines = f.readlines()
        print(f"count lines before: {len(lines)}.")
        lines += lines
        print(f"count lines after: {len(lines)}.")
        j = 0
        for k, line in enumerate(lines):
            if persent < k / len(lines) :
                print_with_time(f"{persent*100}% Done.")
                persent += 0.1
            new_line = ""
            words = re.findall(r'\d+[.,]\d+\([.,]\d\)*|\w+[\"\']\w+|\w+|\S',line)
            for i, word in enumerate(words):
                if j % len(vector) == 0:
                    random.shuffle(vector)
                if i > 0:
                    new_line += " "
                j += 1
                if vector[j % len(vector)] == 0: # For word
                    new_line += word
                elif vector[j % len(vector)] == 1: # For tokens 
                    if word in dic:
                        new_line += " ".join(list(dic[word]))
                    else:
                        new_line += word
                        missing_words += 1
                        #print(f"The word {word} is missing from the dicinary, till now missing {missing_words} words.")
                elif vector[j % len(vector)] == 2: # For letters
                    new_line += " ".join(list(word))
            new_lines.append(new_line)
    print(f"missing words: {missing_words}.")
    with open(my_file.replace(".txt", "_triangle.txt"),"w", encoding="utf-8") as triangle:
        for line in new_lines:
            triangle.write(line.replace("   "," ").replace("  ", " "))

def build_sentenses_file(my_file, out_file):
    with_ner, without_ner = get_sentenses(my_file)
    print("Finish to build the sentenses.")
    final_lines = []
    for i in range(1100):
        rand = random.randint(0,99)
        if rand % 2 == 0 and len(with_ner) > 0:
            final_lines.append(with_ner.pop().rstrip())
            final_lines.append("1")
        elif rand % 2 == 1 and len(without_ner) > 0:
            final_lines.append(without_ner.pop().rstrip())
            final_lines.append("0")
    
    print("Start to write the out file.")
    with open(out_file, "w", encoding = "utf-8") as out:
        for i,line in enumerate(final_lines):
            if i >= 2000:
                break
            print(line, file=out)


def get_sentenses(my_file):
    print("Start to reading...")
    #source_lines = textHandler.get_sentences_from_file(my_file)
    #print("Finish reading.")
    #random.shuffle(source_lines)
    #print("Finish shuffling.")
    source_lines = []
    with open(my_file + "2.txt", encoding = "utf-8") as f:
        source_lines += f.readlines()
    with_ner = []
    without_ner = []
    for line in source_lines:
        if len(with_ner) >= 550 and len(without_ner) >= 550: return with_ner, without_ner
        is_ner = ner.is_ner_exsits(line)
        if is_ner and len(with_ner) <= 550:
            with_ner.append(line)
        elif not is_ner and len(without_ner) <= 550:
            without_ner.append(line) 
    return with_ner, without_ner

def print_with_time(text):
    print(datetime.datetime.now(), text)        

def build_dict_heb_statistics(my_file, dictionary_file):
    #dic = Statistics.checking_tokenizer_of_all_text(my_file,dictionary_file)
    #Statistics.check_ave_of_token(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\full_hebrew_part_1.txt_tokenizer.csv")
    Statistics.check_ave_length_word(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\HebrewTextPart\HebrewTextPart.txt")

def create_vector(words, partials, letters = 0):
    vector = []
    for i in range(words):
        vector.append(0)
    for i in range(partials):
        vector.append(1)
    for i in range(letters):
        vector.append(2)
    
    random.shuffle(vector)

    return vector




if __name__ == "__main__":
    start = datetime.datetime.now()
    print("start:", start)

    #tokenizer_main()
    #statistics_main()    
    #Tokenizer.tokenizer("this sets a minimum bound on the number of times a bigram needs to appear before it can be considered a collocation, in addition to log likelihood statistics", {}, 0)
    #Tokenizer.tokenizer("collocation", {}, 0)
    #parse_html_files()
    #split_text_to_sentence("C:/Users/rzanzuri/Desktop/reference_project/Data/online_text/f_01682.txt")
    #write_text_to_file_from_url("https://news.walla.co.il/item/", 2600000, 10000, "walla")
    #run_threads(100,3000000, 1000, "https://news.walla.co.il/item/", "walla")
    #run_threads(thread_count=100, start=20000, count=100, url="http://www.hidush.co.il/hidush.asp?id=", name="hidush",tag=["span","p"],string=0)
    #reorg_files(my_dir = r"C:/Users/rzanzuri/Desktop/reference_project/Data/Sefaria-Export-master/txt")
    #textHandler.clean_hebrew_text_from_dir(r"C:/Users/rzanzuri/Desktop/hebrew_data","full_hebrew.txt")
    #build_stem_text_that_contain_all_words(r"C:/Users/rzanzuri/Desktop/hebrew_data/full_hebrew.txt")
    # split_file("D:\\Final Project\\reference_project\\Data\\hebrew_data\\full_hebrew_stem.txt", 0.1*1024*1024)
    run_in_parallel_go_command(100,100)
    run_in_parallel_go_command(300,100)
    run_in_parallel_go_command(500,100)
    run_in_parallel_go_command(700,100)
    run_in_parallel_go_command(900,100)
    run_in_parallel_go_command(1100,100)
    run_in_parallel_go_command(1300,100)
    run_in_parallel_go_command(1500,100)
    run_in_parallel_go_command(1700,100)
    run_in_parallel_go_command(1900,100)
    run_in_parallel_go_command(2100,100)
    run_in_parallel_go_command(2300,100)
    run_in_parallel_go_command(2500,100)
    run_in_parallel_go_command(2700,100)
    run_in_parallel_go_command(2900,100)
    #split_file(r"C:/Users/rzanzuri/Desktop/reference_project/Data/hebrew_data", "full_hebrew_stem.txt", 100*1024)

    #run_in_parallel_go_command(2000,100 )
    #run_in_parallel_go_command(2100,100 )
    #run_in_parallel_go_command(2200,100 )
    #parse_tzarfati_tagger(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\out", r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\full_hebrew_part_1.txt")
    #Statistics.check_ave_of_token(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\out\heb_dictionaty.csv")
    #triangle_text(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\full_hebrew_part1.txt", r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\out\heb_dictionaty.csv", create_vector(300,206))
    # with open(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\full_hebrew_part_1.txt",encoding="utf-8") as ff:
    #     text = ff.read().split(".")
    #     text = 
    #     with open(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\full_hebrew_part_1_1.txt","w",encoding="utf-8") as f:
    #         for line in text:
    #             print(line.rstrip(),file=f)
    #triangle_text(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\full_hebrew_part_1_1.txt", r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\out\heb_dictionaty.csv", create_vector(300,206))
    #build_sentenses_file(r"C:\Users\rzanzuri\Desktop\reference_project\Data\final_eng_text.txt", r"C:\Users\rzanzuri\Desktop\reference_project\Data\English_NER_1000.txt")
    build_dict_heb_statistics(r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\full_hebrew_part_1.txt",r"C:\Users\rzanzuri\Desktop\reference_project\Data\hebrew_data\out\heb_dictionaty.csv")
    
    finish = datetime.datetime.now()
    print("end:", finish)
    print("total:", finish - start)

