# pip install yandex-translater
import sys
from yandex.Translater import Translater
translator = Translater()
translator.set_key('trnsl.1.1.20190228T112412Z.abfa0588f62c8726.d69833662a3ed1c8a58af0fc9a15a200bc5fdcf9') # Api key found on https://translate.yandex.com/developers/keys
#translator.set_text_format('utf-8')

start_from = 259
length = 0
num = 0
total_ratio = (0,0) # (count,average)
data = ""
capacity = 5000
name_file = './wiki_10'
source = 'en'
destination = 'he'
translator.set_from_lang(source)
translator.set_to_lang(destination)

with open(name_file, encoding='utf-8') as f:
    src_file = open(name_file + '_' + source, "a", encoding='utf-8')
    dest_file = open(name_file + '_' + destination, "a",encoding='utf-8')
    for line in f:
        if(line == "\n" or line == ' ' or len(line.split(" ")) <= 5 or len(line) >= 5000):
            continue
        else:
            line = line.rstrip()
            if(length + len(line) < capacity): # colloecting the data            
                data += line
                length += len(line)
            else: # have enough data to translate
                if(num >= start_from): # this line wasn't translated in the past
                    try:
                        translator.set_text(data)
                        translation = translator.translate()
                        # write to the files
                        src_file.write(data + "\n")
                        dest_file.write(translation + "\n")
                        # calculate the rational
                        len_orig = len(data.split(" "))
                        len_trans = len(translation.split(" "))
                        ratio = len_orig / len_trans
                        print("Line: ",total_ratio[0] + 1, "; English: ", len_orig, ' -> ',"Hebrew: ", len_trans, "; ratio: ",ratio )
                        # update variables
                        total_ratio = (total_ratio[0] + 1, ((total_ratio[1] * (total_ratio[0])) + ratio) / (total_ratio[0] + 1))
                        num += 1
                        data = line 
                        length = len(line)
                        print("---------------- The number of the translation till now is: ", num, " -----------------" )
                    except:
                        print("You have an error: ", sys.exc_info()[0])
                        print("---------------- Total ratio is: ", total_ratio[1], " -----------------" )
                        print("---------------- The number of the translation till now is: ", num, " -----------------" )
                        src_file.close()
                        dest_file.close()
                        break
                else:
                    num += 1
                    data = ""
                
  
    src_file.close()
    dest_file.close()