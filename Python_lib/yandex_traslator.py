# pip install yandex-translater
import sys
import os
from yandex.Translater import Translater
translator = Translater()
keys = []
# translator.set_key('trnsl.1.1.20190228T112412Z.abfa0588f62c8726.d69833662a3ed1c8a58af0fc9a15a200bc5fdcf9') # Api key found on https://translate.yandex.com/developers/keys
# translator.set_key('trnsl.1.1.20190310T193008Z.cb8fa6e506b8f839.eefc45434b6c3467d0179fa9524fbd7aa144c86a')
# translator.set_key('trnsl.1.1.20190310T193633Z.bbb8dcc3120d02b3.fd9b78bfa8d658c235e9e23970943f55141f0826')
# translator.set_key('trnsl.1.1.20190310T193827Z.3325c6145924dd82.14cbc482d2530b71311743d52c71735e5df310a1')

keys.append('trnsl.1.1.20190228T112412Z.abfa0588f62c8726.d69833662a3ed1c8a58af0fc9a15a200bc5fdcf9') 
keys.append('trnsl.1.1.20190310T193008Z.cb8fa6e506b8f839.eefc45434b6c3467d0179fa9524fbd7aa144c86a')
keys.append('trnsl.1.1.20190310T193633Z.bbb8dcc3120d02b3.fd9b78bfa8d658c235e9e23970943f55141f0826')
keys.append('trnsl.1.1.20190310T193827Z.3325c6145924dd82.14cbc482d2530b71311743d52c71735e5df310a1')
#translator.set_text_format('utf-8')

# start_from = 1148
start_from = 176
length = 0
num = 0
total_ratio = (0,0) # (count,average)
data = ""
capacity = 5000
name_file = os.getcwd() + r'\wiki_03'
source = 'he'
destination = 'en'
translator.set_from_lang(source)
translator.set_to_lang(destination)
translator.set_key(keys.pop())

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
                        print("Line: ",total_ratio[0] + 1, "; Hebrew: ", len_orig, ' -> ',"English: ", len_trans, "; ratio: ",ratio )
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
                        if(len(keys) > 0):
                            translator.set_key(keys.pop())
                            continue
                        src_file.close()
                        dest_file.close()
                        break
                else:
                    num += 1
                    data = ""
                    length = 0
                
  
    src_file.close()
    dest_file.close()