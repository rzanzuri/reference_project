#import googletrans
import sys
from googletrans import Translator
translator = Translator()

start_from = 69
length = 0
num = 0
total_ratio = (0,0) # (count,average)
data = []
name_file = './wiki_10'
source = 'en'
destination = 'iw'
with open(name_file, encoding='utf-8') as f:
    src_file = open(name_file + '_' + source, "a", encoding='utf-8')
    dest_file = open(name_file + '_' + destination, "a",encoding='utf-8')
    for line in f:
        if(line == "\n" or line == ' ' or len(line.split(" ")) <= 5 or len(line) >= 5000):
            continue
        else:
            line = line.rstrip()
            if(length + len(line) < 5000): # colloecting the data            
                data.append(line)
                length += len(line)
            else: # have enough data to translate
                if(num >= start_from): # this line wasn't translated in the past
                    try:
                        translations = translator.translate(data, dest=destination, src=source)
                        for translation in translations:
                            # write to the files
                            src_file.write(translation.origin + "\n")
                            dest_file.write(translation.text + "\n")
                            # calculate the rational
                            len_orig = len(translation.origin.split(" "))
                            len_trans = len(translation.text.split(" "))
                            ratio = len_orig / len_trans
                            print("Line: ",total_ratio[0] + 1, "; English: ", len_orig, ' -> ',"Hebrew: ", len_trans, "; ratio: ",ratio )
                            # update variables
                            total_ratio = (total_ratio[0] + 1, ((total_ratio[1] * (total_ratio[0])) + ratio) / (total_ratio[0] + 1))
                        num += 1
                        data = [line] 
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
                    data = []
                
  
    src_file.close()
    dest_file.close()