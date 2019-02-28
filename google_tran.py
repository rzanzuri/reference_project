import goslate
gs = goslate.Goslate()
start_from = 5
num = 0
total_ratio = 0
data = ""
print(gs.translate("Promoted to commander on 5 November 1745, Howe was commanding officer of the sloop HMS 'Baltimore' in the North Sea during the Jacobite rising of 1745 and was severely wounded in the head while cooperating with a frigate in an engagement with two French privateers. Promoted to post-captain on 10 April 1746, he was given command of the sixth-rate HMS 'Triton' and took part in convoy duties off Lisbon. He transferred to the command of the fourth-rate HMS 'Ripon' in Summer 1747 and sailed to the West Indies before becoming Flag Captain to Admiral Sir Charles Knowles, Commander-in-Chief, Jamaica, in the third-rate HMS 'Cornwall' in October 1748. He was given command of the fifth-rate HMS 'Glory' off the coast of West Africa in March 1751 and then transferred to the command of the sixth-rate HMS 'Dolphin' in the Mediterranean Fleet in June 1752.", 'iw'))
with open('./wiki_10', encoding = "UTF-8") as f:
    with open('./wiki_10_heb', "a") as hebrew_f:
        with open('./wiki_10_eng', "a") as english_f:
            total_ratio = 0
            for line in f:
                if(line == "\n" or line == ' ' or len(line.split(" ")) <= 5):
                    continue
                else:
                    try:
                    #line = line.rstrip()
                        if(num >= start_from):
                            trans = gs.translate(line, 'iw')
                            hebrew_f.write(trans)
                            english_f.write(line)
                            orig_len = len(line.split(" "))
                            trans_len = len(trans.split(" "))
                            ratio = orig_len / trans_len
                            num += 1
                            print("Line: ",num, "; English: ", orig_len, ' -> ',"Hebrew: ", trans_len, "; ratio: ",ratio )
                            count = num - start_from
                            total_ratio = ((total_ratio * (count - 1)) + ratio) / count 
                        else:
                            num += 1
                    except:
                        print("You have an error in this line:", line)
                        print("---------------- Total ratio is: ", total_ratio, " -----------------" )
                        print("---------------- Exit status is: 1 -----------------" )
                        break
print("---------------- Total ratio is: ", total_ratio, " -----------------" )
print("---------------- Exit status is: 0 -----------------" )


