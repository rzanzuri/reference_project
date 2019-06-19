import Python_lib.Tokenizer as Tokenizer

aaa = Tokenizer.load_tokens(r"C:\Users\rzanzuri\Desktop\reference_project\text.txt",0)
with open(r"C:\Users\rzanzuri\Desktop\reference_project\text.1.txt", encoding="utf-8") as f:
    b = f.readlines()
    for zz in aaa:
        print(zz)
    for x in b:
        print (aaa[x.strip()][1])
#print(aaa[b][1])
#print(*aaa['????"?'])
#print(*aaa["???'?"])

print("fff")