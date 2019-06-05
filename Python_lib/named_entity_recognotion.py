#import paralleldots 
#paralleldots.set_api_key("9hFGKKNP1hfGuWCeKZZLZGx3If214gRcBNRRWJcayCg")

# pip install spacy 
# python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")

def get_ner_with_spacy(sentence):
    doc = nlp(sentence)

    #Find named entities, phrases and concepts
    result = []
    for entity in doc.ents:
        if(entity.label_ == "PERSON"):
            # print(entity.text)
            result.append(entity.text)
    return ", ".join(result)

#def get_ner (sentense):
    # output = paralleldots.ner(sentense)
    # if not "entities" in output: return None,None
    # for element in output['entities']:
    #     if element["category"] == "name":
    #         return element["name"],element["confidence_score"]
    # return None,None

def is_ner_exsits(sentence):
    ner_entities = get_ner_with_spacy(sentence)
    if(len(ner_entities) > 0):
        return 1
    else:
        return 0


def refactor_sentence_by_entity(sentence, name):
    subSent = sentence.split(name)
    ner_sent = subSent[0]
    sign = "name"

    for i in range(1, len(subSent)):
        ner_sent = ner_sent + sign + ' ' + name + ' ' + sign + subSent[i]

    return ner_sent

def get_ner_sentence (sentence):
    output = paralleldots.ner(sentence)
    if "entities" in output:
        for element in output['entities']:
            if (element["category"] == "name"):
                return refactor_sentence_by_entity(sentence,element["name"])
    return sentence


if __name__ == "__main__":
    name_file = './wiki_10'
    num = 0
    count = 10
    with open(name_file, encoding='utf-8') as f:
        with open("./sentenses.txt", "w") as result_file:
            #text = "The White House is the official residence and workplace of the President of the United States. It is located at 1600 Pennsylvania Avenue NW in Washington, D.C. and has been the residence of every U.S. President since John Adams in 1800. The term "+ '"White House"' + " is often used as a metonym for the president and his advisers. The residence was designed by Irish-born architect James Hoban in the neoclassical style. Hoban modelled the building on Leinster House in Dublin, a building which today houses the Oireachtas, the Irish legislature. Construction took place between 1792 and 1800 using Aquia Creek sandstone painted white. When Thomas Jefferson moved into the house in 1801, he (with architect Benjamin Henry Latrobe) added low colonnades on each wing that concealed stables and storage.In 1814, during the War of 1812, the mansion was set ablaze by the British Army in the Burning of Washington, destroying the interior and charring much of the exterior. Reconstruction began almost immediately, and President James Monroe moved into the partially reconstructed Executive Residence in October 1817. Exterior construction continued with the addition of the semi-circular South portico in 1824 and the North portico in 1829. Because of crowding within the executive mansion itself, President Theodore Roosevelt had all work offices relocated to the newly constructed West Wing in 1901. Eight years later in 1909, President William Howard Taft expanded the West Wing and created the first Oval Office, which was eventually moved as the section was expanded. In the main mansion, the third-floor attic was converted to living quarters in 1927 by augmenting the existing hip roof with long shed dormers. A newly constructed East Wing was used as a reception area for social events; Jefferson's colonnades connected the new wings. East Wing alterations were completed in 1946, creating additional office space. By 1948, the residence's load-bearing exterior walls and internal wood beams were found to be close to failure. Under Harry S. Truman, the interior rooms were completely dismantled and a new internal load-bearing steel frame constructed inside the walls. Once this work was completed, the interior rooms were rebuilt. The modern-day White House complex includes the Executive Residence, West Wing, East Wing, the Eisenhower Executive Office Building—the former State Department, which now houses offices for the President's staff and the Vice President—and Blair House, a guest residence. The Executive Residence is made up of six stories—the Ground Floor, State Floor, Second Floor, and Third Floor, as well as a two-story basement. The property is a National Heritage Site owned by the National Park Service and is part of the President's Park. In 2007, it was ranked second on the American Institute of Architects list of 'America's Favorite Architecture'."
            for line in f:
                if(line == "\n" or line == ' ' or len(line.split(" ")) <= 3 or len(line) >= 5000):
                    continue
                line = line.rstrip()
                sentences = line.split(".")
                print("number lines is: " + str(len(sentences)))
                for sentence in sentences:
                    result = get_ner_with_spacy(sentence)
                    if result == None or result == '':
                        result_file.write("missing for:\t'" + sentence + "'.\n")
                    else:
                        result_file.write(result + "for:\t'" + sentence + "'.\n")
                if num > count:
                    break
                num += 1
