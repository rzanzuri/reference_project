import paralleldots 
import re
paralleldots.set_api_key("VSdXYtzLhCXaGPzQXbSKhyMpzI795aATgFVekoN2EEA")

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
