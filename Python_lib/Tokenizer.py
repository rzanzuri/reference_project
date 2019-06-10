#pip install --upgrade google-cloud-translate
import re
import datetime
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer 
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize
import spacy
import csv
import random
nlp = spacy.load("en_core_web_sm")
# token_words = {}

# From https://dictionary.cambridge.org/grammar/british-grammar/word-formation/prefixes
english_prefixes = {
"anti": "",    # e.g. anti-goverment, anti-racist, anti-war
"auto": "",    # e.g. autobiography, automobile
"de": "",      # e.g. de-classify, decontaminate, demotivate
"dis": "",     # e.g. disagree, displeasure, disqualify
"down": "",    # e.g. downgrade, downhearted
"extra": "",   # e.g. extraordinary, extraterrestrial
"hyper": "",   # e.g. hyperactive, hypertension
"il": "",     # e.g. illegal
"im": "",     # e.g. impossible
"in": "",     # e.g. insecure
"ir": "",     # e.g. irregular
"inter": "",  # e.g. interactive, international
"mega": "",   # e.g. megabyte, mega-deal, megaton
"mid": "",    # e.g. midday, midnight, mid-October
"mis": "",    # e.g. misaligned, mislead, misspelt
"non": "",    # e.g. non-payment, non-smoking
"over": "",  # e.g. overcook, overcharge, overrate
"out": "",    # e.g. outdo, out-perform, outrun
"post": "",   # e.g. post-election, post-warn
"pre": "",    # e.g. prehistoric, pre-war
"pro": "",    # e.g. pro-communist, pro-democracy
"re": "",     # e.g. reconsider, redo, rewrite
"semi": "",   # e.g. semicircle, semi-retired
"sub": "",    # e.g. submarine, sub-Saharan
"super": "",   # e.g. super-hero, supermodel
"tele": "",    # e.g. television, telephathic
"trans": "",   # e.g. transatlantic, transfer
"ultra": "",   # e.g. ultra-compact, ultrasound
"un": "",      # e.g. under-cook, underestimate
"up": "",      # e.g. upgrade, uphill
}

porter = PorterStemmer()
stemmer = SnowballStemmer("english",)

whitelist = set(list(wn.words()) + words.words())
english_words = ""

#with open(r'./Data/EnglishWords.txt') as word_file:
#    english_words = set(word.strip().lower() for word in word_file)  

def stem_prefix(word, prefixes, roots):
    original_word = word
    for prefix in sorted(prefixes, key=len, reverse=True):
        # Use subn to track the no. of substitution made.
        # Allow dash in between prefix and root. 
        word, nsub = re.subn(r"{}[\-]?".format(prefix), "", word)
        if nsub > 0 and word in roots:
            return word
    return original_word

def porter_english_plus(word, prefixes=english_prefixes):
    return porter.stem(stem_prefix(word, prefixes, whitelist))

def englishWords():
    with open(r'./Data/') as word_file:
        return set(word.strip().lower() for word in word_file)  


def is_english_word(word, english_words):
    return word.lower() in english_words


def removePref(word):
    #prefs = ['a','ab','abs','ac','acanth','acantho','acous','acr','acro','ad','aden','adeno','adren','adreno','aer','aero','af','ag','al','all','allo','alti','alto','am','amb','ambi','amphi','amyl','amylo','an','ana','andr','andro','anem','anemo','ant','ante','anth','anthrop','anthropo','anti','ap','api','apo','aqua','aqui','arbor','arbori','arch','archae','archaeo','arche','archeo','archi','arteri','arterio','arthr','arthro','as','aster','astr','astro','at','atmo','audio','auto','avi','az','azo','bacci','bacteri','bacterio','bar','baro','bath','batho','bathy','be','bi','biblio','bio','bis','blephar','blepharo','bracchio','brachy','brevi','bronch','bronchi','bronchio','broncho','caco','calci','cardio','carpo','cat','cata','cath','cato','cen','ceno','centi','cephal','cephalo','cerebro','cervic','cervici','cervico','chiro','chlor','chloro','chol','chole','cholo','chondr','chondri','chondro','choreo','choro','chrom','chromato','chromo','chron','chrono','chrys','chryso','circu','circum','cirr','cirri','cirro','cis','cleisto','co','cog','col','com','con','contra','cor','cosmo','counter','cranio','cruci','cry','cryo','crypt','crypto','cupro','cyst','cysti','cysto','cyt','cyto','dactyl','dactylo','de','dec','deca','deci','dek','deka','demi','dent','denti','dento','dentro','derm','dermadermo','deut','deutero','deuto','dextr','dextro','di','dia','dif','digit','digiti','dipl','diplo','dis','dodec','dodeca','dors','dorsi','dorso','dyna','dynamo','dys','e','ec','echin','echino','ect','ecto','ef','el','em','en','encephal','encephalo','end','endo','ennea','ent','enter','entero','ento','entomo','eo','ep','epi','equi','erg','ergo','erythr','erythro','ethno','eu','ex','exo','extra','febri','ferri','ferro','fibr','fibro','fissi','fluvio','for','fore','gain','galact','galacto','gam','gamo','gastr','gastri','gastro','ge','gem','gemmi','geo','geront','geronto','gloss','glosso','gluc','gluco','glyc','glyph','glypto','gon','gono','grapho','gymn','gymno','gynaec','gynaeco','gynec','gyneco','haem','haemato','haemo','hagi','hagio','hal','halo','hapl','haplo','hect','hecto','heli','helic','helico','helio','hem','hema','hemi','hemo','hepat','hepato','hept','hepta','heter','hetero','hex','hexa','hist','histo','hodo','hol','holo','hom','homeo','homo','hydr','hydro','hyet','hyeto','hygr','hygro','hyl','hylo','hymeno','hyp','hyper','hypn','hypno','hypo','hypso','hyster','hystero','iatro','ichthy','ichthyo','ig','igni','il','ile','ileo','ilio','im','in','infra','inter','intra','intro','ir','is','iso','juxta','kerat','kerato','kinesi','kineto','labio','lact','lacti','lacto','laryng','laryngo','lepto','leucleuco','leuk','leuko','lign','ligni','ligno','litho','log','logo','luni','lyo','lysi','macr','macro','magni','mal','malac','malaco','male','meg','mega','megalo','melan','melano','mero','mes','meso','met','meta','metr','metro','micr','micro','mid','mini','mis','miso','mon','mono','morph','morpho','mult','multi','my','myc','myco','myel','myelo','myo','n','naso','nati','ne','necr','necro','neo','nepho','nephr','nephro','neur','neuro','nocti','non','noso','not','noto','nycto','o','ob','oc','oct','octa','octo','ocul','oculo','odont','odonto','of','oleo','olig','oligo','ombro','omni','oneiro','ont','onto','oo','op','ophthalm','ophthalmo','ornith','ornitho','oro','orth','ortho','ossi','oste','osteo','oto','out','ov','over','ovi','ovo','oxy','pachy','palae','palaeo','pale','paleo','pan','panto','par','para','pari','path','patho','ped','pedo','pel','pent','penta','pente','per','peri','petr','petri','petro','phago','phleb','phlebo','phon','phono','phot','photo','phren','phreno','phyll','phyllo','phylo','picr','picro','piezo','pisci','plan','plano','pleur','pleuro','pluto','pluvio','pneum','pneumat','pneumato','pneumo','poly','por','post','prae','pre','preter','prim','primi','pro','pros','prot','proto','pseud','pseudo','psycho','ptero','pulmo','pur','pyo','pyr','pyro','quadr','quadri','quadru','quinque','re','recti','reni','reno','retro','rheo','rhin','rhino','rhiz','rhizo','sacchar','sacchari','sacchro','sacr','sacro','sangui','sapr','sapro','sarc','sarco','scelero','schisto','schizo','se','seba','sebo','selen','seleno','semi','septi','sero','sex','sexi','shiz','sider','sidero','sine','somat','somato','somn','sperm','sperma','spermat','spermato','spermi','spermo','spiro','stato','stauro','stell','sten','steno','stere','stereo','stom','stomo','styl','styli','stylo','sub','subter','suc','suf','sug','sum','sup','super','supra','sur','sus','sy','syl','sym','syn','tachy','taut','tauto','tel','tele','teleo','telo','terra','the','theo','therm','thermo','thromb','thrombo','topo','tox','toxi','toxo','tra','trache','tracheo','trans','tri','tris','ultra','un','undec','under','uni','up','uter','utero','vari','vario','vas','vaso','ventr','ventro','vice','with','xen','xeno','zo','zoo','zyg','zygo','zym','zymo']
    #english_words = englishWords()
    for pre in english_prefixes:
        if word.startswith(pre):
            withoutPref = word[len(pre):]
            if is_english_word(withoutPref,english_words):
                return(withoutPref)
    return word  

def get_prefix(word, root):
    if(len(word) <= len(root)): return ""
    diff = len(word) - len(root)
    for i in reversed(range(len(word) - 1)):
        try:
            if i < diff or word[i] != root[i-diff]:
                return word[:i+1]
        except:
            print(word,root)
    return ""

def get_suffix(word, root):
    if(len(word) <= len(root)): return ""
    for i in range(len(word)):
        try:
            if i >= len(root) or word[i] != root[i]:
                return word[i:]
        except:
            print(word,root)
    return ""

# set option to 0 to running tokenizer on line
# set option to 1 (default) to running tokenizer per word 
def tokenizer(line, token_words, option = 1):
    if option == 1:
        words = re.findall(r'\d+\([.,]\d\)*|\w+\.\w+|\w+|\S',line)
        for word in words:
            word = word.lower()
            if word in token_words:
                continue
            #print(word)
            temp = removePref(word)
            prefix = get_prefix(word, temp)
            root = nlp(temp)[0].lemma_
            suffix = get_suffix(temp ,root)
            token_words[word] = (prefix,root,suffix)
    else:
        token_line = " ".join([token.lemma_ for token in nlp(line)])
        if len(line.split()) == len(token_line.split()):
            print("line:",line)
            print("token line:",token_line)
            
def load_tokens(file_name, with_count = 0):
    token_words = {}
    with open(file_name, encoding='utf-8') as file_tokens:
        file_tokens.readline() # to removing the header
        lines = csv.reader(file_tokens)
        for line in lines:
            try:
                if with_count:
                    token_words[line[0]]=(line[1],line[2],line[3],line[4])
                else:
                    token_words[line[0]]=(line[1],line[2],line[3])
            except:
                print(f"have an issue with {line}")
    return token_words

def save_tokens(file_name, token_words , with_count = 0):
    with open(file_name, "w", encoding='utf-8') as token_file:
        if with_count:
            print("word", "prefix", "root", "suffix", "count", file=token_file, sep=',')
        else:
            print("word", "prefix", "root", "suffix", file=token_file, sep=',')
        for word in token_words:
            try:
                if with_count:
                    print('"' + str(word), token_words[word][0], token_words[word][1],  token_words[word][2], str(token_words[word][3]) + '"', file=token_file , sep='","')
                else:
                    print('"' + str(word), token_words[word][0], token_words[word][1],  token_words[word][2] + '"', file=token_file , sep='","')
            except:
                print(f"{word} is not wrote to tokens.")

def split_to_tokens(word):
    temp = removePref(word)
    prefix = get_prefix(word, temp)
    root = nlp(temp)[0].lemma_
    suffix = get_suffix(temp ,root)
    return (prefix,root,suffix)

def get_token_as_word(tokens):
    words = " ".join(token for token in tokens)
    return words.rstrip()