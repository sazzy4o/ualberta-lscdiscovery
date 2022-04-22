from nltk.tokenize import sent_tokenize
import string
import time
import urllib
import json
import gzip
import requests

class WSD:
    def __init__(self, old_filename, modern_filename, target_filename):
        self.old_filename = old_filename
        self.modern_filename = modern_filename
        self.target_filename = target_filename

        self.targetWords = self.readTargets(self.target_filename)

        self.old_word_tag_contexts = {}
        self.new_word_tag_contexts = {}

        self.splitSents(self.old_filename, self.old_word_tag_contexts)
        self.splitSents(self.modern_filename, self.new_word_tag_contexts)

        self.word_sent_results_old = {}
        self.word_sent_results_new = {}

        self.disambiguateAmuse(self.old_word_tag_contexts, self.word_sent_results_old, 'old_wsd/', 'old_disambig_full_v3_eval.jsonl')
        self.disambiguateAmuse(self.new_word_tag_contexts, self.word_sent_results_new, 'new_wsd/', 'new_disambig_full_v3_eval.jsonl')

        return
    
    def readTargets(self, f):
        target_words = []
        words = open(f, 'r').readlines()
        for word in words:
            target_words.append(word.strip())
        return target_words

    def splitSents(self, f, d):
        lines = open(f, 'r').readlines()
        for line in lines:
            prts = line.strip().split()
            words = []
            poss = [] 
            for prt in prts:
                word_pos = prt.split("_")
                word = word_pos[0]
                if len(word_pos) == 1:
                    continue
                pos = word_pos[1]
                words.append(word)
                poss.append(pos)
            # sents = " ".join(words)
            # token_text = sent_tokenize(sents, language='spanish')
            s = " ".join(words)
            for w in words:
                target_word = self.checkTargetWords(w)
                if target_word != None:
                    if target_word not in d.keys():
                        d[target_word] = []
                    else:
                        pos_tag = poss[words.index(target_word)]
                        d[target_word].append([pos_tag, s])
        return

    def checkTargetWords(self, w):
        for tgt in self.targetWords:
            if tgt == w:
                return tgt
        return None
    
    def disambiguateAmuse(self, d, results, folder, filename):
        url = "http://nlp.uniroma1.it/amuse-wsd/api/model"
        headers = {'accept': 'application/json', 'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        all_results = []
        used_sents = []
        for target_word in list(d.keys()):
            for [pos, context] in d[target_word]:
                if context in used_sents:
                    continue
                data = []
                data.append({"text":context, "lang":"ES"})
                used_sents.append(context)
                result = requests.post(url, data=json.dumps(data), headers=headers)
                if not result.ok:
                    print(target_word, context)
                    continue
                result = result.json()
                all_results.append(result)
            # while not result.ok:
            #     result = requests.post(url, data=json.dumps(data), headers=headers)
            #     print(target_word, result)            
            # result = result.json()
        text = "\n".join([json.dumps(json_) for json_ in all_results])    
        with open(folder + filename, 'w') as f:
            f.write(text)
            #json.dump(result, f)
            
        return

if __name__ == "__main__":
    root = "/home/dteodore/starting_kit/"
    modern_filename = root + "corpora/modern_corpus/public_www/public_www/modern_corpus_pos.txt"
    old_filename = root + "corpora/old_corpus/dataset_XIX_pos.txt"
    target_filename = root + "target_words_part.txt"
    #target_filename = root + "target_words_evaluation_phase2.txt"
    WSD(old_filename, modern_filename, target_filename)
