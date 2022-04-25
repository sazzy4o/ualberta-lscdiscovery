import json
import argparse
from sklearn.metrics import precision_recall_fscore_support
class LSC:
    def __init__(self, root_dir, target_filename, old_wsd_folder, new_wsd_folder, old_wsd_jsonl, new_wsd_jsonl, gold_filename, threshold):
        self.targetWords = self.readTargets(target_filename)
        self.old_word_senses = {}
        self.new_word_senses = {}

        self.readWordSenses(root_dir, old_wsd_folder, old_wsd_jsonl, self.old_word_senses)
        self.readWordSenses(root_dir, new_wsd_folder, new_wsd_jsonl, self.new_word_senses)

        self.readGoldScores(gold_filename)
        self.getF1ScoreRelativeChangeDiff(threshold)
        return

    def readTargets(self, f):
        target_words = []
        words = open(f, 'r').readlines()
        for word in words:
            target_words.append(word.strip())
        return target_words

    def checkTargetWords(self, w, lemma):
        for tgt in self.targetWords:
            if tgt == w or tgt == lemma:
                return tgt
        return None

    def readWordSenses(self, root_dir, folder, jsonl, d):
        filename = root_dir + folder + jsonl
        with open(filename, 'r') as f:
            data_lst = list(f)
        for data in data_lst:
            data = json.loads(data)
            for sent_dict in data:
                lst_results = sent_dict["tokens"]
                for token_dict in lst_results:
                    wrd = token_dict["text"]
                    lemma = token_dict["lemma"]
                    w = self.checkTargetWords(wrd, lemma)
                    if w != None:
                        id = token_dict["bnSynsetId"]
                        if w not in d.keys():
                            d[w] = {}
                            d[w][id] = 1
                        else:
                            if id not in d[w].keys():
                                d[w][id] = 1
                            else:
                                d[w][id] += 1
                        continue
        return
    
    def findTotal(self, d):
        total = 0
        for k,v in d.items():
            total += v
        return total

    def readGoldScores(self, gold_filename):
        lines = open(gold_filename, 'r').readlines()
        y_true = []
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            y_true.append(int(line))
        self.y_true = y_true
        return 
    
    def getF1ScoreRelativeChangeDiff(self, threshold):
        y_pred = []
        f = open("submission.tsv", "w")
        f.write('word' + "\t" + 'change_binary' + "\t" + "change_binary_gain" + "\t" + "change_binary_loss" + "\t" + "change_graded" + "\t" + "COMPARE" + "\n")
        for target_word in self.targetWords:
            f.write(target_word + "\t")
            found_score = False 
            if target_word in self.old_word_senses.keys() and target_word in self.new_word_senses.keys():
                old = self.old_word_senses[target_word]
                new = self.new_word_senses[target_word]
                old_total = self.findTotal(old)
                new_total = self.findTotal(new)

                all_senses = set(list(old.keys()) + list(new.keys()))
                for id in all_senses:
                    if id in old.keys() and id in new.keys(): # in both old and new
                        old_freq = old[id]
                        new_freq = new[id]
                        old_prob = old_freq/old_total
                        new_prob = new_freq/new_total
                        if abs(old_prob - new_prob)/max(old_prob, new_prob) > threshold:
                            y_pred.append(1)
                            found_score = True
                            break
                    else: # Sense is missing in a corpus --> change for word
                        y_pred.append(1)
                        found_score = True
                        break
                if found_score == False: # wasn't big enough change for any 1 sense
                    y_pred.append(0)
                f.write(str(y_pred[-1]) + "\t" + str(0) + "\t" + str(0) + "\t" + str(0.5) + "\t" + str(0.5) + "\n")
            else:
                print("Not in one files", target_word)
        p, r, f1, _ = precision_recall_fscore_support(self.y_true, y_pred, average='binary')
        print("Precision", p, "Recall", r, "F-score:", f1)
        f.close()
        return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help='Working directory')
    parser.add_argument('target_filename', type=str, help='Target word txt')
    parser.add_argument('old_wsd_folder', type=str, help='Foldername for old wsd results')
    parser.add_argument('new_wsd_folder', type=str, help='Foldername for new wsd results')
    parser.add_argument('old_wsd_jsonl', type=str, help='Jsonl name for old wsd results')
    parser.add_argument('new_wsd_jsonl', type=str, help='Jsonl name for new wsd results')
    parser.add_argument('gold_filename', type=str, help='Gold binary change detection scores txt')
    parser.add_argument('threshold', type=str, help='Relative change probability threshold')

    args = parser.parse_args()
    root_dir = args.root_dir
    target_filename = args.target_filename
    old_wsd_folder = args.old_wsd_folder
    new_wsd_folder = args.new_wsd_folder
    old_wsd_jsonl = args.old_wsd_jsonl
    new_wsd_jsonl = args.new_wsd_jsonl
    gold_filename = args.gold_filename
    threshold = float(args.threshold)
    
    LSC(root_dir, target_filename, old_wsd_folder, new_wsd_folder, old_wsd_jsonl, new_wsd_jsonl, gold_filename, threshold)
