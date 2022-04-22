import pandas as pd
import random
import spacy
spacy.require_gpu()

from pathlib import Path
from tqdm import tqdm

out_folder = Path('models/t5_embedding/')
out_folder.mkdir(exist_ok=True,parents=True)

nlp = spacy.load("es_dep_news_trf")

from pathlib import Path

root = Path(__file__).parent.resolve()

random.seed(42)

def load_dataset_df(path):
    rows = []
    with open(path) as dataset_file:
        for line in tqdm(dataset_file.readlines()):
            doc = nlp(line)
            content_words = [(i,x) for i,x in enumerate(doc) if x.pos_ in ['VERB','NOUN','ADV','ADJ']]
            if len(content_words) == 0:
                continue
            content_index,content_word = random.choice(content_words)
            rows.append({
                'input_text': ' '.join([x.text if i!=content_index else '<extra_id_0>' for i,x in enumerate(doc)]),
                'target_text': content_word.text,
            })
    # target_text     input_text      prefix
    df = pd.DataFrame(rows)
    # df['prefix'] = 'predict word' 
    return df

def count_words(text):
    return len(text.split(' '))

df = pd.concat([
    load_dataset_df(root/'datasets/es-lscdiscovery/c1.txt'), 
    load_dataset_df(root/'datasets/es-lscdiscovery/c2.txt')
], ignore_index=True, sort=False)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('t5_training_data.csv',index=False)