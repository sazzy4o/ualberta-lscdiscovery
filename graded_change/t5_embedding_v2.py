#%%
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# %%
#%% Based on https://colab.research.google.com/drive/1OFiiaBA40EdReaeKB6MHnC7Dvl8u5a2T?usp=sharing#scrollTo=-pMMQ_3t10DT
import pandas as pd
import random

from pathlib import Path
from simpletransformers.t5 import T5Model, T5Args

out_folder = Path('models/t5_embedding_small/')
out_folder.mkdir(exist_ok=True,parents=True)

# doc = nlp("Esto es una frase.")
# print([(w.text, w.pos_) for w in doc])

from pathlib import Path

root = Path(__file__).parent.resolve()

random.seed(42)

df = pd.read_csv(root/'t5_training_data.csv')
df['prefix'] = 'predict word'
df['input_text'] = df['input_text'].astype(str, copy=False)
df['target_text'] = df['target_text'].astype(str, copy=False)
#%%
import torch
from tqdm import tqdm
from transformers import Trainer,TrainingArguments, MT5ForConditionalGeneration, T5Tokenizer
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')

max_length = 256

# TODO: See https://github.com/huggingface/notebooks/blob/master/examples/summarization.ipynb
def train_data_generator(train_df):
    for i,row in tqdm(train_df.iterrows(),total=len(train_df)):
        input_encodings = tokenizer(row['prefix']+row['input_text'], pad_to_max_length = True, max_length = max_length, truncation = True, return_tensors='pt')
        target_encodings = tokenizer(row['target_text'], pad_to_max_length = True, max_length = 16, truncation = True, return_tensors='pt')

        yield {
            **{k:v[0] for k,v in input_encodings.items()},
            'labels': target_encodings['input_ids'][0],
            # 'target_attention_mask': torch.tensor(target_encodings['attention_mask'] + [tokenizer.pad_token_id], dtype = torch.long)
        }

training_args = TrainingArguments(
    output_dir=out_folder,
    overwrite_output_dir=True,
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=64,
    # gradient_checkpointing=True,
    save_steps=100,
    logging_steps=100,
    save_total_limit=2,
    do_train = True,
    # bf16=True,
    optim = 'adafactor'
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = list(train_data_generator(df)),
)
trainer.train()
model.save_pretrained(out_folder)
# %%
