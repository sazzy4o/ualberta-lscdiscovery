#%%
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# %%
#%% Based on https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c
import pandas as pd
import random

from pathlib import Path
from simpletransformers.t5 import T5Model, T5Args

out_folder = Path('models/t5_embedding/')
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
model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 400,
    "train_batch_size": 2,
    "num_train_epochs": 1,
    "save_steps": -1,
    "use_multiprocessing": False,
    # "evaluate_during_training": True,
    # "evaluate_during_training_steps": 15000,
    # "evaluate_during_training_verbose": True,
    "local_files_only":True,
    "fp16": False,
    "output_dir": str(out_folder.resolve()),

    "manual_seed": 314,

    "wandb_project": "Spanish Shared Task T5 Training",
}

model = T5Model("mt5", "google/mt5-large", args=model_args)

model.train_model(df)