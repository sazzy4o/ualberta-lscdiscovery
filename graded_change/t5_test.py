#%%
import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer

model = MT5ForConditionalGeneration.from_pretrained("models/t5_embedding_small")
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
input_text = "el uno en pos de el otro , el cuerpo inclinado hacia atrás y el ancho pies hincado fuertemente en el <extra_id_0> de el playa , parecer nuevo Hércules dispuesto a combatir con el elemento . "
expected = "<extra_id_0>"
inputs = tokenizer(input_text, return_tensors="pt")
with tokenizer.as_target_tokenizer():
    labels = tokenizer(expected, return_tensors="pt")
outputs = model(**inputs, labels=labels["input_ids"],output_hidden_states=True)
# %%
