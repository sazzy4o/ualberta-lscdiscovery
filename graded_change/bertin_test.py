#%%
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bertin-project/bertin-roberta-base-spanish")

model = AutoModelForMaskedLM.from_pretrained("bertin-project/bertin-roberta-base-spanish")
# %%
inputs = tokenizer('Fui a la librería a comprar un <mask>.', return_tensors="pt")
# %%
outputs = model(**inputs,output_hidden_states=True)
# %%
mask_index = (inputs['input_ids']==4).nonzero()[0][-1]
outputs.hidden_states[-1][0][mask_index]
# %%
