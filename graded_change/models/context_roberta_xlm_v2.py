""" Model for unsupervised lexical semantic change ranking based on context-dependent word representations. """

from models.utils.io_utils import make_masked_copy, load_dataset, load_pretrained_bert, load_local_bert, collect_sentences, load_rep_dict
from models.utils.general_utils import find_first_seq_ics, apply2dicts, dict2array

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from scipy.spatial.distance import cdist
from tqdm import tqdm
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

import pandas as pd
import numpy as np
import torch
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM ,AutoTokenizer

def make_classification_dataset(dataset_dir, experiment_dir):
    """ Creates a balanced time classification dataset from a diachronic LSCD dataset. """

    prep_dir = experiment_dir + "preprocessed_texts/"
    os.makedirs(prep_dir, exist_ok=True)

    with open(dataset_dir + "c1.txt", "r") as fh:
        sents_c1 = fh.read().splitlines()

    with open(dataset_dir + "c2.txt", "r") as fh:
        sents_c2 = fh.read().splitlines()

    # detemine thresholds
    n_samples_per_class = (min(len(sents_c1), len(sents_c2)) // 10_000) * 10_000
    n_train = int(n_samples_per_class * 2 * 0.8)
    n_test = n_samples_per_class * 2 - n_train

    # collect samples for each class
    df_0 = pd.DataFrame({"text": sents_c1, "label": 0}).sample(n=n_samples_per_class, replace=False)
    df_1 = pd.DataFrame({"text": sents_c2, "label": 1}).sample(n=n_samples_per_class, replace=False)

    # sample train and test data for each label without overlap
    perm = np.random.permutation(n_samples_per_class)
    train_df = pd.concat([df_0.iloc[perm[:(n_train // 2)]], df_1.iloc[perm[:(n_train // 2)]]], ignore_index=True)
    test_df = pd.concat([df_0.iloc[perm[(n_train // 2):]], df_1.iloc[perm[(n_train // 2):]]], ignore_index=True)

    # check balance of labels
    assert np.all(train_df.label.value_counts() / n_train == test_df.label.value_counts() / n_test), "Classification dataset is not balanced!"

    # shuffle train and test data
    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)

    for df in [train_df, test_df]:
        
        # remove year that is at beginning of sentence in some datasets
        df["text"] = df["text"].str.rsplit("\t", expand=True)[0]

        # create dummy columns to conform with BERT dataset format
        df["alpha"] = ["a"] * len(df.index)
        df["id"] = range(len(df.index))

    train_df[["id", "label", "alpha", "text"]].to_csv(prep_dir + "train.tsv", sep="\t", index=False, header=False)
    test_df[["id", "label", "alpha", "text"]].to_csv(prep_dir + "test.tsv", sep="\t", index=False, header=False)

    make_masked_copy(prep_dir + "train.tsv")
    make_masked_copy(prep_dir + "test.tsv")

# Base on another project (private repo)
# Which was based on https://colab.research.google.com/drive/1RyUsYDAo6bA1RZICMb-FxYLszBcDY81X?usp=sharing#scrollTo=0U2cqAsrzp7E
def finetune_roberta_xlm_v2(experiment_dir, device="cpu", bert_name="bert-base-multilingual-cased", masked=True, **params):
    """ Finetunes a pretrained BERT model on a sentence time classification objective. """

    model_dir = experiment_dir + "model/"
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device(device)

    print("Loading pretrained roberta-xlm model ...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

    train_fp = experiment_dir + "preprocessed_texts/train.tsv"
        
    # train bert model
    df = pd.read_csv(train_fp, sep="\t", header=None, names=["id", "label", "alpha", "text"])
    df['text']= df['text'].astype(str)
    train_dataset = Dataset.from_pandas(df.sample(frac=1, random_state=42).reset_index(drop=True))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.1)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["id", "label", "alpha", "text"])

    block_size = 512

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    training_args = TrainingArguments(
        'training_out',
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        # no_cuda=True,
        save_strategy = "no", # If this is enabled "Disk quota exceeded" can happen fast
        num_train_epochs=1
    )

    lm_train_dataset = train_dataset_tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=lm_train_dataset, 
        # eval_dataset=lm_test_dataset,
        data_collator=data_collator
    )

    trainer.train()

    # save bert model
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # # reload bert model and test dataset
    # tokenizer, model = load_local_bert(bert_dir, device)
    # test_dataset = load_dataset(tokenizer, bert_name, experiment_dir + "preprocessed_texts/train.tsv")

    # # evaluate bert model and save results
    # acc = test_bert(test_dataset, model, tokenizer, device, **params)
    # np.save(bert_dir + "classification_accuracy.npy", np.round(acc, decimals=2))

def test_bert(test_dataset, model, tokenizer, device, batch_size=10, **kwargs):
    """ Evaluates a BERT model on a test dataset. """

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    preds = None
    model.eval()

    for batch in tqdm(test_dataloader, desc="BERT Testing"):

        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():

            inputs = {"input_ids":      batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": batch[2],
                      "labels":         batch[3]}

            logits = model(**inputs)[1]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    accuracy = (out_label_ids == preds).sum() / preds.shape[0]

    return accuracy


def extract_representations_roberta_xlm_v2(dataset_dir, experiment_dir, device="cpu", **kwargs):
    """ Extracts last hidden layer values for all target words in a datasets."""

    model_dir = experiment_dir + "model/"

    rep_dir_c1 = experiment_dir + "word_representations/c1/"
    rep_dir_c2 = experiment_dir + "word_representations/c2/"

    os.makedirs(rep_dir_c1, exist_ok=True)
    os.makedirs(rep_dir_c2, exist_ok=True)

    with open(dataset_dir + "targets.tsv", "r") as fh:
        targets = fh.read().splitlines()

    sents_c1 = collect_sentences(targets, dataset_dir + "c1.txt")
    sents_c2 = collect_sentences(targets, dataset_dir + "c2.txt")

    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = AutoModelForMaskedLM.from_pretrained(model_dir).eval().to(device)

    save_representations(sents_c1, model, tokenizer, device, rep_dir_c1)
    save_representations(sents_c2, model, tokenizer, device, rep_dir_c2)


def save_representations(word_sents, model, tokenizer, device, rep_dir, max_sql=128):
    """ Saves representations extracted from BERT for sentences in a given folder. """

    for word, sents in word_sents.items():

        word_hidden_states = []
        word_tokens = tokenizer.encode(word)

        for sent in tqdm(sents, desc="Extracting Representations for '{}'".format(word)):
            try:
                word_hidden_states.append(get_hidden_from_sent(sent, model, tokenizer, device, max_sql, word))
            except IndexError: pass
            # hidden_states, encoded = get_hidden_from_sent(sent, model, tokenizer, device, max_sql)

            # if len(word_tokens) > 0:
            #     word_token_ics = find_first_seq_ics(encoded, word_tokens)
            # else:
            #     word_token_ics = [np.argmax(encoded == word_tokens[0])]

            # if len(word_token_ics) > 0:
            #     mean_state_last_layer = hidden_states[-1, word_token_ics, :].mean(axis=0)
            #     word_hidden_states.append(mean_state_last_layer)

        np.save(rep_dir + word + ".npy", np.array(word_hidden_states))

# ! TODO: This could  be a lot faster if this was batched...
def get_hidden_from_sent(sent, model, tokenizer, device, max_sql,word):
    """ Returns BERT last layer activations with shape (layers, tokens, hidden_size) as well as the encoded sentence. """

    with torch.no_grad():
        # expected = "<extra_id_0>"
        inputs = tokenizer(sent.replace(word,'<mask>'),return_tensors="pt", max_length = 512, truncation = True).to(device)
        outputs = model(**inputs,output_hidden_states=True)

        mask_index = (inputs['input_ids']==tokenizer.mask_token_id).nonzero()[0][-1]
        # outputs.hidden_states[-1][0][mask_index]

        return outputs.hidden_states[-1][0][mask_index].cpu().numpy()
        # return outputs.logits[0][mask_index].cpu().numpy()


def compare_context_dependent_representations(dataset_dir, experiment_dir):
    """ Compares extracted representations for all target words and makes a prediction. """

    with open(dataset_dir + "targets.tsv", "r") as fh:
        targets = fh.read().splitlines()
  
    c1_reps = load_rep_dict(experiment_dir + "word_representations/c1/", targets)
    c2_reps = load_rep_dict(experiment_dir + "word_representations/c2/", targets)

    dist_func = lambda x, y: np.mean(cdist(x, y, metric="euclidean"))
    dist_dict = apply2dicts(c1_reps, c2_reps, dist_func)
    dists = dict2array(dist_dict, targets)
    
    pd.DataFrame({"word": targets, "change": dists}).to_csv(experiment_dir + "prediction.tsv", sep="\t", index=False, header=False)
    
