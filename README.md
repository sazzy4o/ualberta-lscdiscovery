# ualberta-lscdiscovery

The UAlberta team at the shared task on Lexical Semantic Change Discovery (LSCDiscovery) in Spanish, collocated with the 3rd International Workshop on Computational Approaches to Historical Language Change 2022 (LChange’22).
Our approach obtains competitive results.

For phase 1 (graded change discovery), we ensemble static and contextual embeddings.
For phase 2 (binary change detection), we frame it as a word sense disambiguation problem.
Code for graded change discovery and binary change detection are in the respective folders.

# Citation

Please cite

```
@inproceedings{teodorescu-etal-2022-black,
    title = "{UA}lberta at {LSCD}iscovery: Lexical Semantic Change Detection via Word Sense Disambiguation",
    author = "Teodorescu, Daniela  and
      von der Ohe, Spencer  and
      Kondrak, Grzegorz",
    booktitle = "Proceedings of the 3rd Workshop on Computational Approaches to Historical Language Change",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.lchange-1.19",
    doi = "10.18653/v1/2022.lchange-1.19",
    pages = "180--186",
    abstract = {We describe our two systems for the shared task on Lexical Semantic Change Discovery in Spanish. For binary change detection, we frame the task as a word sense disambiguation (WSD) problem. We derive sense frequency distributions for target words in both old and modern corpora. We assume that the word semantics have changed if a sense is observed in only one of the two corpora, or the relative change for any sense exceeds a tuned threshold. For graded change discovery, we follow the design of CIRCE (P{\"o}msl and Lyapin, 2020) by combining both static and contextual embeddings. For contextual embeddings, we use XLM-RoBERTa instead of BERT, and train the model to predict a masked token instead of the time period. Our language-independent methods achieve results that are close to the best-performing systems in the shared task.},
}
```
