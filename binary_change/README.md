# Binary change detection

We approach binary change detection using Word Sense Disambiguation (WSD).
The WSD system used is AMUsE by Orlando et al. 2021.

To obtain WSD results for target words run
```
python wsd.py <modern_filename.txt> <old_filename.txt> <target_words_filename.txt> <output_old_wsd_folder> <output_new_wsd_folder> <output_old_wsd.jsonl> <output_new_wsd.jsonl>
```

To determine binary change detection labels for each target word run
```
python change_labels.py
```

# References
To cite AMuSE WSD
```
@inproceedings{orlando-etal-2021-amuse,
    title = "{AMuSE-WSD}: {A}n All-in-one Multilingual System for Easy {W}ord {S}ense {D}isambiguation",
    author = "Orlando, Riccardo  and
      Conia, Simone  and
      Brignone, Fabrizio  and
      Cecconi, Francesco  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-demo.34",
    doi = "10.18653/v1/2021.emnlp-demo.34",
    pages = "298--307",
    abstract = "Over the past few years, Word Sense Disambiguation (WSD) has received renewed interest: recently proposed systems have shown the remarkable effectiveness of deep learning techniques in this task, especially when aided by modern pretrained language models. Unfortunately, such systems are still not available as ready-to-use end-to-end packages, making it difficult for researchers to take advantage of their performance. The only alternative for a user interested in applying WSD to downstream tasks is to rely on currently available end-to-end WSD systems, which, however, still rely on graph-based heuristics or non-neural machine learning algorithms. In this paper, we fill this gap and propose AMuSE-WSD, the first end-to-end system to offer high-quality sense information in 40 languages through a state-of-the-art neural model for WSD. We hope that AMuSE-WSD will provide a stepping stone for the integration of meaning into real-world applications and encourage further studies in lexical semantics. AMuSE-WSD is available online at http://nlp.uniroma1.it/amuse-wsd.",
}
```
