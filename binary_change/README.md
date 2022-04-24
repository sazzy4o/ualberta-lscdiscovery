# Binary change detection

We approach binary change detection using Word Sense Disambiguation (WSD).
The WSD system used is AMUsE by Orlando et al. 2021.

To obtain WSD results for target words run
```
python wsd.py <modern_filename.txt> <old_filename.txt> <target_filename.txt> <old_wsd_folder> <new_wsd_folder> <old_wsd.jsonl> <new_wsd.jsonl>
```

To determine binary change detection labels for each target word run
```
python change_labels.py
```
