# NERFinalProject

Simple evaluation script for NER tagger.

Requires Python 3.x to run.

To train the BERT model on Twitter data, run:
python3 train.py -t n -m BERT

To test the BERT model on dev dataset, run:
python3 train.py -t nerModel -m BERT

To train the BERTCRF model on Twitter data, run:
python3 train.py -t n -m BERTCRF

To test the BERTCRF model on dev dataset, run:
python3 train.py -t nerCRFModel -m BERTCRF

Look at train.py for more information.
