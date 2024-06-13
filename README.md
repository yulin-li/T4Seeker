# T4Seeker
This is for identifing type IV secretion effectors. This repository contains a deep learning model and prediction code for T4Seeker. Users can utilize T4Seeker to predict the amino acid sequence.

![flowchart T4Seeker.](https://github.com/lijingtju/T4Seeker/blob/main/flowchat.png)

### Manual Start
```
git clone https://github.com/lijingtju/T4Seeker.git
cd /path/to/T4Seeker
pip install fair-esm
mkdir features
mkdir features_code
cd features_code
git clone https://github.com/banshanren/Pse-in-One-2.0.git
cd ..
```

### Commands to do prediction
####Step 1:
```
python extract_features.py —fastafile ./data/test.fasta
```
####Step 2:
```
python T4SeekerPredict.py --fasta_test test.csv --DR_test ./data/test_data.csv --ESM_test ./data/ESM_features_test_data.csv
```
