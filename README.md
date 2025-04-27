# T4Seeker
This is for identifying type IV secretion effectors. This repository contains a deep learning model and prediction code for T4Seeker. Users can utilize T4Seeker to predict the amino acid sequence.

![flowchart T4Seeker.](https://github.com/lijingtju/T4Seeker/blob/main/flowchart.png)

### Manual Start
```
git clone https://github.com/yulin-li/T4Seeker.git
cd T4Seeker
pip install "torch<2.6" --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm
mkdir features_code
cd features_code
git clone https://github.com/banshanren/Pse-in-One-2.0.git PseinOne2
cd ..
```

### Commands to do prediction
####Step 1:
```
python extract_features.py --fastafile ./data/test.fasta
python to_csv.py
```
####Step 2:
```
python T4SeekerPredict.py --fasta_test ./data/test.csv --DR_test ./features/test_DR.csv --ESM_test ./features/test_ESM_features.csv
```
