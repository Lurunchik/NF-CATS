# Non Factoid Question Category classification

## NF_CATS Dataset
The dataset for training is located in [nfcats/data](nfcats/data/)


## Installation

From source:

    cd NF-CATS
    pip install poetry>=1.0.5
    poetry install


## Model
Run test validation of best fine-tuned model:

    python nfcats/predict.py

Train transformer model

    python nfcats/train.py

Tf-idf experiments:
 
    python tf_idf.py 




