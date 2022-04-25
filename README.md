# Non Factoid Question Category classification


This repository contains code for the following paper:

>["A Non-Factoid Question-Answering Taxonomy" published at SIGIR '22]()  
> Valeriia Bolotova, Vladislav Blinov, W. Falk Scholer, Bruce Croft, Mark Sanderson
> ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2022


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




