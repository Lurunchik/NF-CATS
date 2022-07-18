# Non Factoid Question Category classification


This repository contains code for the following paper:

>["A Non-Factoid Question-Answering Taxonomy" published at SIGIR '22, won "Best Paper" Award](https://dl.acm.org/doi/pdf/10.1145/3477495.3531926)  
> Valeriia Bolotova, Vladislav Blinov, W. Falk Scholer, Bruce Croft, Mark Sanderson
> ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2022


## NF_CATS Dataset
The dataset for training is located in [nfcats/data](nfcats/data/)

## Model 
The trained could be downloaded from [the hugginface repository](https://huggingface.co/Lurunchik/nf-cats) and you test the model via [hugginface space](https://huggingface.co/spaces/Lurunchik/nf-cats)

[![demo.png](demo.png)](https://huggingface.co/spaces/Lurunchik/nf-cats)

## Installation

From source:

    cd NF-CATS
    pip install poetry>=1.0.5
    poetry install


## Usage
Run test validation of best fine-tuned model:

    python nfcats/predict.py

Train transformer model

    python nfcats/train.py

Tf-idf experiments:
 
    python tf_idf.py 

## Citation

If you use `NFQA-cats` in your work, please cite [this paper](https://dl.acm.org/doi/10.1145/3477495.3531926)

```
@misc{bolotova2022nfcats,
        author = {Bolotova, Valeriia and Blinov, Vladislav and Scholer, Falk and Croft, W. Bruce and Sanderson, Mark},
        title = {A Non-Factoid Question-Answering Taxonomy},
        year = {2022},
        isbn = {9781450387323},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3477495.3531926},
        doi = {10.1145/3477495.3531926},
        booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
        pages = {1196–1207},
        numpages = {12},
        keywords = {question taxonomy, non-factoid question-answering, editorial study, dataset analysis},
        location = {Madrid, Spain},
        series = {SIGIR '22}
}
```


