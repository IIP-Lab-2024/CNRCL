# CNRCL

This repository contains the code for "A Contrastive News Recommendation Framework based on Curriculum Learning".

## Introduction

In this paper, "A Contrastive News Recommendation Framework based on Curriculum Learning", we design a novel approach to link curriculum learning with negative sample screening and use contrastive learning to improve the model's interest modeling capabilities.

## Requirement

- python~=3.8

- torch==1.12.1

- torchtext==0.13.1

- torch-scatter==2.0.9

- nltk==3.7 

- scikit-learn==1.1.3 

- pandas==1.5.3

- numpy==1.23.4 



## Dataset Preparation

The experiments are conducted on the MIND dataset. Our code will try to download and sample the MIND dataset to the directory `../MIND`.

Assume that now the pwd is `./CNRCL`, the downloaded and extracted MIND dataset should be organized as

    (terminal) $ bash download_extract_MIND.sh # Assume this command is executed successfully
    (terminal) $ cd ../MIND
    (terminal) $ tree -L 2
    (terminal) $ .
                 ├── dev
                 │   ├── behaviors.tsv
                 │   ├── entity_embedding.vec
                 │   ├── news.tsv
                 │   ├── __placeholder__
                 │   └── relation_embedding.vec
                 ├── dev.zip
                 ├── train
                 │   ├── behaviors.tsv
                 │   ├── entity_embedding.vec
                 │   ├── news.tsv
                 │   ├── __placeholder__
                 │   └── relation_embedding.vec
                 ├── train.zip
    
<br/>

Then run prepare_MIND-dataset.py to preprocess the data.

<br/>

##  Get two important files

1. Run "generate_similarities.py" to get the data file for comparative learning
2. Run "generate_similarity_result_history" to get the data file for curriculum learning

## Run

<pre><code>python main.py --news_encoder=CNRCL --user_encoder=CNRCL</code></pre>


### Credits
- Reference https://github.com/Veason-silverbullet/NNR
