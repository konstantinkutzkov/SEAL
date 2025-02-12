# SEAL: Scalable Entity ALignment

Below we describe the overall structure of the folder, the code organization as well as instructions for code reproducibility.

## Running the demo:
 Once the required packages are installed, one can start the demo with

 ``
    cd src/
 ``

 ``   
    streamlit run demo.py
 ``

## Overall structure
The pipeline consists of three parts as shown in three different pages of the demo:
1. Learn entity embeddings. This is the most interesting part. Currently, we start with precomputed LLM embeddings for the entities and relations, and then train a DeepWalk model on them. In addition we generate different features such n-grams of entity and relations, structural properties, etc,
2. Train a model for the classification of entity pairs. We use the provided positive pairs (sup_pars) and also generate negative pairs. Given two (embedding) vectors for an entity pair, we train a binary classification model (currently a GBT model but anything can be used). 
3. Prediction/evaluation of results. For a given query entity, generate all possible pairs and predict the probability for each pair, output the top-k pairs.

## Organization
- data: The data was downloaded from https://github.com/kosugi11037/bert-int/tree/master/data/dbp15k
- embeddings: storing the precomputed entity embeddings
- images: images for the demo 
- models: the trained classification models for given hyperparameters
- src: the folder containing the main code
- notebooks: used for testing


## Requiremenents
The file freeze.txt lists all installed packages in the virtual environment but here are the main ones:
- python 3.10 (This is important for tensorflow-text to work)
- tensorboard==2.18.0
- tensorflow==2.18.0
- tensorflow-hub==0.16.1
- tensorflow-text==2.18.0
- scikit-learn==1.5.2
- pandas==2.2.3
- gensim==4.3.3
- lightgbm==4.5.0
- nltk==3.9.1
- numpy==1.26.4
- streamlit==1.41.1

## Code
The main files are as follows:
- demo.py The demo interface. The different functions are called from here.
- entity_embeddings.py The utilities for training embeddings and generating features for the input vectors
- train_model. Creating a dataset for binary classification using negative sampling and then training a LightGBM model with early stopping.
- utils.py Different functions 
- evaluate.py Computing hits@k for the test dataset, not used in the demo
