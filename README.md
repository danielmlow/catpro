# CATPro: Clinical Audio Text Processing

![CATPro](https://github.com/danielmlow/catpro/blob/main/catpro/docs/catpro_logo.jpg?raw=true=50x)
          

Tools 


`$ conda activate catpro`



- qualitative 
    - inter-rater agreement
    - annotation


- tools
    - dimensionality reduction
    - rename_files

- Audio
    - tutorial: ipynb that calls the different functions. 
    - preprocessing
        - downsample
        
    - features
        - different embeddings
        - egemaps
        - CPP
    - quality control
        - dimensionality reduction across samples of the same speaker
    - individual_dashboards

- Text 
    - tutorial: ipynb that calls the different functions. 
    - `utils`
        - lemmatizer
        - stemmer
        - count_words
    - `embeddings.py` returns embeddings for a doc, specify model from huggingface
    - `embeddings_sense_disambiguation.py` looks for embeddings as used in context in a dataset.
        - specify: dataset, window length, N windows, model, layer, average_method 
    
    - `preprocess_lexicons.ipynb` # for non concept tracker categories
    - `features.py`
        - lexicons: SEANCE, LGBTQ, ethnicities
        - emotions: valence and arousal + new model with 27 emotiones
        - lexical diversity
        - concreteness 
        - incoherence
        - disorganization
        - readability
        - POS
        - colloquial
    
    - `network co-occurence maps` alternative to word clouds
        - optional: supervised, which does within each level of y label values
        - remove vague words that attract many others
        - default: up to eg 250 words can fit 
- Plot
    - split_strings



- Data `catpro/data/`
    - `lexicons/`
    - `rmhd`
    - `new_york_times`
    - `jam_study`



# TODOs
[] install cookiecutter: https://cookiecutter-hypermodern-python.readthedocs.io/en/2020.6.15/guide.html or simpler: https://cookiecutter-poetry.readthedocs.io/en/latest/tutorial.html


