# CATPRO

# TODOs
- install cookiecutter
- pip install private repo https://stackoverflow.com/questions/4830856/is-it-possible-to-use-pip-to-install-a-package-from-a-private-github-repository


# Concept Tracker 

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
    - `concept_tracker.ipynb` Concept Tracker main script
        
        1. build lexicons: 
            - wordhoard_by_sense.py
                TODO: make modular 
                1. Obtain synonyms
                ```
                lexicon = concept_tracker()
                lexicon.gen_wordhoard_df(words,input_dir_stored_searches=None, return_all_stored_searches=True, search_for='synonyms')
                ```
                2. annotate which senses to remove (but do not remove; keep everything saved)
                3. add tokens (from your imagination, Reddit, questionnaires, etc.)
                - TODO: Find similar words and contexts in relevant corpora (Reddit) 
                    - find_similar_tokens(dataset, token)
                - TODO: add SEANCE 
                - TODO: add prior lexicons. Check if phrase is in Reddit longitudinal SW dataset or in Bursting Study
                4. Save wordhoard+manually added results to json `./data/lexicons/wordhoard_manual_synonyms/synonyms_<timestamp>.json`
                5. Cocatenate tokens from all sources and save to json `./data/lexicons/wordhoard_manual_synonyms/lexicon_ct_<timestamp>.json`
                - TODO: automatically generate different POS from the queried word and search for those (insomnia, insomniac)
                - TODO: save queries for all categories in SEANCE and LIWC
        2. TODO: Remove outliers: automatically (> 90% similarity with regards to category-label embedding, infrequent in target database), judges, tests (ask participants to suggest words to see if it is included)
            - if using judges: inter-rater agreement
        - Simmatizer / synonymatizer: expanding lemmatizer to similar words
        3. Sense disambiguation: look for phrases including words in larges corpora to get different senses 
        4. Counter concept tracker
        *    - lemmatize doc and lexicons
            - word_count 
        5. semantic concept tracker
        *    - tokenize in clauses (VF, complete sentence, split by conjunctions) or ngrams
            - negations: remove clauses with negation or quasi-negations
            - encode into embeddings
                - (optional: sense_disambiguation)
            - compute similarity
        *    - summarize similarity
                - threshold as a function of convex hull
                - function

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

