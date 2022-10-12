

# !pip install -q --upgrade transformers

from transformers import pipeline
import pandas as pd
import numpy as np

def huggingface_output_2_df(output_dict, col_prefix =None):
	feature_vectors = []
	col_names = []
	for doc in output_dict:
		feature_vectors_doc = []
		for feature in doc:
			feature_vectors_doc.append(feature.get('score'))
			col_names.append(feature.get('label'))
		feature_vectors.append(feature_vectors_doc)
	_, idx = np.unique(col_names, return_index=True)
	col_names = np.array(col_names)[np.sort(idx)]
	if col_prefix:
		col_names = [col_prefix+'_'+n for n in col_names]
		feature_vectors = pd.DataFrame(feature_vectors, columns = col_prefix)
	feature_vectors = pd.DataFrame(feature_vectors, columns= col_names)
	return feature_vectors


def huggingface(docs, task='sentiment-analysis',model_name=None, return_all_scores=True, col_prefix = 'distilbert_'):
	'''

	Args:
		docs: list of strings
		task: see list on huggingface, default='sentiment_analysis'
		model_name: {'arpanghoshal/EmoRoBERTa', "bhadresh-savani/distilbert-base-uncased-emotion"}, default=None
		return_all_scores: bool, default=True
		col_prefix: select subset
			Example: ['distilbert_sadness', 'distilbert_joy', 'distilbert_love', 'distilbert_anger', 'distilbert_fear', 'distilbert_surprise']

	Returns:
		if return_all_scores=False, list of dict
		[
			{'label': 'NEGATIVE', 'score': 0.9997896552085876},
	        {'label': 'POSITIVE', 'score': 0.9493240714073181}
        ]

		if return_all_scores=True, list of list of dict
		[
			[   {'label': 'NEGATIVE', 'score': 0.9997896552085876},
                {'label': 'POSITIVE', 'score': 0.00021033342636656016}
            ],

            [   {'label': 'NEGATIVE', 'score': 0.050675973296165466},
                {'label': 'POSITIVE', 'score': 0.9493240714073181}
            ]
        ]

	'''
	if model_name:
		pipe = pipeline(model = model_name, task = task)
	else:
		pipe = pipeline(task = task)
	output_dict = pipe(list(docs), return_all_scores=return_all_scores)
	if return_all_scores:
		feature_vectors = huggingface_output_2_df(output_dict, col_prefix = col_prefix)
	else:
		feature_vectors = pd.DataFrame(output_dict)
	return feature_vectors

'''
docs = ['I am happy', 
        "I'm happy, but worried about tomorrow", 
        "I'm miserable", 
        "I'm sad, but hopeful", 
        'I am not happy', 
        'I wish I were happy', 
        "I'm sad, but hopeful",
        "Don't talk to me like that!", 
        "Really? I'm shocked!"]
        
features = huggingface(docs, col_prefix=None)      

'''