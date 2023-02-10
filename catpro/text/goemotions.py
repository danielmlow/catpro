
# pip install -q transformers
from transformers import pipeline
sentiment_pipeline = pipeline(model = "bhadresh-savani/distilbert-base-uncased-emotion")
import pandas as pd


def huggingface_output_2_df(output_dict, add_to_col_names = None):
	feature_names = [n.get('label') for n in output_dict[0]]
	if add_to_col_names:
		feature_names = [add_to_col_names+n for n in feature_names]
	feature_vectors = []
	for doc in output_dict:
		feature_vectors_doc = []
		for feature in doc:
			feature_vectors_doc.append(feature.get('score'))
		feature_vectors.append(feature_vectors_doc)
	feature_vectors = pd.DataFrame(feature_vectors, columns = feature_names)
	return feature_vectors

'''

docs = ['I am happy', 'I have happy, but worried about tomorrow', "I'm miserable", "I'm sad, but hopeful", "Don't talk to me like that!", "Really? I'm shocked!"]
output_dict = sentiment_pipeline(list(docs), return_all_scores=True)
feature_vectors = huggingface_output_2_df(output_dict, feature_names = ['distilbert_sadness', 'distilbert_joy', 'distilbert_love', 'distilbert_anger', 'distilbert_fear', 'distilbert_surprise'])
feature_vectors

'''

# https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion
# models = ["bhadresh-savani/distilbert-base-uncased-emotion",
#           'arpanghoshal/EmoRoBERTa']
