


from transformers import pipeline


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



# https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion
# models = ["bhadresh-savani/distilbert-base-uncased-emotion",
#           'arpanghoshal/EmoRoBERTa']
