
import pandas as pd
import re
from collections import Counter
from .utils.count_words import word_count
# from text.utils.count_words import word_count
import numpy as np

def count_lexicons_in_doc(doc,tokens=[], return_zero = [], return_matches=False):
	# TODO, perhaps return match and context (3 words before and after)
	'''

	Args:
		doc:
		tokens: lexicon tokens
		return_zero:
		normalize:
		return_matches:

	Returns:

	'''

	# remove punctuation except apostrophes because we need to search for things like "don't want to live"
	text = re.sub("[^\w\d'\s]+",'',doc.lower())
	counter = 0
	matched_tokens = []
	for token in tokens:
		token = token.lower()
		matches = text.count(token)
		counter += matches
		if return_matches and matches>0:
			# print(matches, token)
			matched_tokens.append(token)



	if return_matches:
		return counter, matched_tokens
	else:
		return counter

def extract(docs,lexicons, normalize = True, return_zero =[], return_matches=False, add_word_count = True):
	# TODO: return zero is for entire docs, shouldnt it be for tokens?
	'''

	Args:
		docs:
		lexicons:
		normalize:
			divide by zero
		return_zero:

	Returns:

	'''
	#process all posts
	# docs is list of list
	# lexicons is dictionary {'category':[token1, token2, ...], 'category2':[]}
	docs = [doc.replace('\n', ' ').replace('  ',' ').replace('“', '').replace('”', '') for doc in docs]

	# feature_names = list(lexicons.keys())
	# full_column_names = list(df_subreddit.columns) + feature_names
	# subreddit_features = pd.DataFrame(columns=full_column_names)

	# word_counts = reddit_data.n_words.values
	# all words in subgroup

	feature_vectors = {}
	matches = {}

	for category in list(lexicons.keys()):

		lexicon_tokens = lexicons.get(category)
		if return_matches:
			counts_and_matched_tokens = [count_lexicons_in_doc(doc,tokens=lexicon_tokens,  return_zero = return_zero, return_matches=return_matches) for doc in docs]
			counts =  [n[0] for n in counts_and_matched_tokens]
			matched_tokens =  [n[1] for n in counts_and_matched_tokens if n[1]!=[]]
			matches[category] = matched_tokens



		else:
			counts = [count_lexicons_in_doc(doc,tokens=lexicon_tokens, return_zero = return_zero, return_matches=return_matches) for doc in docs]
		# one_category = one_category/word_counts #normalize

		feature_vectors[category]=counts


		# # feature_vector = extract_NLP_features(post, features) #removed feature_names from output
		# if len(feature_vector) != 0:
		#     raw_series = list(df_subreddit.iloc[pi])
		#     subreddit_features = subreddit_features.append(pd.Series(raw_series + feature_vector, index=full_column_names), ignore_index=True)

	# feature_vectors0   = pd.DataFrame(docs, columns = ['docs'])
	# feature_vectors = pd.concat([feature_vectors0,pd.DataFrame(feature_vectors)],axis=1)
	feature_vectors   = pd.DataFrame(feature_vectors)

	#     feature_vectors   = pd.DataFrame(docs)
	#     feature_vectors['docs']=docs

	if normalize:

		wc = word_count(docs, return_zero=return_zero)
		wc = np.array(wc)
		feature_vectors_normalized = np.divide(feature_vectors.values.T,wc).T
		feature_vectors = pd.DataFrame(feature_vectors_normalized, index = feature_vectors.index, columns=feature_vectors.columns)

	if add_word_count and normalize:
		feature_vectors['word_count'] = wc
	elif add_word_count and not normalize:
		wc = word_count(docs, return_zero=return_zero)
		feature_vectors['word_count'] = wc





		# feature_vectors = feature_vectors/wc

	if return_matches:
		# all lexicons
		matches_counter_d = {}
		for lexicon_name_i in list(lexicons.keys()):
			if matches.get(lexicon_name_i):
				x = Counter([n for i in matches.get(lexicon_name_i) for n in i])
				matches_counter_d[lexicon_name_i]= matches_counter_d[lexicon_name_i]={k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
		# Counter([n for i in matches.get(lexicon_name_i) for n in i]) for lexicon_name_i in lexicons_d.keys()]
		
		
		
		
		return feature_vectors, matches_counter_d
	else:
		return feature_vectors
