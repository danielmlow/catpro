
# authors: William Ma, Daniel M. Low

import math
import pandas as pd
from .utils.stop_words import remove
from .utils.lemmatizer import spacy_lemmatizer
from .utils.tokenizer import spacy_tokenizer

concreteness_df = pd.read_excel('http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx')

def concreteness_text(document, concreteness_df, remove_stopwords=False):
	# initialization
	total_concreteness = count = 0
	docs = document[0].split()
	docs_cleaned = remove(docs, language = 'en', remove_punct=True, exclude_stopwords=True) if remove_stopwords else docs
	docs_words = spacy_lemmatizer(docs_cleaned)
	df_list = concreteness_df['Word'].tolist()

	# flatten
	temp = []
	for lst in docs_words:
		for elt in lst:
			temp.append(elt)
	docs_words = temp

	# calculate totals and count
	for word in docs_words:
		if word not in df_list:
			continue
		index = concreteness_df['Word'].tolist().index(word)
		total_concreteness += concreteness_df['Conc.M'][index]
		count += 1

	concreteness_avg = total_concreteness / count if count else 0

	# calculate stdev
	sum_squares = 0
	for word in docs_words:
		if word not in df_list:
			continue
		index = concreteness_df['Word'].tolist().index(word)
		deviation = concreteness_df['Conc.M'][index] - concreteness_avg
		sum_squares += deviation * deviation

	variance = sum_squares / (count - 1)
	concreteness_stddev = math.sqrt(variance)
	return total_concreteness, concreteness_avg, concreteness_stddev



def concreteness_sentence(docs):
	concreteness_sums = []
	concreteness_avgs = []
	tokenized = spacy_tokenizer(docs, language = 'en', model='en_core_web_sm',
	                            token = 'clause',lowercase=False, display_tree = False,
	                            remove_punct=True, clause_remove_conj = True)
	# print(tokenized)
	for j in range(len(tokenized)):
		for i in tokenized[j]:
			concreteness_sums.append(concreteness_text([i])[0])
			concreteness_avgs.append(concreteness_text([i])[1])
	return concreteness_sums, concreteness_avgs