
import spacy

import string
def remove_punctuation(doc):

	return doc.translate(str.maketrans('', '', string.punctuation)) #remove punctuation


def remove_extra_white_space(doc):
	punctuations_closing = ['.', ',','!', '?',']',')']
	punctuations_opening =  ['(', '[', '$']
	# punctuations_both = []

	for punctuation in punctuations_closing:
		doc = doc.replace(' '+punctuation,punctuation )
	for punctuation in punctuations_opening:
		doc = doc.replace(punctuation+' ',punctuation )
	return doc


def spacy_normalizer(docs):

	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'lemmatizer'])

	spacy_docs = list(nlp.pipe(docs))
	docs_processed = []
	for doc, spacy_doc in zip(docs,spacy_docs):
		doc = ' '.join([str(t.norm_).replace("'ve", 'have') for t in spacy_doc])
		doc = remove_extra_white_space(doc)
		docs_processed.append(doc)
	return docs_processed