import spacy

def spacy_lemmatizer(docs, language = 'en'):
	# docs = ['recovery limbo anyone else recovered still unhealthy relationship foodbody basically eating maintenance past 6 years period quite severe anorexia 14ish still feel like barely changed still feel proud skip meal feel bad body etc anyone else similar boat feels like better long time guess',
	#         'food worst drug hey guys fat fatty fat thinking sucks quit food eat like dopamine stop quit smoking drinking weed heroin meth food sucks relationship food sucks hate sucks thanks coming rant wanted chest',
	#         'focus anything bring eat guess going fail finals one freshman fuck ups taken gap year feel consumed pain crippling falling knees type']
	# todo my_module = importlib.import_module("spacy.lang."+language) # from spacy.lang.en import English

	if language=='en':
		nlp = spacy.load('en_core_web_sm', disable = ['parser','ner'])
	docs_lemmatized = []
	for doc in docs:
		doc = nlp(doc)
		doc_lemmatized = [token.lemma_ for token in doc]
		docs_lemmatized.append(doc_lemmatized)
	return docs_lemmatized


'''
import catpro.utils.stop_words as stop_words
docs =['alone', "I've been worried but hopeful", "I've been feeling all alone but hopeful and I'll do therapy. Gotta take it step by step."] 
docs = spacy_lemmatizer(docs)
docs = [stop_words.remove(i) for i in docs]
print(docs)
>>> [['alone'], ['', '', '', 'worry', '', 'hopeful'], ['', '', '', 'feel', '', 'alone', '', 'hopeful', '', '', '', '', 'therapy', '', 'got', '', 'take', '', 'step', '', 'step', '']]


docs =['alone', "I've been worried but hopeful", "I've been feeling all alone but hopeful and I'll do therapy. Gotta take it step by step.", "I've been feeling all alone but hopeful and I'll do therapy. Gotta take it step by step."]
docs = stop_words.remove(docs)
docs = spacy_lemmatizer(docs)
print(docs)
>>> [['alone'], ['worried', 'hopeful'], ['feel', 'alone', 'hopeful', 'ill', 'therapy', 'got', 'to', 'take', 'step', 'step']]

'''
