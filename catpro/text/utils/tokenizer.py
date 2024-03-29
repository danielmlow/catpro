

'''

Source: https://stackoverflow.com/questions/65227103/clause-extraction-long-sentence-segmentation-in-python

Alternatives:
- second response: https://stackoverflow.com/questions/39320015/how-to-split-an-nlp-parse-tree-to-clauses-independent-and-subordinate
- TODO: also consider subordinate clauses while, if, becuase, instead https://stackoverflow.com/questions/68616708/how-to-split-sentence-into-clauses-in-python

'''

import spacy
import deplacy
import importlib




def spacy_tokenizer(docs, language = 'en', model='en_core_web_sm', token = 'clause',lowercase=False, display_tree = False, remove_punct=True, clause_remove_conj = True, avoid_split_tokenizer=True):
	'''



	Args:
		docs:
		model:
		token:
		display_tree:

	Returns:

	Example:
	docs = ['Recovery limbo Anyone else "recovered", but still have an unhealthy relationship with food/body? I\'ve basically been eating maintenance for the past 6 years after a period of quite severe anorexia at 14-ish, but I still feel like I\'ve barely changed. Still feel proud if I skip a meal, feel bad about my body, etc.\n\n&amp;#x200B;\n\nAnyone else in a similar boat? Just feels like it should be better after such a long time I guess.',
        'Food is the worst "drug" Hey guys I\'m fat. I\'m a fatty fat and I was just thinking it sucks that you cant quit food. When I eat its like all the dopamine and I cant stop. You can quit smoking or drinking or weed or heroin or meth but not food and that sucks because my relationship with food sucks and I hate it and it sucks. \n\nThanks for coming to my rant I just wanted that off my chest',
        "I can't focus on anything but just can't bring myself to eat. So I guess I'm going to fail finals and be one of those freshman fuck ups who should've taken a gap year. I feel consumed by pain. The crippling, falling to your knees type."]

	'''
	# TODO: split if you find ";"
	# TODO: make into list comprehensions for faster processing
	if token == 'word':
		# doc = 'I am a boy'
		my_module = importlib.import_module("spacy.lang."+language) # from spacy.lang.en import English
		if language=='en':
			nlp = my_module.English()
			if avoid_split_tokenizer:
				nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}
		tokens_for_all_docs = []
		for doc in docs:
			doc = nlp(doc)
			if lowercase:
				tokens = [token.text.lower() for token in doc]
			else:
				tokens = [token.text for token in doc]
			tokens_for_all_docs.append(tokens)
		return tokens_for_all_docs

	elif token =='clause':
		nlp = spacy.load(model)
		if avoid_split_tokenizer:
			nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key}

		chunks_for_all_docs = []
		for doc in nlp.pipe(docs):
			# doc = en(text)
			if display_tree:
				print(doc)
				print(deplacy.render(doc))


			seen = set() # keep track of covered words
			chunks = []
			for sent in doc.sents:
				heads = [cc for cc in sent.root.children if cc.dep_ == 'conj']

				for head in heads:
					words = [ww for ww in head.subtree]
					if remove_punct:
						words = [n for n in words if not n.is_punct]
					for word in words:

						seen.add(word)
					if clause_remove_conj:
						chunk = []
						for i,word in enumerate(words):
							len_minus_1 = len(words)-1
							# print(i, word.tag_, word.text)
							if not (word.tag_=='CC' and i==len_minus_1):
								chunk.append(word)
						chunk = (' '.join([ww.text for ww in chunk]))
					else:
						# dont remove
						chunk = (' '.join([ww.text for ww in words]))
					chunks.append( (head.i, chunk) )

				unseen = [ww for ww in sent if ww not in seen]
				if remove_punct:
					unseen = [n for n in unseen if not n.is_punct]
				if clause_remove_conj:
					chunk = []
					for i,word in enumerate(unseen):
						# print(i, word.tag_, word.text)
						len_minus_1 = len(unseen)-1
						if not (word.tag_=='CC' and i==len_minus_1):
							chunk.append(word)
					chunk = (' '.join([ww.text for ww in chunk]))
				else:
					chunk = ' '.join([ww.text for ww in unseen])
				chunks.append( (sent.root.i, chunk) )

			chunks = sorted(chunks, key=lambda x: x[0])
			chunks = [n[1] for n in chunks]

			if lowercase:
				chunks = [n.lower() for n in chunks]

			chunks_for_all_docs.append(chunks)
		return chunks_for_all_docs
	else:
		raise ValueError("Possible tokens are 'word', 'clause', others are not implemented.")


'''
docs = ["I've been feeling all alone and I feel like a burden to my family. I'll do therapy, but I'm pretty hopeless."]
docs_tokenized = spacy_tokenizer(docs, language = 'en', model='en_core_web_sm', 
					token = 'clause',lowercase=False, display_tree = True, 
					remove_punct=True, clause_remove_conj = True)
print(docs_tokenized)


docs = ['I am very sad but hopeful and I will start therapy', 'I am very sad, but hopeful and I will start therapy', 
"I've been feeling all alone but hopeful and I'll do therapy. Gotta take it step by step."]
docs_tokenized = spacy_tokenizer(docs, language = 'en', model='en_core_web_sm', 
					token = 'clause',lowercase=False, display_tree = True, 
					remove_punct=True, clause_remove_conj = True)
print(docs_tokenized)
'''

def spacy_merge_tokenizer(docs, merge='entity', replace_entities = False):

	'''
	If you want to replace some entities and and just merge without replacing on other entities, first run with replace_entitites = ['entitity1, 'entitity2'] and then run replace_entitites = False
	:param doc:
	:param merge:
	:param replace_entities: False, ['all'], or list containing the ones to be replaced: Person, Norp, Fac, Org, Gpe, Loc, Product, Event, Work_Of_Art, Law, Language, Date, Time, Percent, Money, Quantity, Ordinal, Cardinal
	:return:
	'''
	nlp = spacy.load("en_core_web_sm", disable = ['parser', 'lemmatizer'])  # or "en_core_web_trf" which is for using or fine tuning transformer models. install here: https://spacy.io/models
	if merge == 'entity':
		nlp.add_pipe(nlp.create_pipe('merge_entities'))
		# nlp.add_pipe("merge_entities")
	elif merge == 'nouns':
		nlp.add_pipe(nlp.create_pipe('merge_noun_chunks'))
		# nlp.add_pipe("merge_noun_chunks")
		# texts = [t.text for t in nlp("I have a blue car")]
		# assert texts == ["I", "have", "a blue car"]

	# Example
	# doc = 'I met John, Mary, and Michael Jordan at Madison Square Garden on January 9, 2021. The ticket cost me 100 dollars ($100), and I bought it on the phone.'
	# spacy_doc = nlp(doc)
	spacy_docs = list(nlp.pipe(docs))

	docs_processed = []
	if replace_entities:
		for doc, spacy_doc in zip(docs,spacy_docs):


			# replace identying names () for PERSON. However, probably not best to replace famous names (Obama, Trump)
			# best practice is to manually remove using visualizer because you could be removing "Barack Obama"
			ents = [str(tok) for tok in spacy_doc.ents]
			labels = [tok.label_ for tok in spacy_doc.ents]
			entities = dict(zip(ents, labels))

			# Todo fix: "I bought it on one phone" > "I bought it on CARDINAL phCARDINAL"

			for entity, label in entities.items():

				if label in replace_entities or 'all' in replace_entities:
					doc = doc.replace(entity, label)
			docs_processed.append(doc)
		return docs_processed
	else:
		import re
		for doc, spacy_doc in zip(docs,spacy_docs):
			doc = " ".join([str(token).replace(' ', '_') for token in spacy_doc])

			doc = remove_extra_white_space(doc)
			# for punctuation in punctuations_both:
			# 	doc = doc.replace(f' {punctuation} ',punctuation )
			docs_processed.append(doc)

		return docs_processed



remove_if_startswith = ['I thought of ', 'I thought about ', 'I had ', 'I imagined ', 'I just ','I was ']