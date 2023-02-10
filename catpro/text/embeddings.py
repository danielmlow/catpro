
'''
pip install --upgrade jax jaxlib
pip install -q flax
'''

import numpy as np
from sentence_transformers import util
import torch

# from utils.backend._utils import select_backend



# # Create the parser
# parser = argparse.ArgumentParser(description='Convert all mp3 in input_dir to other format such as wav')
# # Add an argument
# parser.add_argument('--input_dir', type=str, required=True)
# parser.add_argument('--output_dir', type=str, required=True)
# parser.add_argument('--output_format', type=str,default='wav')
# parser.add_argument('--output_bitrate', type=str,default='32k')

# # Parse the argument
# args = parser.parse_args()
# input_dir = args.input_dir
# output_dir = args.output_dir
# output_format = args.output_format
# output_bitrate = args.output_bitrate

def cosine_similarity(embeddings1, embeddings2):
	# from sklearn.metrics.pairwise import cosine_similarity
	#Compute cosine-similarits
	cosine_scores = util.cos_sim(np.array(embeddings1,dtype=float), np.array(embeddings2,dtype=float))
	# print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
	return cosine_scores


def cosine_similarity_target(target_embeddings, embeddings):
	from scipy.spatial import distance

	distances = distance.cdist([target_embeddings], embeddings, "cosine")[0]
	min_index = np.argmin(distances)
	min_distance = distances[min_index]
	max_similarity = 1 - min_distance
	return distances, min_index, min_distance, max_similarity



def semantic_search(corpus_embeddings, query_embeddings):

	# corpus_embeddings = corpus_embeddings.to('cuda')
	corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

	# query_embeddings = query_embeddings.to('cuda')
	query_embeddings = util.normalize_embeddings(query_embeddings)
	hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)
	return hits

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask,layer = 12):
    '''

	Args:
		model_output:
		attention_mask:
		layer:
			see all responses here https://stackoverflow.com/questions/63461262/bert-sentence-embeddings-from-transformers
			see here as well:https://stackoverflow.com/questions/61323621/how-to-understand-hidden-states-of-the-returns-in-bertmodelhuggingface-transfo

	Returns:

	'''
    token_embeddings = model_output.get('hidden_states')[layer]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def vectorize(docs, list_of_lists = False, package = 'flair', embedding_type = 'sentence', model_name = 'default'):
	'''
	Args:
		docs: list (each element is a string for an entire document) or list of lists (each inner list represents a
		documents and contains tokens (words, clauses, sentences)). See list_of_lists argument below.

		list_of_lists : bool, default=False
			if False, function expects a list of strings.
				[ 'happy', 'table', ... ]

			if True, function expects a list of list of strings
				[   ['I', 'thought', ...],
					['By', 'the', ...],
					...
				]
				or
				[   ['I went to the movies', 'I hated it', ...],
					['By the time she arrived I had already left', 'cannot stand shouting', ...],
					...
				]


		package : {'sentence_transformers', 'flair', 'transformers'}, default='flair'
			Python packages
			sentence_transformers (aka, sBERT)  https://github.com/UKPLab/sentence-transformers
			flair                               https://github.com/flairNLP/flair
			transformers (aka huggingface)      https://github.com/huggingface/transformers

		embedding_type : {'word', 'transformer_word', 'sentence', 'document'}, default='sentence'
			Only applicable if package == 'flair'
			Embeddings can be created for words, sentences, and documents.
			word : traditional word embeddings (glove, word2vec)

		model_name : {'default' or see under each package type for URLs to list of model names},
			default for
			default for ='all-MiniLM-L6-v2'

	Returns:
		array of embeddings

	'''

	from flair.data import Sentence
	if package=='sentence_transformers':
		from sentence_transformers import SentenceTransformer
		# https://www.sbert.net/docs/pretrained_models.html

		if model_name == 'default':
			model_name = 'all-MiniLM-L6-v2' #fast and high performing

		print(f'encoding {package} model: {model_name}')
		model = SentenceTransformer(model_name) # model = select_backend(model_name)
		embeddings = model.encode(docs)
		# embeddings = embeddings[index].reshape(1, -1)
		print('docs x embedding size:', embeddings.shape)
		return embeddings
	elif package == 'flair':
		# todo file_download.py:624: FutureWarning: `cached_download` is the legacy way to download files from the HF hub, please consider upgrading to `hf_hub_download`
		#   FutureWarning,

		if embedding_type == 'word':
			from flair.embeddings import WordEmbeddings
			# model_names: https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md
			if model_name == 'default':
				model_name = 'glove'
			embedder = WordEmbeddings(model_name) # embedder = DocumentPoolEmbeddings([embedder])

			# Sentence() can take a list of words, a sentence or a document and it will turn it into a single string and tokenize into words.


		elif embedding_type == 'transformer_word':
			# Transformer word embeddings
			# ====================================
			# documentation: https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md
			# models: https://huggingface.co/transformers/v2.3.0/pretrained_models.html
			from flair.embeddings import TransformerWordEmbeddings
			if model_name == 'default':
				model_name = 'bert-base-uncased'

			embedder = TransformerWordEmbeddings(model_name)# init embedding

		# =========================================================================================================
		# TODO
		# More Word embeddings: ELMO, Flair
		# https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md

		# Also, average pool or RNN word embeddings
		# https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md#embeddings
		# =========================================================================================================

		elif embedding_type == 'sentence':
			# same as using sentence_transformers, but more decimals (sentence_transformers rounds to 8)
			# Embeddings from sBERT
			# https://www.sbert.net/docs/pretrained_models.html
			# https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
			# https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md#sentencetransformerdocumentembeddings
			from flair.embeddings import SentenceTransformerDocumentEmbeddings

			if model_name == 'default':
				model_name = 'all-MiniLM-L6-v2'

			embedder = SentenceTransformerDocumentEmbeddings(model_name)
		# 	todo: file_download.py:624: FutureWarning: `cached_download` is the legacy way to download files from the HF hub, please consider upgrading to `hf_hub_download`
		#   FutureWarning,

		elif embedding_type == 'document':
			# Transformer document embeddings
			# ====================================
			# tutorial: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md#transformerdocumentembeddings
			# models from Huggingface: https://huggingface.co/models
			from flair.embeddings import TransformerDocumentEmbeddings

			if model_name == 'default':
				model_name = 'bert-base-uncased'
			# todo: play around with layers and layer_mean

			embedder = TransformerDocumentEmbeddings(model_name,layers='-1',layer_mean=False)# init embedding

		if 'word' in embedding_type:
			# docs = ['happy', 'sad']
			print(f'encoding {package} {embedding_type} model: {model_name}')
			# doc.tokens will take the tokens you give or split into words by default and returns list of list either way
			flair_docs = [embedder.embed(Sentence(doc))[0] for doc in docs]
			embeddings = np.array([np.array([token.embedding.cpu().detach().numpy() for token in doc.tokens], dtype=object)
			                       for doc in flair_docs], dtype=object)
			print('docs:', embeddings.shape)
			# print('tokens x embedding size:', [n.shape for n in embeddings])
			return embeddings
		else:
			print(f'encoding {package} {embedding_type} model: {model_name}')

			if list_of_lists:
				'''
				docs = [['i am happy', 'not today'], ['i went to the movies', 'i like popcorn', 'how about you']]
				'''
				# from datetime import datetime
				# start=datetime.now()
				embeddings = []
				for doc in docs:
					flair_tokens = [embedder.embed(Sentence(token))[0] for token in doc]
					docs_embeddings = np.array([flair_token.embedding.cpu().detach().numpy() for flair_token in flair_tokens], dtype=object)
					embeddings.append(docs_embeddings)
				embeddings = np.array(embeddings, dtype=object)
				# print('tokens x embedding size:', [n.shape for n in embeddings])
				# print(datetime.now()-start)
			else:
				if type(docs)==str:
					embeddings = np.array([embedder.embed(Sentence(docs))[0].embedding.cpu().detach().numpy()], dtype=object)
				else:
					# docs = ['i am a boy', 'you are a boy']
					# Warning: I had previously wrapped each embedding in an additional list: resulting in shape (len_of_docs,1,384). removed so I can compute similarity with single words (1,384)
					embeddings = np.array([embedder.embed(Sentence(doc))[0].embedding.cpu().detach().numpy() for doc in docs], dtype=object)

			print('docs x embedding size:', embeddings.shape)
			return embeddings

	elif package == 'transformers': #aka huggingface
		print('WARNING: not recommended: is slower than flair, and returns different output than flair and sentence_transformers. '
		      'Also each model might need a specific way to extract features, something that flair resolves for you.')
		# todo play around with pooling https://stackoverflow.com/questions/61323621/how-to-understand-hidden-states-of-the-returns-in-bertmodelhuggingface-transfo
		import torch.nn.functional as F
		from transformers import AutoTokenizer, AutoModel
		if embedding_type=='sentence':
			# Load model from HuggingFace Hub
			if model_name == 'default':
				model_name = 'sentence-transformers/all-mpnet-base-v2'

		elif embedding_type=='document':
			if model_name == 'default':
				# todo:
				pass

		from transformers import AutoModelForMaskedLM
		tokenizer = AutoTokenizer.from_pretrained(model_name)

		if model_name in ['mnaylor/psychbert-cased']:
			print('WARNING: jax not available on M1 chip: https://github.com/google/jax/issues/5501')
			model = AutoModelForMaskedLM.from_pretrained(model_name, from_flax=True) # it uses flax
		else:
			model = AutoModel.from_pretrained(model_name)
		encoded_input = tokenizer(docs, padding=True, truncation=True, return_tensors='pt')
		# Compute token embeddings
		with torch.no_grad():
			model_output = model(**encoded_input,output_hidden_states=True)
		# Perform pooling
		embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
		# Normalize embeddings
		embeddings = F.normalize(embeddings, p=2, dim=1)
		return embeddings


'''
docs =  ['i am a boy. How about you?', 'I went to the supermarket']
embeddings1 = vectorize(docs, package = 'sentence_transformers', model_name = 'all-MiniLM-L6-v2', embedding_type = 'sentence')
embeddings2 = vectorize(docs, package = 'flair', model_name = 'glove', embedding_type = 'word')
embeddings3 = vectorize(docs, package = 'flair', model_name = 'distilbert-base-uncased', embedding_type = 'transformer_word')
embeddings4 = vectorize(docs, package = 'flair', model_name = 'all-MiniLM-L6-v2', embedding_type = 'sentence')
embeddings5 = vectorize(docs, package = 'flair', model_name = 'distilroberta-base', embedding_type = 'document')
embeddings6 = vectorize(docs, package = 'flair', model_name = 'sentence-transformers/all-MiniLM-L6-v2', embedding_type = 'document')
embeddings6 = vectorize(docs, package = 'transformers', model_name = 'mnaylor/psychbert-cased', embedding_type = 'document')


docs = ['combat engineer', 'pets', 'substance use', 'disability']
embeddings = vectorize(docs, package = 'flair', model_name = 'bert-base-uncased', embedding_type = 'transformer_word')
for e in embeddings:
	print(e.shape)

embeddings = vectorize(docs, package = 'flair', model_name = 'bert-base-uncased', embedding_type = 'sentence')





# models from Huggingface: https://huggingface.co/models
model_name = 'distilbert-base-uncased'
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
embedder = TransformerDocumentEmbeddings(model_name,layers='-1',layer_mean=False)# init embedding
embeddings1 = embedder.embed(Sentence(docs))
print(embeddings1[0].embedding)

embeddings2 = np.array([embedder.embed(Sentence(doc))[0].embedding for doc in docs])
print(embeddings2.shape)
print(embeddings2[0].shape)
print(embeddings2[1].shape)



'''

	# # join words
	# seed_embeddings = model.embed([" ".join(seed_keywords)])
	# doc_embedding = np.average([doc_embedding, seed_embeddings], axis=0, weights=[3, 1])


# # Calculate distances and extract keywords
# if use_mmr:
# 	keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity)
# elif use_maxsum:
# 	keywords = max_sum_similarity(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates)
# else:
# 	distances = cosine_similarity(doc_embedding, candidate_embeddings)
# 	keywords = [(candidates[index], round(float(distances[0][index]), 4))
# 	            for index in distances.argsort()[0][-top_n:]][::-1]
#


# if __name__ == '__main__':
    # try:
    #     os.mkdir(output_dir)
    # except:
    #     pass

    # files = os.listdir(input_dir)
    # try:
    #     files.remove('.DS_Store')
    # except:
    #     pass
    # for n in files:
        # convert_mp3(input_dir+n,output_dir+n[:-3]+output_format, output_format=output_format )


'''
docs =  ['i am a boy. How about you?', 'I went to the supermarket']

e_w = vectorize(docs, embedding_type = 'word', model_name = 'all-MiniLM-L6-v2')
print(e_w.shape)



e_s = vectorize(docs, embedding_type = 'sentence', model_name = 'distilroberta-base')
e_d.shape = vectorize(docs, embedding_type = 'document', model_name = 'distilroberta-base')

e_w = vectorize(['hello', 'lonely'], embedding_type = 'word', model_name = 'distilroberta-base')
'''