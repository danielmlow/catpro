
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
    #Compute cosine-similarits
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    return cosine_scores


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def vectorize(docs, package = 'sentence_transformers', model_name = 'all-MiniLM-L6-v2', embedding_type = 'document'):
	'''

	Args:
		docs:
		package: sentence_transformers, flair
		model_name:
			'default'
			word:
			transformer_word:
			document:
			sentence:

		embedding_type: [only used if package == 'flair'], word, transformer_word, document, sentence

	Returns:
		array of embeddings

	'''
	from flair.data import Sentence
	if package=='sentence_transformers':
		from sentence_transformers import SentenceTransformer
		# https://www.sbert.net/docs/pretrained_models.html
		# model = select_backend(model_name)
		model = SentenceTransformer(model_name)
		embeddings = model.encode(docs)
		# embeddings = embeddings[index].reshape(1, -1)
		print('docs x embedding size:', embeddings.shape)
		return embeddings
	elif package == 'flair':
		if embedding_type == 'word':
			from flair.embeddings import WordEmbeddings
			# model_names: https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md
			if model_name == 'default':
				model_name = 'glove'
			embedder = WordEmbeddings(model_name)
			# embedder = DocumentPoolEmbeddings([embedder])

			# Sentence() can take a list of words, a sentence or a document and it will turn it into a single string and tokenize into words.
			flair_sentences = [embedder.embed(Sentence(doc))[0] for doc in docs]
			embeddings = []
			for doc in flair_sentences:
				embeddings.append(np.array([token.embedding.cpu().detach().numpy() for token in doc.tokens], dtype=object))
			embeddings = np.array(embeddings, dtype=object)
			print('docs:', embeddings.shape)
			print('tokens x embedding size:', [n.shape for n in embeddings])
			return embeddings

		elif embedding_type == 'transformer_word':
			# Transformer word embeddings
			# ====================================
			# documentation: https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md
			# models: https://huggingface.co/transformers/v2.3.0/pretrained_models.html
			from flair.embeddings import TransformerWordEmbeddings
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
			from flair.embeddings import SentenceTransformerDocumentEmbeddings
			embedder = SentenceTransformerDocumentEmbeddings(model_name)

		elif embedding_type == 'document':
			# Transformer document embeddings
			# ====================================
			# tutorial: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md#transformerdocumentembeddings
			# models from Huggingface: https://huggingface.co/models
			from flair.embeddings import TransformerDocumentEmbeddings
			# model_name = 'sentence-transformers/all-MiniLM-L6-v2'
			# todo: play around with layers and layer_mean
			embedder = TransformerDocumentEmbeddings(model_name,layers='-1',layer_mean=False)# init embedding
		if 'word' in embedding_type:
			flair_sentences = [embedder.embed(Sentence(doc))[0] for doc in docs]
			embeddings = []
			for doc in flair_sentences:
				embeddings.append(np.array([token.embedding.cpu().detach().numpy() for token in doc.tokens], dtype=object))
			embeddings = np.array(embeddings, dtype=object)
			print('docs:', embeddings.shape)
			print('tokens x embedding size:', [n.shape for n in embeddings])
			return embeddings
		else:
			embeddings = np.array([embedder.embed(Sentence(doc))[0].embedding.cpu().detach().numpy() for doc in docs], dtype=object)
			print('docs x embedding size:', embeddings.shape)
			return embeddings
	elif package == 'transformers': #aka huggingface
		print('WARNING: not recommended: is slower than flair, and returns different output than flair and sentence_transformers')
		if embedding_type=='sentence':
			from transformers import AutoTokenizer, AutoModel
			import torch.nn.functional as F

			# Load model from HuggingFace Hub
			tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
			model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

			# Tokenize sentences
			encoded_input = tokenizer(docs, padding=True, truncation=True, return_tensors='pt')

			# Compute token embeddings
			with torch.no_grad():
				model_output = model(**encoded_input)
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