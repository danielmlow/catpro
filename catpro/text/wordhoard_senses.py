#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import logging
logging.basicConfig(level = logging.INFO)
import datetime
from IPython.display import display

# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')

from wordhoard import Antonyms, Synonyms, Hypernyms, Hyponyms, Homophones, Definitions


def stored_searches(path_stored_searches_df):
	'''

	Args:
		path_to_stored_df: df resulting from searching with with gen_wordhoard(). Load here to avoid searching again and maxing out requests to dictionary servers.

	Returns:
		stored_words: words for which results already exist

	'''
	stored_searches_df = pd.read_csv(path_stored_searches_df, index_col = 0)
	words_searched = stored_searches_df['category'].unique()
	return stored_searches_df, words_searched




def gen_wordhoard(word, search_for = 'synonyms',
                  output_format='dictionary',
                  max_number_of_requests=30,
                  rate_limit_timeout_period=60,
                  user_agent=None,
                  proxies=None,
                  split_by_senses=True
                  ):
	"""
	Obtain synonyms, definitions, hypernyms, etc. related to a word.

	Built on top of wordhoard https://wordhoard.readthedocs.io/en/latest/basic_usage/
	- Good to keep other thesauri beyond Merriam because Merriam can return None (e.g., paranoia)

	Added:
		-split by dictionary
		-split by senses

	Args:
		word: str
		search_for: {'antonyms', 'synonyms', 'hypernyms', 'hyponyms', 'homophones', 'definitions'}, default='synonyms'
			hypernyms are superordinate
		output_format: {'dictionary', 'json'}, default='dictionary'
		max_number_of_requests: int, default=30
		rate_limit_timeout_period: int, default=60
		user_agent: default=None
		proxies: default=None
		split_by_senses: bool, default=True
			Only implemented for merriam_webster
		words_searched: add list of words already searched to avoid maxing out the requests to dictionaries. If you prefer to search again set to None or remove word from words_search.

	Returns:
		dictionary or json of words with what was searched for (e.g., synonyms), split by thesauri/dictionary (e.g., merriam webster, synonyms.com), split by senses if option chosen.

	"""
	# Capitalizes first character to match method name (see imported methods from wordhoard)
	search_for = search_for.capitalize()
	if search_for!='Synonyms':
		import sys
		logging.warning('split_by_senses is only an argument for Synonyms. Need to implement ignore for other types.')
		sys.exit()
	result = eval(search_for)(word,
	                                       output_format=output_format,
	                                       split_by_senses=split_by_senses)
	results = eval(f'result.find_{search_for.lower()}()') #result becomes the variable once evaluated
	return results

def gen_wordhoard_df(words,input_dir_stored_searches=None, return_all_stored_searches=True, search_for='synonyms'):
	"""

	Args:
		words:
		input_dir_stored_searches: if you have searched for words again, add the dir where those searches were made so they are not searched for again (which can create requests issues if done many times).
		return_all_stored_searches: if True, all stored searches will also be returned. if False, then only strings in words will be returned (whether they are new or are retrieved from stored searches)
		search_for: {'antonyms', 'synonyms', 'hypernyms', 'hyponyms', 'homophones', 'definitions'}, default='synonyms'
			hypernyms are superordinate

	Returns:

	"""


	if input_dir_stored_searches==None:
		return_all_stored_searches = False #override if True because argument only viable if input_dir_stored_searches exists

	if input_dir_stored_searches:
		files = os.listdir(input_dir_stored_searches)#'./data/lexicons/thesauri_tokens_to_annotate.csv'
		files.sort()
		files = [n for n in files if n.endswith('.csv')]
		path_stored_searches_df = input_dir_stored_searches + files[-1] #obtain last file
		stored_searches_df = pd.read_csv(path_stored_searches_df, index_col = 0)
		words_searched = list(stored_searches_df['category'].unique())
	else:
		stored_searches_df = pd.DataFrame()
		words_searched = []
		os.makedirs('./data/', exist_ok=True)
		path_stored_searches_df = './data/' #it will save results in this data dir
	df = []
	columns = ['category', 'source', 'sense', 'sense_definition', 'tokens']
	cache_categories = []

	for word in words:


		if word in words_searched:
			logging.warning(f"word {word} already exists in words_searched, skipping.")
			continue
		else:
			cache_categories.append(word)
			logging.info(f'Searching for {word}...')
			results = gen_wordhoard(word, search_for = search_for,
			                        output_format='dictionary',
			                        max_number_of_requests=30,
			                        rate_limit_timeout_period=60,
			                        user_agent=None,
			                        proxies=None,
			                        split_by_senses=True)

			sense_i=0
			for source in list(results.keys()):

				if source == 'merriam_webster':
					senses = results.get(source)
					if senses != None:
						for sense_definition, sense_tokens in senses.items():
							df.append([word, source,sense_i, sense_definition, str(sense_tokens)])
							sense_i+=1
				else:
					sense_definition = 'nan'
					tokens = results.get(source)
					if tokens != None and str(tokens) != '[]':
						df.append([word, source, sense_i,sense_definition, str(tokens)])
						sense_i+=1
			logging.info('Done.')


	if df != []:
		# if there are new searches:
		words_df = pd.concat([pd.DataFrame(n, index = columns).T for n in df])
		# for n in df:
		# 	words_df_1_category = pd.DataFrame(n, index = columns).T
		words_df['annotation_remove'] = np.zeros(words_df.shape[0]) #create a column of zeros to then annotate, by default we keep
		#done above: words_df = words_df[words_df.tokens!='None'] #remove rows where wordhoard returned None
		if not stored_searches_df.empty:
			# if you have stored prior searches
			words_df_and_stored = pd.concat([stored_searches_df, words_df])
			ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
			words_df_and_stored = words_df_and_stored.reset_index(drop=True)
			words_df_and_stored.to_csv(input_dir_stored_searches+f'{search_for}_{ts}.csv')
		if return_all_stored_searches:
			return words_df_and_stored
		else:
			words_df = words_df_and_stored[words_df_and_stored['category'].isin(words)]
			words_df = words_df.reset_index(drop=True)
			return words_df
	elif not stored_searches_df.empty and return_all_stored_searches:
		# if no new results (df!=[]), stored_searches_df is not empty and return old searches
		return stored_searches_df
	elif not stored_searches_df.empty and not return_all_stored_searches:
		# return only requested words in stored searches
		words_df = stored_searches_df[stored_searches_df['category'].isin(words)]
		words_df = words_df.reset_index(drop=True)
		return words_df

	else:
		# this might happen if you search for a word not in dictionary and you don't have any word queries stored
		logging.warning("No stored searches and word/s not found, returning None.")
		return None

def remove_rows(words_df, categories='all'):
	"""

	Args:
		words_df:
		categories: 'all' or list of categories to annotate.

	Returns:

	"""
	# the index provides a rank for each category. So reset_index() and call it "entry". Then users can remove entries by rank number.
	if categories:
		if type(categories) != list:
			# 	then probably a single str, so make list to iterate through
			categories = [categories]
		categories_to_annotate = categories
	else:
		categories_to_annotate = words_df['category'].unique()

	for category in categories_to_annotate:
		category_annotation = []
		print(category)
		category_df = words_df[words_df['category']==category]
		pd.set_option('display.max_columns', None)
		display(category_df)
		category_df_values = category_df.values



		for sense in category_df_values:
			print('source: ',sense[1])
			print('sense #: ',sense[2])
			print('sense definition: ',sense[3])
			print('tokens: ',eval(sense[4]))
			# print(category_df['source'].values, category_df['sense'].values, category_df['sense_definition'].values)
			# 	print(eval(category_df['tokens'].values))
			# todo: show entire column and just respond, which senses to remove. 1,4.
			response = input("1 = remove (or enter to keep), or type quit to return what you've done so far without this category")
			if response == 'quit':
				return words_df
			elif response == '':
				response = 0
			elif response == '1':
				response = 1
			else:
				print('Your response was neither 1, enter/return (for 0) nor quit, inputting 0')
				response = 0
			category_annotation.append(response)
			print('\n')
		# after annotating all senses, add annotations to main words_df
		to_modify = words_df['annotation_remove'].to_list()
		indexes = list(category_df.index)
		for i, index in enumerate(indexes):
			to_modify[index] = category_annotation[i]
		words_df['annotation_remove'] =to_modify
		display( words_df[words_df['category']==category])
	return words_df



def add_tokens(words_df, category = None, tokens=None, source = None, sense_definition=None):
	category_df = words_df[words_df['category']==category]
	if source == None:
		source = 'manual'
	if not sense_definition:
		sense_definition=np.nan
	if category_df.shape[0]:
		sense = 0
	else:
		sense = np.max(category_df.sense.values)+1
	new_row = pd.DataFrame([category,source, sense, sense_definition, str(tokens), 0.0], index = category_df.columns.to_list()).T
	words_df = pd.concat([words_df,new_row])
	words_df = words_df.sort_values(by = ['category', 'sense'])
	words_df = words_df.drop_duplicates(subset=['category', 'source', 'tokens']) #in case the exact addition (same category, same tokens) is done again by accident.
	words_df = words_df.reset_index(drop=True)
	return words_df

def df_to_json(words_df):

	return


def remove_tokens(words_df, category = None, tokens=None):
	return











if __name__ == '__main__':
	# Parse arguments from command line
	import argparse
	# Create the parser
	parser = argparse.ArgumentParser(description='Convert all mp3 in input_dir to other format such as wav')
	# Add an argument
	parser.add_argument('--input_dir', type=str, required=True)
	parser.add_argument('--output_dir', type=str, required=True)
	parser.add_argument('--word', type=str,default='lonely')

	# Parse the argument
	args = parser.parse_args()
	input_dir = args.input_dir
	output_dir = args.output_dir
	word = args.output_format

	# main
	os.makedirs(output_dir, exist_ok = True)

	input_dir_stored_searches = './data/lexicons/wordhoard_synonyms/'

	# if path_stored_searches_df:
	# 	stored_searches_df, words_searched = stored_searches(path_stored_searches_df)


	# word = 'healthy'
	# results = gen_wordhoard(word, search_for = 'synonyms',
	#                         output_format='dictionary',
	#                         max_number_of_requests=30,
	#                         rate_limit_timeout_period=60,
	#                         user_agent=None,
	#                         proxies=None,
	#                         split_by_senses=True,
	#                         )
	# print(results)
	words = ['lonely', 'healthy', 'love']
	words_df = gen_wordhoard_df(words,input_dir_stored_searches=input_dir_stored_searches, return_all_stored_searches=True)
	categories=['insomnia', 'paranoid']
	words_df_annotated = remove_rows(words_df, categories=categories)





