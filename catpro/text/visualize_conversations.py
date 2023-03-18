import pandas as pd
import numpy as np


def print_conversation_ctl(df,
                       max = 10,
                       conversation_with = 'all',
                       conversation_id_col = 'conversation_id',
                       person_col = 'interaction',
                       timestamp_col = 'message_timestamp_utc',
                       message_col = 'message',
                       remove_seconds = False,
                       ):
	'''

	Args:
		df:
		max: to avoid hanging
		conversation_with: {'all', integer for first N; list of ids or names from conversation_id_col}
		conversation_id_col: only if conversation_with!='all'; {'chat_with', 'conversation_id'}
		person_col:
		timestamp_col:
		message_col:
		remove_seconds:

	Returns:

	'''
	df = df.sort_values([conversation_id_col, timestamp_col])
	if conversation_with == 'all':
		conversations_with = df[conversation_id_col].unique()
	elif type(conversation_with) == int:
		conversations_with = df[conversation_id_col].unique()[:conversation_with]
	conversations_with = conversations_with[:max]
	# else, it is a list defined in conversations_with
	for conversation_id in conversations_with:
		print(f'\n===={conversation_id}')
		df_i = df[df[conversation_id_col]==conversation_id]
		conversation_messages = df_i[[timestamp_col,person_col,message_col]].values
		for message in conversation_messages:
			timestamp = message[0]
			if remove_seconds:
				timestamp = message[0][:-2]
			print(f'{timestamp} {message[1]}: \t{message[2]}')
	return




def print_conversation_instagram(df,
                                 max = 10,
                                 conversation_with = 'all',
                                 conversation_id_col = 'conversation_id',
                                 person_col = 'interaction',
                                 message_from_col = 'message_from',
                                 timestamp_col = 'message_timestamp_utc',
                                 message_col = 'message',
                                 remove_seconds = False,
                                 ):
	'''

	Args:
		df:
		max: to avoid hanging
		conversation_with: {'all', integer for first N; list of ids or names from conversation_id_col}
		conversation_id_col: only if conversation_with!='all'; {'chat_with', 'conversation_id'}
		person_col:
		timestamp_col:
		message_col:
		remove_seconds:

	Returns:

	'''
	df = df.sort_values([conversation_id_col, timestamp_col])
	if conversation_with == 'all':
		conversations_df = df[conversation_id_col].unique()
	elif type(conversation_with) == int:
		conversations_df = df[conversation_id_col].unique()[:conversation_with]
	elif type(conversation_with) == str:
		conversations_df = df[df[person_col] == conversation_with]

	conversations_df = conversations_df[:max]
	# else, it is a list defined in conversations_with

	conversations_df_messages = conversations_df[[timestamp_col,message_from_col,message_col]].values
	for message in conversations_df_messages:
		timestamp = message[0]
		if remove_seconds:
			timestamp = message[0][:-2]
		print(f'{timestamp} {message[1]}: \t{message[2]}')
	return


def message_context(df, message, message_col = 'event', timestamp_col = 'message_timestamp_utc',
                    conversation_id_col = 'event_type',
                    person_col = 'message_from',
                    display_pre = 5, display_post = 5, remove_seconds = False):

	df = df.sort_values([timestamp_col, conversation_id_col])
	message_index = df[df[message_col]==message].index[0]
	event_type = df[df[message_col]==message][conversation_id_col].values[0]
	df_i = df.loc[message_index-display_pre:message_index+display_post]
	df_i = df_i[df_i[conversation_id_col] == event_type] #in case they chated with other people in the prior messages']

	conversation_messages = df_i[[timestamp_col,person_col,message_col]].values
	for message in conversation_messages:
		timestamp = message[0]
		if remove_seconds:
			timestamp = message[0][:-2]
		print(f'{timestamp} {message[1]}: \t{message[2]}')


	return df_i