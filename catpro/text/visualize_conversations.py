
def print_conversation(df,
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

