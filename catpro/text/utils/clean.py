import re

def remove_multiple_spaces(doc):
	return re.sub(' +', ' ', doc)