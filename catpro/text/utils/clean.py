import re
import string


def remove_puctuation(doc):
	doc = doc.translate(str.maketrans('', '',string.punctuation))
	return doc


def remove_multiple_spaces(doc):
	try: return re.sub(' +', ' ', doc)
	except:
		print('could not parse:', doc)