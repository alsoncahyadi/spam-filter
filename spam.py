from nltk import tokenize, pos_tag
import sklearn
import glob

ENRON_PATH = "enron3/"

def to_features(word):
	features = {
		'word': word,
		'POS tag': word,
	}
	return features

def to_dataset(raw_email, is_spam):
	tokens = tokenize.word_tokenize(raw_email)
	pos_tagged_tokens = pos_tag(tokens)
	for pos_tagged_token in pos_tagged_tokens:
		pos_tagged_token = pos_tagged_token + (is_spam,)
	return pos_tagged_tokens, is_spam

if __name__ == "__main__":
	# Read Enron dataset
	dataset = []
	
	for filename in glob.glob(ENRON_PATH + "ham/*.txt"):
		with open(filename, encoding="ISO-8859-1") as fin:
			dataset.append(to_dataset(fin.read(), 0))

	for filename in glob.glob(ENRON_PATH + "spam/*.txt"):
		print(filename)
		with open(filename, encoding="ISO-8859-1") as fin:
			dataset.append(to_dataset(fin.read(), 1))

	print(len(dataset))
	X, y = dataset[0]
	print(X)
	print(y)
