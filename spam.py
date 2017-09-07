from nltk import tokenize
import sklearn
import glob

ENRON_PATH = "enron3/"

if __name__ == "__main__":
	# Read Enron dataset
	dataset = []
	
	for filename in glob.glob(ENRON_PATH + "ham/*.txt"):
		with open(filename, encoding="ISO-8859-1") as fin:
			dataset.append((tokenize.sent_tokenize(fin.read()), 0))

	for filename in glob.glob(ENRON_PATH + "spam/*.txt"):
		with open(filename, encoding="ISO-8859-1") as fin:
			dataset.append((tokenize.sent_tokenize(fin.read()), 1))

	print(len(dataset))
