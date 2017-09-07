import os
import numpy
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfTransformer

NEWLINE = '\n'

HAM = 'ham'
SPAM = 'spam'

IS_LEMMATIZED = False

SOURCES = [
	('data/enron1/ham', HAM),
	('data/enron1/spam', SPAM)
    # ('data/spam',        SPAM),
    # ('data/easy_ham',    HAM),
    # ('data/hard_ham',    HAM),
    # ('data/beck-s',      HAM),
    # ('data/farmer-d',    HAM),
    # ('data/kaminski-v',  HAM),
    # ('data/kitchen-l',   HAM),
    # ('data/lokay-m',     HAM),
    # ('data/williams-w3', HAM),
    # ('data/BG',          SPAM),
    # ('data/GP',          SPAM),
    # ('data/SH',          SPAM)
]

SKIP_FILES = {'cmds'}
stemmer = SnowballStemmer('english')

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    lines = []
                    f = open(file_path, encoding="latin-1")
                    for i, line in enumerate(f):
                        if i > 0:
                            lines.append(line)
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content

def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        if IS_LEMMATIZED:
            tokens = word_tokenize(text)
            for token in tokens:
                stemmer.stem(token)
                text = " ".join(tokens)
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

def print_classification_report(classifier):
	labels = list(classifier.classes_)
	metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=4
    )

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(numpy.random.permutation(data.index))

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer()),
    # ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',         MultinomialNB())
])

k_fold = KFold(n=len(data), n_folds=6)

f1_scores = []
recall_scores = []
precision_scores = []

confusion = numpy.array([[0, 0], [0, 0]])
for i, (train_indices, test_indices) in enumerate(k_fold):
    print(i)
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    f1 = f1_score(test_y, predictions, pos_label=SPAM)
    f1_scores.append(f1)
    # recall_scores.append(recall)
    # precision_scores.append(precision)

print('Total emails classified:', len(data))
print('Is Lemmatized:', IS_LEMMATIZED)
print('Score:')
print('    > F1        :', sum(f1_scores)/len(f1_scores))
# print('    > Precision :', sum(precision_scores)/len(precision_scores))
# print('    > Recall    :', sum(recall_scores)/len(recall_scores))
print('Confusion matrix:')
print(confusion)