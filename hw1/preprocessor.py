import re

from nltk.stem import PorterStemmer, WordNetLemmatizer

stop_words = {'against', 'wouldn', 'haven', 'ain', 'all', 'been', 'has', 'again', "couldn't", 'too', 'this', 'but',
              'your', 'because', "should've", 'themselves', 'do', 'll', 'yourself', "don't", 'hers', 'are', 'under',
              "haven't", 'after', 'theirs', 'shouldn', 'd', 'same', 'very', 'won', 'y', 'those', 'such', 'don', 'as',
              "needn't", 'them', 'just', "weren't", 'both', "that'll", "you'd", 'where', 'her', 'other', "mightn't",
              'and', 'm', 'we', 'have', 'only', 'itself', 'most', 'shan', 'weren', 'myself', 'had', 'ours', 'am',
              "aren't", 'wasn', "shan't", "wouldn't", "you're", 'whom', 'they', 'does', 'how', 'hadn', 'own', "hadn't",
              'before', 'him', 'below', 'while', 'ma', 'here', 'a', 'into', 'my', "won't", "you'll", 'between', 'its',
              'for', 'from', "she's", 'needn', "hasn't", 'herself', 'some', 'did', "isn't", 'up', 'or', 'our', 'of',
              'who', 'is', 'above', 'which', 'than', 'there', 'about', 'himself', 'so', 'i', 'having', 'to', 'doesn',
              "wasn't", 'that', 'these', 'yours', 'if', 'down', 'once', 'when', 'didn', 'will', 'was', 'on', 't',
              'their', 'ourselves', 'further', 's', "it's", 'then', 'mustn', 've', 'it', 'now', "you've", "doesn't",
              'were', 'yourselves', 'couldn', "didn't", 'an', 'mightn', 'out', 'you', "shouldn't", 'he', 'nor', 'why',
              'at', 'the', 'being', 're', 'doing', 'can', 'each', "mustn't", 'me', 'she', 'what', 'no', 'until', 'any',
              'more', 'aren', 'during', 'isn', 'not', 'in', 'o', 'his', 'off', 'few', 'be', 'over', 'hasn', 'by',
              'should', 'through', 'with'}
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def scrub(words):
    scrubbed_words = []
    for word in words:
        # remove trailing spaces
        word = word.strip()
        # remove non-ascii chars and digits
        word = re.sub(r'(\W|\d)', '', word)
        if word:
            scrubbed_words.append(word)
    return scrubbed_words


def remove_stopwords(words):
    return [word for word in words if word not in stop_words]


def lower(words):
    return [word.lower() for word in words]


def stem(words):
    return [stemmer.stem(word) for word in words]


def lemmatize(words):
    return [lemmatizer.lemmatize(word, pos='v') for word in words]


def preprocess(sentence):
    words = sentence.split()
    words = scrub(words)
    words = remove_stopwords(words)
    words = lower(words)
    # words = stem(words)
    words = lemmatize(words)
    return ' '.join(words)
