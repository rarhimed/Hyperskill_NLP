from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from string import punctuation
from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter


STOP_TOKENS = list(punctuation) + stopwords.words('english')
lemmatizer = WordNetLemmatizer()

root = etree.parse('news.xml').getroot()
corpus = root[0]

all_topics = []
all_news = []

for news in corpus:
    title, content = news[0].text, news[1].text
    all_topics.append(title)
    # 1. Get rid of punctuation and stop words
    error_dict = {'ha': 'has',
                  'wa': 'was',
                  'u': 'us',
                  'a': 'as'}

    tokens = []
    for token in word_tokenize(content.lower()):
        lemm_token = lemmatizer.lemmatize(token) if token not in error_dict.values() else token
        if lemm_token not in STOP_TOKENS:
            tokens.append(lemm_token)
    # 2. Keep only "NN" tokens
    tokens = [token for token in tokens if pos_tag([token])[0][1] == "NN"]

    # tokens = [token if token not in error_dict else error_dict[token] for token in tokens]
    print(title, Counter(tokens), sep='\n')
    # if 'us' in title.lower():
    #     print(tokens)
    # 3. Save tokenized text in our corpus (all_news)
    all_news.append(" ".join(tokens))

# 4. Apply TF-IDF
vectorizer = TfidfVectorizer()
weighted_matrix = vectorizer.fit_transform(all_news)
terms = vectorizer.get_feature_names()
for i, doc in enumerate(weighted_matrix):
    tfidf_sorting = np.argsort(doc.toarray()).flatten()[::-1]
    top_5 = tfidf_sorting[:5]
    top_5_freq = [(terms[word], weighted_matrix[(i, word)]) for word in top_5]
    top_5_sorted = sorted(top_5_freq, reverse=True, key=lambda x: x[0])  # sort by name
    top_5_sorted = sorted(top_5_sorted, reverse=True, key=lambda x: x[1])  # sort by value
    top_5_print = " ".join([k for k, _ in top_5_sorted])
    print(f'{all_topics[i]}:\n{top_5_print}')
