from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from string import punctuation
from lxml import etree
from collections import Counter


STOP_TOKENS = list(punctuation) + stopwords.words('english')
lemmatizer = WordNetLemmatizer()

root = etree.parse('news.xml').getroot()
corpus = root[0]

for news in corpus:
    title, content = news[0].text, news[1].text
    # 1. Get rid of punctuation and stop words
    tokens = []
    for token in word_tokenize(content.lower()):
        lemm_token = lemmatizer.lemmatize(token)
        if lemm_token not in STOP_TOKENS:
            tokens.append(lemm_token)
    # 2. Keep only "NN" tokens
    tokens = [token for token in tokens if pos_tag([token])[0][1] == "NN"]

    tokens = sorted(tokens, reverse=True)
    key_terms = Counter(tokens)

    top_5_terms = " ".join([k for k, v in key_terms.most_common(5)])
    print(f'{title}:\n{top_5_terms}')
