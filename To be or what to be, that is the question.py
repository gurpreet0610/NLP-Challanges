from sklearn.feature_extraction.text import  TfidfVectorizer
from nltk.tokenize import wordpunct_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re

def clean_up(text):
    clean_text = re.sub(r'[*|?|!|\'|"|#|.|,|)|(|\|/|”|“|_|:|;]', r' ', text.lower())
    clean_text = clean_text.replace('\n', ' ')
    clean_text = clean_text.strip()
    clean_text = ' '.join(clean_text.split())
    return clean_text


corp = open('corpus.txt', 'r').read()

pat = '\n{5,}\s+THE SECRET CACHE\n{5,}.*'
corp = re.search(pat, corp, flags=re.S+re.M).group()
corp = re.sub(r'-', r'', corp)
corp = clean_up(corp)
corp_toks = wordpunct_tokenize(corp)

labels = ['am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']

def generate_inp(corp_toks, ksize):
    data = []
    for i, tok in enumerate(corp_toks):
        if tok in labels:
            inp_toks = corp_toks[max(i-ksize, 0):i] + corp_toks[i+1:min(i+ksize, len(corp_toks))]
            txt = ' '.join(inp_toks)
            data.append(txt)
    
    return data


def generate_data(corp_toks, ksize=5):
    corp_nolabel = [tok for tok in corp_toks if tok in labels]
    data = generate_inp(corp_toks, ksize)
    
    return data, corp_nolabel


def generate_test(corp_toks, ksize=5):
    data = []
    for i, tok in enumerate(corp_toks):
        if tok == '----':
            inp_toks = corp_toks[max(i-ksize, 0):i] + corp_toks[i+1:min(i+ksize, len(corp_toks))]
            txt = ' '.join(inp_toks)
            data.append(txt)
    
    return data

X, y = generate_data(corp_toks)

pipe = Pipeline([
    ('tf', TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode')),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=50))
])

pipe.fit(X, y)

N = input()
test_corp = input()
test_corp = clean_up(test_corp)
test_toks = wordpunct_tokenize(test_corp)
test_txts = generate_test(test_toks)
test_preds = pipe.predict(test_txts)

for p in test_preds:
    print(p)



