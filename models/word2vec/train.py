# models/word2vec/train.py

import os
from gensim.models import Word2Vec
from gensim.downloader import load as gensim_load

OUTPUT = os.path.join(os.path.dirname(__file__), 'weights', 'word2vec.bin')

# text8 is a standard clean Wikipedia sample — ships with gensim, no scraping
print('Downloading text8 corpus...')
corpus = gensim_load('text8')

print('Training Word2Vec...')
model = Word2Vec(
    sentences  = corpus,
    vector_size= 100,
    window     = 5,
    min_count  = 5,
    sg         = 1,       # Skip-gram
    workers    = 4,
    epochs     = 5,
    seed       = 42
)

print(f'Vocab size: {len(model.wv)}')
print(f'Similar to "disease": {[w for w,_ in model.wv.most_similar("disease", topn=5)]}')

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
model.wv.save_word2vec_format(OUTPUT, binary=True)
print(f'Saved to {OUTPUT}')