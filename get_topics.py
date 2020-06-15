import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

import argparse

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)
"""
implemented topic modeling using Gensim

for optimal performance, please run sentiment_score_generator first to clean up tweet content

this file is built largely based on this article here:
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
"""

parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--path', type=str,
                    help='File path to COVID-19 tweets JSON',
                    default='hash_tag_covid_processed.json')
parser.add_argument('--topic_number', type=int,
                    help='The number for latent topics for this LDA topic model (if not auto select)',
                    default=10)
parser.add_argument('--auto_topic', type=bool,
                    help='Auto select the optimal number of latent topics',
                    default=True)
args = parser.parse_args()

# Import Dataset
tweets_file = open(args.path, "r")
df = pd.read_json(tweets_file, lines=True)
print("Preview of loaded data : \n")
print (df)
print ("______________ \n \n ")

# Convert to list
data = df.text.values.tolist()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print ("Preview of first three tweets after lemmatization: \n \n ")
print(data_lemmatized[:3])
print ("______________ \n \n ")

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print("Preview of first three tweets after substituted by their ID in dictionary: \n \n")
print(corpus[:3])
print ("______________ \n \n ")


# Directly built model / Build test model to get optimal latent number
if not args.auto_topic:
    latent_num = args.topic_number
    print ("Topic number given: ", latent_num)
    print("______________ \n \n ")

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

else:
    print ("Topic number NOT given. Training multiple models. Optimal one will return at the end...")
    print("______________ \n \n ")
    # this may take a long time - every model in the list will be evaluated based on coherence
    coherence_values = []
    model_list = []
    for num_topics in range(5, 20, 1):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        pprint(model.print_topics())
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        this_model_coherence = coherencemodel.get_coherence()
        coherence_values.append(this_model_coherence)
        print ("Coherence Score: ", this_model_coherence)
        print ("______________ \n \n ")
    best_coherence_index = coherence_values.index(max(coherence_values))
    best_model = model_list[best_coherence_index]
    print("______________ \n \n Here is the best one: ")
    pprint(best_model.print_topics())
    print("Coherence Score: ", max(coherence_values))
