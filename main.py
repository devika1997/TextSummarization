from preprocess import Preprocess
import pandas as pd
import numpy as np
from model import graph
from os.path import exists
from test import testing
import warnings
warnings.filterwarnings("ignore")

def summarize():

    data=pd.read_csv("Reviews.csv")
    p=Preprocess()
    reviews=p.removenullandothercolumns(data)
    texts,summaries=p.retrieveclean(reviews)
    word_counts = {}

    word_counts=p.count_words(word_counts, summaries)
    word_counts=p.count_words(word_counts, texts)
    embeddings_index=p.embedding()
    
    vocab_to_int = {} 
    vocab_to_int,int_to_vocab=p.vocab(vocab_to_int,embeddings_index,word_counts)
    word_embedding_matrix=p.embeddingmatrix(vocab_to_int,embeddings_index)

    # Apply convert_to_ints to clean_summaries and clean_texts
    word_count = 0
    unk_count = 0

    int_summaries, word_count, unk_count = p.convert_to_ints(vocab_to_int,summaries, word_count, unk_count)
    int_texts, word_count, unk_count = p.convert_to_ints(vocab_to_int,texts, word_count, unk_count, eos=True)

    text_lengths,summaries_lengths=p.create_lengths(int_texts,int_summaries)

    s_texts,s_summaries=p.sorting(text_lengths,vocab_to_int,int_texts,int_summaries)

    y = s_summaries[200000:250000]
    x = s_texts[200000:250000]

    if(exists('best_model.ckpt.meta')):
        print("Testing starts")
        s=testing(texts,summaries,vocab_to_int,int_to_vocab,64)
    else:
        graph(word_embedding_matrix,vocab_to_int,x,y)
    #return s

if __name__ == "__main__":
    summarize()
    