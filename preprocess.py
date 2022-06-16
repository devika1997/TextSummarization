import numpy as np
import pandas as pd 
import re,string
from nltk.corpus import stopwords

class Preprocess:
    def __init__(self):
        print(" Starting Preprocessing")
        #from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
        self.contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

    def removenullandothercolumns(self,data):
        data.drop_duplicates(subset=['Text'],inplace=True)#dropping duplicates
        data.dropna(axis=0,inplace=True)#dropping na
        reviews = data.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time'],1)
        return reviews

    def clean(self,Text):

        Text=Text.lower()
        Text=Text.split()
        temp=[]
        for word in Text:
            if word in self.contraction_mapping:
                temp.append(self.contraction_mapping[word])
            else:
                temp.append(word)
        Text=" ".join(temp)

        Text=Text.split()
        # Combine stopwords and punctuation
        stops = set(stopwords.words("english") + list(string.punctuation))
        Text=[word for word in Text if not word in stops]
        Text=" ".join(Text)
        
        Text = re.sub(r'https?:\/\/.*[\r\n]*', '', Text, flags=re.MULTILINE)
        Text = re.sub(r'\<a href', ' ', Text)
        Text = re.sub(r'&amp;', '', Text) 
        Text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', Text)
        Text = re.sub(r'<br />', ' ', Text)
        Text = re.sub(r'\'', ' ', Text)
        
        return Text

    def retrieveclean(self,reviews):
        summaries = []
        for summary in reviews.Summary:
            summaries.append(self.clean(summary))
        texts = []
        for text in reviews.Text:
            texts.append(self.clean(text))
        return texts,summaries

    def count_words(self,count_dict, text):
        for sentence in text:
            for word in sentence.split():
                if word not in count_dict:
                    count_dict[word] = 1
                else:
                    count_dict[word] += 1
        return count_dict

    def embedding(self):
        embeddings_index = {}
        with open('numberbatch-en.txt', encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embedding

        return embeddings_index
        
    def vocab(self,vocab_to_int,embeddings_index,word_counts):
        value = 0
        for word, count in word_counts.items():
            if count >= 20 or word in embeddings_index:
                vocab_to_int[word] = value
                value += 1

        # Special tokens that will be added to our vocab
        codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

        # Add codes to vocab
        for code in codes:
            vocab_to_int[code] = len(vocab_to_int)

        # Dictionary to convert integers to words
        int_to_vocab = {}
        for word, value in vocab_to_int.items():
            int_to_vocab[value] = word
        return vocab_to_int,int_to_vocab

    def embeddingmatrix(self,vocab_to_int,embeddings_index):
        
        word_embedding_matrix = np.zeros((len(vocab_to_int), 300), dtype=np.float32)
        for word, i in vocab_to_int.items():
            if word in embeddings_index:
                word_embedding_matrix[i] = embeddings_index[word]
            else:
                # If word not in CN, create a random embedding for it
                new_embedding = np.array(np.random.uniform(-1.0, 1.0, 300))
                embeddings_index[word] = new_embedding
                word_embedding_matrix[i] = new_embedding

        return word_embedding_matrix

    def convert_to_ints(self,vocab_to_int,text, word_count, unk_count, eos=False):
        '''Convert words in text to an integer.
        If word is not in vocab_to_int, use UNK's integer.
        Total the number of words and UNKs.
        Add EOS token to the end of texts'''
        ints = []
        for sentence in text:
            sentence_ints = []
            for word in sentence.split():
                word_count += 1
                if word in vocab_to_int:
                    sentence_ints.append(vocab_to_int[word])
                else:
                    sentence_ints.append(vocab_to_int["<UNK>"])
                    unk_count += 1
            if eos:
                sentence_ints.append(vocab_to_int["<EOS>"])
            ints.append(sentence_ints)
        return ints, word_count, unk_count

    def create_lengths(self,texts,summaries):
        texts_lengths = []
        summaries_lengths=[]
        for sentence in texts:
            texts_lengths.append(len(sentence))
        
        for sentence in summaries:
            summaries_lengths.append(len(sentence))
        return pd.DataFrame(texts_lengths, columns=['counts']),pd.DataFrame(summaries_lengths, columns=['counts'])

    def unk_counter(self,vocab_to_int,sentence):
        '''Counts the number of time UNK appears in a sentence.'''
        unk_count = 0
        for word in sentence:
            if word == vocab_to_int["<UNK>"]:
                unk_count += 1
        return unk_count

    def sorting(self,text_lengths,vocab_to_int,int_texts,int_summaries):
        # Sort the summaries and texts by the length of the texts, shortest to longest
        # Limit the length of summaries and texts based on the min and max ranges.
        # Remove reviews that include too many UNKs

        sorted_summaries = []
        sorted_texts = []
        max_text_length = 87
        max_summary_length = 8
        min_length = 2
        unk_text_limit = 1
        unk_summary_limit = 0

        for length in range(min(text_lengths.counts), max_text_length): 
            for count, words in enumerate(int_summaries):
                if (len(int_summaries[count]) >= min_length and len(int_summaries[count]) <= max_summary_length and len(int_texts[count]) >= min_length and self.unk_counter(vocab_to_int,int_summaries[count]) <= unk_summary_limit and self.unk_counter(vocab_to_int,int_texts[count]) <= unk_text_limit and length == len(int_texts[count])):
                    
                    sorted_summaries.append(int_summaries[count])
                    sorted_texts.append(int_texts[count])
                
        return sorted_texts,sorted_summaries

    def text_to_seq(self,text,vocab_to_int):
        '''Prepare the text for the model'''
        
        text = self.clean(text)
        return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]