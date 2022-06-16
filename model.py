import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
#https://github.com/Raksha-Ramesh/Amazon-Reviews-Summarization/blob/master/AmazonReviewSummarization.ipynb
#https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/master/How_to_build_own_text_summarizer_using_deep_learning.ipynb
#https://github.com/Currie32/Text-Summarization-with-Amazon-Reviews/blob/master/summarize_reviews.ipynb
def model_inputs():
    '''Create palceholders for inputs to the model'''
    
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length

def process_encoding_input(target_data, vocab_to_int, batch_size):  
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1]) # slice it to target_data[0:batch_size, 0: -1]
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,rnn_inputs,sequence_length,dtype=tf.float32)
            enc_output = tf.concat(enc_output,2)
            
            rnn_inputs = enc_output
    return enc_output, enc_state

def training_decoding_layer(dec_embed_input, summary_length, dec_cell, output_layer,vocab_size, max_summary_length,batch_size):
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,sequence_length=summary_length,time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,helper=training_helper,initial_state=dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                       output_layer = output_layer)

    training_logits = tf.contrib.seq2seq.dynamic_decode(training_decoder,output_time_major=False,impute_finished=True,maximum_iterations=max_summary_length)
    return training_logits

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, output_layer,max_summary_length, batch_size):
    
    
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens,end_token)
                
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,inference_helper,dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                                                        output_layer)
                
    inference_logits = tf.contrib.seq2seq.dynamic_decode(inference_decoder,output_time_major=False,impute_finished=True,maximum_iterations=max_summary_length)
    
    return inference_logits

def lstm_cell(lstm_size, keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    dec_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(rnn_size, keep_prob) for _ in range(num_layers)])
    output_layer = Dense(vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,enc_output,text_length,normalize=False,name='BahdanauAttention')
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,attn_mech,rnn_size)
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,summary_length,dec_cell,output_layer,vocab_size,max_summary_length,batch_size)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,vocab_to_int['<GO>'],vocab_to_int['<EOS>'],dec_cell,output_layer,max_summary_length,batch_size)
    
    return training_logits, inference_logits

def seq2seq_model(word_embedding_matrix,input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''
    
    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size) #shape=(batch_size, senquence length) each seq start with index of<GO>
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    training_logits, inference_logits  = decoding_layer(dec_embed_input,embeddings,enc_output,enc_state, 
                                                        vocab_size,text_length,summary_length,max_summary_length,rnn_size,vocab_to_int, 
                                                        keep_prob,batch_size,num_layers)
    return training_logits, inference_logits

def pad_sentence_batch(sentence_batch,vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(summaries, texts, batch_size,vocab_to_int):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch,vocab_to_int))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch,vocab_to_int))
        
        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
        
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        
        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


def graph(word_embedding_matrix,vocab_to_int,x,y):
    # Set the Hyperparameters
    epochs = 100
    batch_size = 64
    rnn_size = 256
    num_layers = 2
    learning_rate = 0.005
    keep_probability = 0.95
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():
        
        # Load the model inputs    
        input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()
        print(lr)
        # Create the training and inference logits
        training_logits, inference_logits = seq2seq_model(word_embedding_matrix,tf.reverse(input_data, [-1]),
                                                        targets, 
                                                        keep_prob,   
                                                        text_length,
                                                        summary_length,
                                                        max_summary_length,
                                                        len(vocab_to_int)+1,
                                                        rnn_size, 
                                                        num_layers, 
                                                        vocab_to_int,
                                                        batch_size)
        
        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_logits[0].rnn_output, 'logits')
        inference_logits = tf.identity(inference_logits[0].sample_id, name='predictions')
        
        # Create the weights for sequence_loss, the sould be all True across since each batch is padded
        masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
    print("Graph is built.")
    graph_location = "./graph"
    #print(graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(train_graph)

    learning_rate_decay = 0.95
    min_learning_rate = 0.0005
    display_step = 20 # Check training loss after every 20 batches
    stop_early = 0 
    stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
    per_epoch = 3 # Make 3 update checks per epoch
    update_check = (len(x)//batch_size//per_epoch)-1

    update_loss = 0 
    batch_loss = 0
    summary_update_loss = [] # Record the update losses for saving improvements in the model

    checkpoint = "./best_model.ckpt" 
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        # If we want to continue training a previous session
        #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
        #loader.restore(sess, checkpoint)
        
        for epoch_i in range(1, epochs+1):
            update_loss = 0
            batch_loss = 0
            for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                    get_batches(y, x, batch_size,vocab_to_int)):
                start_time = time.time()
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: texts_batch,
                    targets: summaries_batch,
                    lr: learning_rate,
                    summary_length: summaries_lengths,
                    text_length: texts_lengths,
                    keep_prob: keep_probability})

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                        .format(epoch_i,
                                epochs, 
                                batch_i, 
                                len(x) // batch_size, 
                                batch_loss / display_step, 
                                batch_time*display_step))
                    batch_loss = 0

                if batch_i % update_check == 0 and batch_i > 0:
                    print("Average loss for this update:", round(update_loss/update_check,3))
                    summary_update_loss.append(update_loss)
                    
                    # If the update loss is at a new minimum, save the model
                    if update_loss <= min(summary_update_loss):
                        print('New Record!') 
                        stop_early = 0
                        saver = tf.train.Saver() 
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break
                    update_loss = 0
                
                        
            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            
            if stop_early == stop:
                print("Stopping Training.")
                break
