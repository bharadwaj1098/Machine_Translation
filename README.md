# Machine_Translation
Translation of Japanese to English language. 

## Project Topic: Neural Machine Translation

## A. Project Team members

1. Sai Bharadwaj Reddy Arrabelly - 801166672
2. Monesa Thoguluva Janardhanan - 801167556
3. Prashanth Minkuri - 801166901

## B. Project Topic Introduction
We are planning to implement a Neural Machine Translation using the Transformer Model for
translation of Japanese sentences to English. As of January 2020, Transformers are the dominant
architecture in NLP and are used to achieve state-of-the-art results for many tasks and it appears
as if they will be for the near future.
Recurrent neural networks are very slow to train and without LSTM the model is not very
accurate. But with LSTM, the model makes it much slower to train.We made the Seq2Seq model
as our baseline model but the performance was very low. To enhance the performance we are
using the Transformer model for our project.Attention is all you need: paper deals with English
to German and English to French and have shown the BLEU score of 28.1 and 41.0 respectively.
For Japanese to English we have implemented the state of art model and achieved a BLEU score
of 41.0.


# Approaches

We used Spacy for Tokenizing the Data. Our Implementation is being Done is Pytorch due to which TorchText Fields were also used in preprocessing. Torch Text fields make it easy to make Dataloaders of the data for the models to train on.
First we implemented a Seq2seq which uses Encoder - Decoder. Both Encoder and Decoder use RNN. Encoder encodes the input sentence into a single vector, Decoder takes in that vector and outputs the sentence. After Viewing the Results we moved on to Implement the Transformer Model. Our Approach is to Implement the Transformer Architecture which is introduced in the paper
“Attention Is All You Need”. As the title says the Transformer Model uses Attention which is a mechanism of which looks at parts of input and decides at each step on which part of the input is important. Transformer architecture also has Encoder and Decoder, but “Multi Head Attention” is Used at Encoder and “Multi-Headed Attention” , “Masked Multi-headed Attention” are used in Decoder.
For encoder as the paper suggests first the tokens are passed to an Input embedding layer. Next a Positional Embedding layer is used to help our model for the sequence by injecting some information about the relative or absolute position of the tokens. Next we have used Multihead attention means many attention vectors will be created for each word and the Wz weight will choose which attention vector to take. (Multiple attention vector for one word) And the rest of the things in the model are normal like Feed Forward Neural Network and Normalization. We are using the BLEU metric in our model as this is the basic metric which provides quick and
quality assessment of our translation. It measures direct word-to-word similarity that provides first hand analysis of the quality of our translation. BLEU has frequently been reported as correlating well with human judgement. Another metric we are using our project is Perplexity. 
In natural language processing, perplexity is a way of evaluating language models. A language model is a probability distribution over entire sentences or texts.


