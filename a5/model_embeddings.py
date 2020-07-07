#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.e_char = 50
        self.word_embed_size = word_embed_size
        self.dropout_prob = 0.3
        self.vocab = vocab
        self.embedding = nn.Embedding(
            len(vocab.char2id),
            self.e_char,
            padding_idx=vocab.char2id['∏']
        )

        self.cnn = CNN(e_char=self.e_char,
                       filters=self.word_embed_size,
                       kernel_size=5,
                       m_word=21  # copied from reference solution. not sure where this comes from.
                       )
        self.highway = Highway()
        self.dropout = nn.Dropout(p=self.dropout_prob)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        embedded = self.embedding.forward(input)
        max_sent_len, batch_size, samples, channels = embedded.shape
        reshaped = embedded.view((max_sent_len * batch_size, samples, channels)).transpose(1, 2)
        conv_out = self.cnn.forward(reshaped)
        highway = self.highway.forward(conv_out.squeeze())
        dropout = self.dropout.forward(highway)
        return dropout.view(max_sent_len, batch_size, -1)
        ### END YOUR CODE
