#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):

    # Remember to delete the above 'pass' after your implementation

    def __init__(self, w1_init, b1_init, w2_init, b2_init):
        super().__init__()
        self.w1_init = w1_init
        self.b1_init = b1_init
        self.w2_init = w2_init
        self.b2_init = b2_init

    ### YOUR CODE HERE for part 1f

    def forward(self, x_conv_out):
        # Map convolution outputs of shape (batch_size, embedding_size) to x_highway with the same shape
        e_word_size = x_conv_out.shape[1]
        w_proj = nn.Linear(e_word_size, e_word_size)
        self.w1_init(w_proj.weight)
        self.b1_init(w_proj.bias)
        x_proj = nn.Sequential(w_proj, nn.ReLU()).forward(x_conv_out)
        w_gate = nn.Linear(e_word_size, e_word_size)
        self.w2_init(w_gate.weight)
        self.b2_init(w_gate.bias)
        x_gate = nn.Sequential(w_gate, nn.Sigmoid()).forward(x_conv_out)
        x_highway = torch.mul(x_proj, x_gate) + torch.mul((1 - x_gate), x_conv_out)
        return x_highway
    ### END YOUR CODE
