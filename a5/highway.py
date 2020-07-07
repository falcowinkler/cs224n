#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):

    # Remember to delete the above 'pass' after your implementation

    def __init__(self, e_word_size, w1_init=nn.init.uniform_, b1_init=nn.init.uniform_, w2_init=nn.init.uniform_,
                 b2_init=nn.init.uniform_):
        super().__init__()
        self.w1_init = w1_init
        self.b1_init = b1_init
        self.w2_init = w2_init
        self.b2_init = b2_init
        self.e_word_size = e_word_size
        self.w_proj = nn.Linear(self.e_word_size, self.e_word_size)
        self.w_gate = nn.Linear(self.e_word_size, self.e_word_size)

        self.w1_init(self.w_proj.weight)
        self.b1_init(self.w_proj.bias)
        self.w2_init(self.w_gate.weight)
        self.b2_init(self.w_gate.bias)

    ### YOUR CODE HERE for part 1f

    def forward(self, x_conv_out):
        # Map convolution outputs of shape (batch_size, embedding_size) to x_highway with the same shape
        # for some reason (during beam search ? ) we get inputs of shape (e_word_size) (without batch dim)
        # e_word_size = x_conv_out.shape[1] if len(x_conv_out.shape) > 1 else x_conv_out.shape[0]
        x_proj = nn.Sequential(self.w_proj, nn.ReLU()).forward(x_conv_out)
        x_gate = nn.Sequential(self.w_gate, nn.Sigmoid()).forward(x_conv_out)
        x_highway = torch.mul(x_proj, x_gate) + torch.mul((1 - x_gate), x_conv_out)
        return x_highway
    ### END YOUR CODE
