#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CNN(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, filters, kernel_size, e_char, m_word):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.e_char = e_char
        self.m_word = m_word

    def forward(self, x_reshaped):
        x_conv = nn.Conv1d(in_channels=self.e_char, out_channels=self.filters, kernel_size=self.kernel_size,
                           padding=1).forward(x_reshaped)
        return nn.MaxPool1d(self.m_word - self.kernel_size + 1 + 2).forward(x_conv)
    ### END YOUR CODE
