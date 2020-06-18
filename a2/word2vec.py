#!/usr/bin/env python

import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax

from collections import Counter


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE
    return s


def naiveSoftmaxLossAndGradient(
        v_c,
        outsideWordIdx,
        U,
        dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    y_hat = softmax(np.dot(U, v_c))
    loss = -np.log(y_hat[outsideWordIdx])
    y_hat[outsideWordIdx] -= 1  # y_hat - y, y is a one hot vec with y_i == 1 if i == o.
    y_diff = y_hat  # rename variable to denote we are working with y_hat - y now.
    gradCenterVec = np.dot(y_diff, U)  # See answer to (b)
    # It's a bit hard to see why we need to reshape(-1, 1) and reshape(1, -1).
    # Reason is that y_diff and v_c are in the shapes (word_size,) and (word_vec_length,)
    # If we build the dot product we get a singular value, but as described in the solution to (c)
    # We multiply each probability element-wise with v_c
    #
    # >>> v_c = np.array([[1, 2, 3]]) # (original form would be np.array([1, 2, 3]))
    # >>> ydiff = np.array([[0.1], [0.2], [0.7]]) # (original form would be np.array([0.1, 0.2, 0.7]))
    # >>> np.dot(ydiff, v_c) # now we can get the gradient for each of the outside vectors in matrix U
    # array([[0.1, 0.2, 0.3],
    #        [0.2, 0.4, 0.6],
    #        [0.7, 1.4, 2.1]])
    gradOutsideVecs = np.dot(y_diff.reshape(-1, 1), v_c.reshape(1, -1))
    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
        centerWordVec,
        outsideWordIdx,
        outsideVectors,
        dataset,
        K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)
    # we will be able to express it as one sum if we compute the scores (the vector product inside the sigmoid)
    # for u_oT v_c and - u_w Tv_c
    sampledOutsideVectors = outsideVectors[indices]
    scores = -np.dot(sampledOutsideVectors, centerWordVec)
    scores[0] *= -1
    scores = sigmoid(scores)
    loss = -np.sum(np.log(scores))
    # as shown in the written answer, the derivatives of the objective wrt. vc uses 1 - <the score computed above>
    scores = 1 - scores
    # in case we derive wrt. to the outside word vector u_o, the derivative has a negative sign.
    scores[0] *= -1
    # from now on we are computing sums like (1-sigma(-u_oTv_c))*u_k
    # simply compute the sum by calculating the dot product of the matrix with a singular first dimension.
    grad_center_vec = np.dot(scores.reshape(1, -1), sampledOutsideVectors)
    # gradient for a single vector, expected to return flat shape
    grad_center_vec = grad_center_vec.reshape(-1)
    outside_vector_grads = np.zeros_like(outsideVectors)  # shape convention
    # here we multiply each vector entry of scores as a scalar with the outside vector, giving us the gradient.
    outside_grad_repeated = np.dot(scores.reshape(-1, 1), centerWordVec.reshape(1, -1))
    # we have duplicates in the outside_grads, that we need to just add together.
    # the numpy.add.at function does exactly that when you give it duplicate indices.
    np.add.at(
        outside_vector_grads,
        indices,
        outside_grad_repeated,
    )
    ### Please use your implementation of sigmoid in here.

    ### END YOUR CODE

    return loss, grad_center_vec, outside_vector_grads


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0

    ### YOUR CODE HERE (~8 Lines)
    center_word_indx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[center_word_indx]
    loss_and_gradients = [word2vecLossAndGradient(centerWordVec, word2Ind[word], outsideVectors,
                                                  dataset) for word in outsideWords]
    losses = [loss for loss, grad_center_vec, grad_outside_vec in loss_and_gradients]
    loss = sum(losses) # easy
    center_grads = np.array([grad_center_vec for loss, grad_center_vec, grad_outside_vec in loss_and_gradients])
    outside_grads = np.array([grad_outside_vec for loss, grad_center_vec, grad_outside_vec in loss_and_gradients])
    grad_center_vec = np.sum(center_grads, axis=0)
    # As stated in written assignment, deriving wrt to v_k where k!=c results in 0 derivative
    gradCenterVecs = np.zeros_like(centerWordVectors)
    gradCenterVecs[center_word_indx] = grad_center_vec
    # Outside vectors are just the sum of gradients.
    gradOutsideVectors = np.sum(outside_grads, axis=0)
    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N / 2), :]
    outsideVectors = wordVectors[int(N / 2):, :]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N / 2), :] += gin / batchsize
        grad[int(N / 2):, :] += gout / batchsize

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in range(2 * C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
                    dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
                    dummy_vectors, "negSamplingLossAndGradient Gradient")

    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)


if __name__ == "__main__":
    test_word2vec()
