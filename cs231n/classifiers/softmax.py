import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  train_num, D = X.shape
  num_classes = W.shape[1]
  for i in range(train_num):
    # normalize those bitches
    scores = X[i].dot(W)
    C = - np.max(scores)
    scores = scores + C
    # find the correct score 
    correct_class = y[i]
    correct_score = scores[correct_class]
    exp_sum = np.sum(np.exp(scores))
    # caculate loss function for each example
    loss_single_exmp = -correct_score + np.log(exp_sum)
    dW[:, correct_class] -= X[i]
    for j in range(num_classes):
        dW[:, j] += (np.exp(scores[j]) / exp_sum) * X[i]
    ## print("for example ", i, "The loss is: ", loss_single_exmp)
    loss += loss_single_exmp
  dW /= train_num
  dW += 2 * reg * W  
  loss /= train_num
  loss += reg * np.sum(W ** 2)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  train_num, dimension = X.shape
  classes_num = W.shape[1]
  scores = X.dot(W)
  scores = scores - np.max(scores, axis=1).reshape(train_num, 1)

  correct_scores = scores[np.arange(train_num), y]

  exp_scores = np.exp(scores)
  exp_scores_sum = np.sum(exp_scores, axis=1)
    
  losses = -correct_scores + np.log(exp_scores_sum)
  loss = np.sum(losses) / train_num
  loss += reg * np.sum(W ** 2)
  
  temp = np.zeros_like(scores)
  temp[np.arange(train_num), y] = -1
  dW = X.T.dot(temp)
  factors = exp_scores / exp_scores_sum.reshape(train_num, 1)
  print('The factors size is ', factors.shape)
  dW += X.T.dot(factors)
  dW /= train_num
  dW += 2 * reg * W  
  
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

