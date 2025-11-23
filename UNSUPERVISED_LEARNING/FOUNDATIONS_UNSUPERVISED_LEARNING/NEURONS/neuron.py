'''
Project background ( neuron.py ) : 
Emualte classic examples of single neuron binary classification
The AND and OR gate examples for a singel binary classification neuron.
Implement mini-batch gradient descent for training.
Single neuron models are linear models.

NN Notes :
- operate on non-linear data.
- it is supervised learning
- specializations needed on Perceptron?
- (sigmoid(sum(inputs*weights + bias)) = output)
- sigmoid(weight*feature + bias ) = prediction

No regularization ( huh )?
Use sigmoid activation func.
Why to use mini-batch gradient descent?

summation = sum_{i=1}^{k}w_i*f_i + w_b*bias
sigma : non-linear func 
    sigma(summation) => output
( single bias, multiple features )

wTest scoping :
a. does the model generalize well to non-training data?
b.
c.
d.
e.

Resources :
a. https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a

'''
import numpy as np


class Neuron:
    # Don't change anything in the `__init__` function.
    def __init__(self, examples):
        np.random.seed(42)
        # Three weights: one for each feature and one more for the bias.
        self.weights = np.random.normal(0, 1, 3 + 1)
        self.examples = examples
        self.train()

    # Don't use regularization.
    # Use mini-batch gradient descent.
    # Use the sigmoid activation function.
    # Use the defaults for the function arguments.
    # Log loss as the loss function

    # Ahh shit a single neuron has to execute learning on the feature set too
    # Training should adjust our weights
    # Gradient(of loss function) -> gradient on an error ( ohh ) 


    # A compromise between computing the true gradient and the gradient at a single sample
    # is to compute the gradient against more than one training sample (called a "mini-batch") at each step.
    # This can perform significantly better than "true" stochastic gradient descent described,
    # because the code can make use of vectorization libraries rather than computing each
    # step separately as was first shown in [6] where it was called
    # "the bunch-mode back-propagation algorithm". It may also result in smoother convergence, as
    # the gradient computed at each step is averaged over more training samples.
    # Is mini-batch even the norm for NNs?
    # Batch size of 10 : kept small for faster convergence
    # randomize samples or not?

    # Multiple epochs in training ( iterate over each epoch ) ; data fed in multiple times.
    # Self-feed data into NN multiple times.
    # time to epoch completion = ( data_set_size / batch_size )
        
    def train(self, learning_rate=0.01, batch_size=10, epochs=200):
        # Set up sigmoid functionality
        # [1] step one : compute sigmoid(transpose(weight)*feature) and losses for each example
        lossVals = []
        datasetSize = len(self.examples)
        for epoch in range(len(epochs)):
            for(batchPtr in range(0, datasetSize, batch_size)):
                # [1] Compute gradient on the batch size
                

                # [2] update all examples ( in the current batch ) accordingly
                y = example[-1]
                curFeature = example[0:len(example) - 1:1]
                neuronSum = 0
                for i in len(curFeature):
                    neuronSum += curFeature[i] * self.weights[i]
                biasIndex = 3
                neuronSum += self.weights[biasIndex]
                yHat = self.sigmoid(neuronSum)
                curLoss = self.lossFunc(yHat,y)
                lossVec.append(curLoss)
            
        
            
        # [3] update trainable params - the weights - with loss values
        # for lossW in self.weights
            # self.weights += learning_rate * lossW
        
    # Negative log loss
    # Woah loss akin to that of logistic regression
    def lossFunc(self,inputVal):
        


    # Initially, sigmoid and nn models similar ( with sigmoid )
    # Solve for the exponential
    # Does this really reduce training loss?
    def sigmoid(self,neuronSum):
        denom = 1 + np.exp(-1 * neuronSum)
        sigmoid = 1 / denom
        return sigmoid

    # Return the probability—not the corresponding 0 or 1 label.
    # input has label of <1> ( single label based probability )
    def predict(self, features):
        groundLabelProb = 0.0
        
        
        return groundLabelProb
