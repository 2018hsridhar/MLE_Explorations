'''
No regularization ( huh )?
Use sigmoid activation func.
Why to use mini-batch gradient descent?

summation = sum_{i=1}^{k}w_i*f_i + w_b*bias
sigma : non-linear func 
    sigma(summation) => output
( single bias, multiple features )



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
    def train(self, learning_rate=0.01, batch_size=10, epochs=200):

        # Set up sigmoid functionality
        lossVals = []
        for example in self.examples
            y = example[-1]
            curFeature = example[0:len(example) - 1:1]
            neuronSum = 0
            for i in len(curFeature):
                neuronSum += curFeature[i] * self.weights[i]
            yHat = self.sigmoid(neuronSum)
            curLoss = self.lossFunc(yHat,y)
            lossVec.append(curLoss)

        # NN stuff is well-predicated on logistic regression ( master this ) 

    # Negative log loss
    # Woah loss akin to that of logistic regression
    def lossFunc(self,inputVal):
        


    # Initially, sigmoid and nn models similar ( with sigmoid )
    def sigmoid(self,neuronSum):
        # bias : a fixed constant?
        sigmoid = func(neuronSum) + bias
        return sigmoid
        

    # Return the probability—not the corresponding 0 or 1 label.
    def predict(self, features):
        groundLabelProb = 0.0
        return groundLabelProb
