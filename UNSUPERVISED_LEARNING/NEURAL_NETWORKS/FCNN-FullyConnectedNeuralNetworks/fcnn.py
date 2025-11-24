'''
TBH, CNNs build atop FCNNs : Fully Connected Neural Networks
What are good minimal examples to start with FCNNs? ( imagine we remove conv2d, maxpool, and flatten layers in sequential )?
'''

'''
XOR Problem : Simplest Non-Linearly Separable Data
Input : 2 features (0 or 1)
 Just 4 data points, requires 1 hidden layer
Dense(2) → Dense(2, relu) → Dense(1, sigmoid)

Classic examle :
- One hidden layer with non-linear activation (ReLU) needed
since output is not linearly separable from inputs
case in point : (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
To say that a dividing line can seperate 0s and 1s is impossible
because no straight line can separate these points in 2D space

'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

