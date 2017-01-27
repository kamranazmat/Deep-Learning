"""Softmax."""
import math
#scores = [3.0, 1.0, 2.0]
scores = [[1, 2, 3, 6], [2, 4, 5, 6], [3, 8, 7, 6]]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    
    scores = np.array(x) # list of LOGITS
    number_of_classes = len(scores) # calculate number of samples to be examined
    exp_sums = []
    probability_list = []
    
    """
    if scores list is 1D then then softmax in 1D and when the scores list is 2D, the softmax will
    be in 2D. So, 2D list need to handled differently
    """
    
    if scores.ndim != 1:
        number_of_samples = len(scores[0])
        for i in range(0, number_of_samples):
            exp_sum = 0
            for class_scores in scores:
                exp_sum += math.exp(class_scores[i])
            exp_sums.append(exp_sum)
            
        for class_scores in scores:
            temp_list = []
            for i in range(0, number_of_samples):
                temp_list.append(math.exp(class_scores[i])/exp_sums[i])
            probability_list.append(temp_list)
            
    else:
        exp_sum = 0
        for class_score in scores:
            exp_sum += math.exp(class_score)
        for class_score in scores:
            probability_list.append(math.exp(class_score)/exp_sum)
    
    return np.array(probability_list)
    

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()