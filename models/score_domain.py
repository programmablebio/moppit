from scipy.stats import norm
import numpy as np
import os


def score(outputs):
    
    weight = 0.03
    binding_size_dist = np.load(os.path.join(os.path.dirname(__file__), "../params/binding_size_train_dist.npy"))


    mean = np.mean(binding_size_dist)

    std = np.std(binding_size_dist)

    dist  = norm(mean, std)
    
    
    max_score = 0
    
    
    
    scores = np.exp(outputs[0])[:, 1]
    
    indices = np.argsort(-1*scores)
                         
    for j in range(1, len(indices)):
        
        
        
        score = (1-weight)*np.mean(scores[indices[:j]]) +  weight*(dist.pdf(j/len(indices)))
        
        
        if score > max_score:
            
            max_score = score
            
            
    return max_score
            