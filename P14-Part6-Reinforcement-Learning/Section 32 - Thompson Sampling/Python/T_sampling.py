#Thompson smapling

#Importing the important libraires
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#Importing the dataset
dataset=pd.read_csv("Ads_CTR_Optimisation.csv")

#Implementing Thompson sampling
import random
N=10000
d=10
ads_selected=[]
number_of_reward_1=[0]*d
number_of_reward_0=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(number_of_reward_1[i]+1,number_of_reward_0[i]+1)
        if(random_beta>max_random):
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        number_of_reward_1[ad]=number_of_reward_1[ad]+1
    else:
        number_of_reward_1[ad]=number_of_reward_1[ad]+1
    total_rewrd=total_reward+reward
        
#Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of ad selection")
plt.xlabel("ads")
plt.ylabel("number of times each ads was selected")
plt.show()    
    
    
    
             
        
        
    