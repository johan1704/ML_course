#Importing the important libraires
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#Importing the dataset
dataset=pd.read_csv("Ads_CTR_Optimisation.csv")

#Implementing UCB
import math
N=10000
d=10
ads_selected=[]
number_of_selection=[0]*d
sum_of_reward=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if (number_of_selection[i]>0):    
           average_reward=sum_of_reward[i]/number_of_selection[i]
           delta_i=math.sqrt(3/2*math.log(n+1)/number_of_selection[i])
           upper_bound=average_reward+delta_i
        else:
            upper_bound=1e400
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)    
    number_of_selection[ad]=number_of_selection[ad]+1
    reward=dataset.values[n,ad]
    sum_of_reward[ad]=sum_of_reward[ad]+reward
    total_reward=total_reward+reward
    
#Visualizing the results
plt.hist(ads_selected)
plt.title("Histogram of ad selection")
plt.xlabel("ads")
plt.ylabel("number of times each ads was selected")
plt.show()    
    
    
    
             
        
        
    