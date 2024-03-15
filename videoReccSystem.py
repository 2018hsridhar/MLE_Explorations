'''
Creating mock template code for a video recommendation system
Will copy-paste code for later and share via e-mails

'''

'''
Challenges addressed in this problem :
1. Identify ML category for <user,video> pairings and assess given this pairing, the likeliehood a user will watch the video.
2. Set up the TTNN model
3.
4.
5.

'''

import pandas as pd
 
data =[['P','Q','S'], ['T',10], [0, 1]]

print("Work on empty dataframe")
empty_frame = pd.DataFrame()
print(empty_frame)
print(empty_frame.shape)

print("Work on non-empty dataframe")

data_frame = pd.DataFrame(data)

print("Shape of the data frame ( data dimension checking ):")
print(data_frame.shape)
print()
print("Raw Data:")
print(data_frame)
