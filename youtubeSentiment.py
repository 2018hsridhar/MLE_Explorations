'''
Youtube Sentiment:

Best type of content to create -> based on highest customer sentiment
For advertisers

sort by descending ( mean_sentiment )
sentiment = #-likes/(#-likes + #-dislikes)
Exec by category 

Dataframes capture statsitics too.

... oohhh -> it's the mean sentiment ( per category ) -> mean ( over each like/like+dislike) row-wise. Nevermind
'''
import pandas as pd
import numpy as np

def get_youtube_sentiment(video_stats_df):
    video_stats_df['sentiment'] = video_stats_df['likes'] / (video_stats_df['likes'] + video_stats_df['dislikes'])
    video_stats_df = video_stats_df.groupby(by=['category_id']).agg(mean_sentiment=('sentiment','mean'))
    video_stats_df.sort_values(by=['mean_sentiment'],ascending=[False],inplace=True)
    return video_stats_df
