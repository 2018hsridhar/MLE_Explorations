'''
RecordLinkage.py

Case Study : M&A Firm -> buyout(C1) by C2
Goal : Find overlapping subscribers to both publicly-traded companies.

Three types of matches ( on our dataframes ):

overlap : union of users ( union of indices ) type of reasoning
    -> split into three seperate tables
    
40 minutes YET getting warmer!
call it a night -> almost getting there now!



'''
import numpy as np
import pandas as pd



def to_lowercase(column):
    return column.str.lower()

def link_records(df_youtube, df_spotify):
    df_youtube['UID'] = range(0, len(df_youtube.index))
    origYoutubeDF = df_youtube
    debugCols =['non_mfa_ip_addresses','first_three_octets_ip']
    
    colsToExplode = ['first_three_octets_ip']
    # gaaah .explode() not in-place method
    # new index instead?
    df_youtube = df_youtube.explode(colsToExplode,ignore_index=True)
    df_spotify = df_spotify.explode(colsToExplode,ignore_index=True)
    # print((df_youtube[debugCols]).head(10))
    # print((df_spotify[debugCols]).head(10))

    condOneList = ['preferred_contact']
    matchedCondOne = pd.merge(df_youtube,df_spotify,how='inner',left_on=condOneList,right_on=condOneList)
    

    condTwoList = ['first_name','last_name','last_four_digits','billing_zip_code']
    matchedCondTwo = pd.merge(df_youtube,df_spotify,how='inner',left_on=condTwoList,right_on=condTwoList)

    # for cond three -> get the other condition of matches
    # naive : explode non_mfa_ip_addresses -> new col : first-three-octets
    # explosion = more rows ( more data ) but so be it
    # can dim reduce, but do later place
    
    # Write your code here ( ANY octet match )
    condThreeList = ['first_name','last_name','billing_zip_code','first_three_octets_ip']
    matchedCondThree = pd.merge(df_youtube,df_spotify,how='inner',left_on=condThreeList,right_on=condThreeList)

    # union of all indices thing
    # .unique() is a numnpy ndarray :-O !!!
    oneUID = matchedCondOne['UID']
    twoUID = matchedCondTwo['UID']
    threeUID = matchedCondThree['UID']
    allUID = pd.concat([oneUID,twoUID,threeUID])
    allUIDUniq = allUID.unique()
    sorted = allUIDUniq
    sorted.sort()

    # can we merge both ( of take their union instead )?
    # hmmmm? we care only about those in the youtube pandas dataframe ( not both frames )
    cohortGroup = origYoutubeDF.loc[origYoutubeDF['UID'].isin(allUIDUniq)]
    return cohortGroup
