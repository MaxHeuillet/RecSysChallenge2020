import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import collections
import re
import pprint as pp
import numpy as np
import collections

import multiprocessing as mp
from multiprocessing.pool import ThreadPool

import math
import gzip
import pickle as pkl
import matplotlib.pyplot as plt

import fonctions
import itertools

from os import listdir
from os.path import isfile, join
from datetime import datetime

from tqdm.notebook import tqdm

import random
random.seed(0)

#####################################################################################
################## PHASE 1
#####################################################################################


def modify_chunk(directory, out_directory, chunk_id,chunk, user_presence, author_presence,val):
    
    for batch_file in tqdm(chunk):

        modify_batch(directory, batch_file, out_directory, user_presence, author_presence,val)
        
    return True

def modify_batch(directory, doc_name, out_directory, user_presence, author_presence, val=False):

    all_features = ["text_tokens", "hashtags", "tweet_id", 
                    "present_media", "present_links", 
                    "present_domains", "tweet_type","language", 
                    "tweet_timestamp", "engaged_with_user_id",
                    "engaged_with_user_follower_count", "engaged_with_user_following_count", 
                    "engaged_with_user_is_verified", "engaged_with_user_account_creation",
                    "engaging_user_id", "engaging_user_follower_count", 
                    "engaging_user_following_count", "engaging_user_is_verified",
                    "engaging_user_account_creation", "engagee_follows_engager"]
    
    df = pd.read_csv(directory+doc_name, encoding="utf-8", sep='\x01', header=None)
    
    if val:
        
        df.columns = all_features
        
    else:
        
        labels = ['reply_timestamp','retweet_timestamp', 'retweet_with_comment_timestamp','like_timestamp']
        df.columns = all_features+labels
        
        #### labels    
        df['reply_timestamp']=[ 0 if math.isnan(x) else 1 for x in df['reply_timestamp'] ]
        df['retweet_timestamp']=[ 0 if math.isnan(x) else 1 for x in df['retweet_timestamp'] ]
        df['retweet_with_comment_timestamp']=[ 0 if math.isnan(x) else 1 for x in df['retweet_with_comment_timestamp'] ]
        df['like_timestamp']=[ 0 if math.isnan(x) else 1 for x in df['like_timestamp'] ]
    
    #### publication date:
    df['tweet_utctimestamp']=[ datetime.utcfromtimestamp(int(date)) for date in df['tweet_timestamp'] ]
    df['tweet_timestamp']=[ datetime.fromtimestamp(int(date)) for date in df['tweet_timestamp'] ] 
    df['day']=[ date.weekday() for date in df['tweet_timestamp'] ]
    df['hour']=[ str(date)[10:13] for date in df['tweet_timestamp'] ]
    df = pd.get_dummies(df, columns=["hour"])
    df = pd.get_dummies(df,columns=["day"])
    
    #### user
    df['engaging_user_account_creation']=[ datetime.utcfromtimestamp(int(date)) for date in df['engaging_user_account_creation'] ]
    df['user_account_age']= (df['tweet_utctimestamp'] - df['engaging_user_account_creation']).dt.days
    df['engaging_user_is_verified']=[ 0 if x==False else 1 for x in df['engaging_user_is_verified'] ]
    
    batch_user_presence = { k:user_presence[k] for k in np.unique(df.engaging_user_id) }
    df['user_presence'] = [ batch_user_presence[k] for k in df.engaging_user_id ]

    #### author
    df['engaged_with_user_account_creation'] = [ datetime.utcfromtimestamp(int(date)) for date in df['engaged_with_user_account_creation'] ]
    df['author_account_age'] = (df['tweet_utctimestamp'] - df['engaged_with_user_account_creation']).dt.days
    df['engaged_with_user_is_verified']=[ 0 if x==False else 1 for x in df['engaged_with_user_is_verified'] ]
    df['engagee_follows_engager'] = [ 0 if x==False else 1 for x in df['engagee_follows_engager'] ]
    
    batch_author_presence = { k:author_presence[k] for k in np.unique(df.engaged_with_user_id) }
    df['author_presence'] = [ batch_author_presence[k][0]  for k in df.engaged_with_user_id ]
    df['mean_daily_author_presence'] = [ batch_author_presence[k][1]  for k in df.engaged_with_user_id ]
    df['std_daily_author_presence'] = [ batch_author_presence[k][2]  for k in df.engaged_with_user_id ]

    #### content
    df = pd.get_dummies(df, columns=["tweet_type"])
    
    df['photo'] = [ x.count('Photo') if type(x) is str else 0 for x in df['present_media'] ]
    df['video'] = [ x.count('Video') if type(x) is str else 0 for x in df['present_media'] ]
    df['gif'] = [ x.count('GIF') if type(x) is str else 0 for x in df['present_media'] ]
    
    df['nb_hashtags']=[len(re.split(r'\t+', x)) if type(x) is str else 0 for x in df['hashtags']]
    df['nb_links']=[len(re.split(r'\t+', x)) if type(x) is str else 0 for x in df['present_links']]
    df['nb_domains']=[len(re.split(r'\t+', x)) if type(x) is str else 0  for x in df['present_domains']]
    
    df.text_tokens = df.text_tokens.apply(lambda x : re.sub('\\t',' ', x) if (x == x) else None)
    df['interogation'] = [ re.split(r' ', x).count('136') for x in df['text_tokens'] ]
    df['exclamation'] = [ re.split(r' ', x).count('106') for x in df['text_tokens'] ]
    df['mention'] = [ re.split(r' ', x).count('137') for x in df['text_tokens'] ]
    W51H_pattern = '(14516|12489|12242|23525|14962|10479|12976|10841|10940|14796|24781|31237)\s(134491|12034|10301|10124)'
    df['W51H_question'] = [  len( re.findall(W51H_pattern, x) ) for x in df['text_tokens'] ]

    df.to_csv(out_directory+'pr_{}'.format(doc_name), header=df.columns, sep=',')

    return True

###################################################################################################
######################## PHASE 2
##################################################################################################

def modify_chunk_phase_2(directory, out_directory, chunk_id, chunk, user_ratio):#author_activity

    for batch_file in tqdm(chunk):
        modify_batch_phase2(directory, batch_file, out_directory, user_ratio)# author_activity
        
    return True

def modify_batch_phase2(directory, doc_name, out_directory, user_ratio): #author_activity

    df = pd.read_csv(directory+doc_name, encoding="utf-8", sep=',', header=0, index_col=False,low_memory=False)
    df = df.drop(df.columns[[0]], axis=1)

    ### authors
#     batch_author_ratio = { k:author_ratio.get(k, [ float('nan') ]*5 ) for k in np.unique(df.engaged_with_user_id) }
#     df['author_engagement'] = [ batch_author_ratio[k][0]  for k in df.engaged_with_user_id ]
#     df['author_ratio_like'] = [ batch_author_ratio[k][1]  for k in df.engaged_with_user_id ]
#     df['author_ratio_retweet'] = [ batch_author_ratio[k][2]  for k in df.engaged_with_user_id ]
#     df['author_ratio_rtc'] = [ batch_author_ratio[k][3]  for k in df.engaged_with_user_id ]
#     df['author_ratio_reply'] = [ batch_author_ratio[k][4]  for k in df.engaged_with_user_id ]
    
#     batch_author_activity = { k:author_activity[k]for k in np.unique(df.engaged_with_user_id) }
#     df['author_activity_week'] = [ batch_author_activity[k]['total_week'] for k in df.engaged_with_user_id]
#     df['author_activity_day'] = [ batch_author_activity[k]['total_day'][time[:10]] for k,time in zip(df.engaged_with_user_id,df.tweet_utctimestamp) ]
        
    ###users
    batch_user_ratio = { k:user_ratio.get(k, [ float('nan') ]*12 ) for k in np.unique(df.engaging_user_id) }
    df['user_0'] = [ batch_user_ratio[k][0]  for k in df.engaging_user_id ]
    df['user_1'] = [ batch_user_ratio[k][1]  for k in df.engaging_user_id ]
    df['user_2'] = [ batch_user_ratio[k][2]  for k in df.engaging_user_id ]
    df['user_3'] = [ batch_user_ratio[k][3]  for k in df.engaging_user_id ]
    df['user_4'] = [ batch_user_ratio[k][4]  for k in df.engaging_user_id ]
    df['user_5'] = [ batch_user_ratio[k][5]  for k in df.engaging_user_id ]
    df['user_6'] = [ batch_user_ratio[k][6]  for k in df.engaging_user_id ]
    df['user_7'] = [ batch_user_ratio[k][7]  for k in df.engaging_user_id ]
    df['user_8'] = [ batch_user_ratio[k][8]  for k in df.engaging_user_id ]
    df['user_9'] = [ batch_user_ratio[k][9]  for k in df.engaging_user_id ]
    df['user_10'] = [ batch_user_ratio[k][10]  for k in df.engaging_user_id ]
    df['user_11'] = [ batch_user_ratio[k][11]  for k in df.engaging_user_id ]
    
#     batch_user_taste = { k:user_tweet_taste.get(k, [ float('nan') ]*3 ) for k in np.unique(df.engaging_user_id) }
#     df['taste_quote'] = [ batch_user_taste[k][0]  for k in df.engaging_user_id ]
#     df['taste_retweet'] = [ batch_user_taste[k][1]  for k in df.engaging_user_id ]
#     df['taste_toplevel'] = [ batch_user_taste[k][2]  for k in df.engaging_user_id ]

    df.to_csv(out_directory+'{}'.format(doc_name), header=df.columns, sep=',')
    
    return True


###################################################################################################
######################## PHASE 3
##################################################################################################


def modify_batch3(out_directory,directory, doc_name, counters,  user_language,init): 
    
    count_bert_lk, count_bert_rt =  counters[0],counters[1]
    count_domains_lk, count_domains_rt = counters[2],counters[3]
    count_links_lk, count_links_rt = counters[4],counters[5]
    count_hashtags_lk, count_hashtags_rt = counters[6],counters[7]
    
    df = pd.read_csv(directory+doc_name, encoding="utf-8", sep=',', header=0, index_col=False,low_memory=False)
    df = df.drop(df.columns[[0]], axis=1)
    df.index = df.tweet_id+'-'+df.engaging_user_id
    
    substr = {k:user_language[k] for k in df.engaging_user_id }
    df['prop_langue1'] = [ substr[k][0]  for k in df.engaging_user_id ]
    df['prop_langue2'] = [ substr[k][1]  for k in df.engaging_user_id ]
    df['match_langue1'] = [ 1 if substr[k][2]==x1 else 0  for x1, k in zip(df.language,df.engaging_user_id) ]
    df['match_langue2'] = [ 1 if substr[k][3]==x2 else 0  for x2, k in zip(df.language,df.engaging_user_id) ]
    
    df['len_tweet'] = [ len( x.split(' ') ) for x in df.text_tokens ]
    
    #### hashtags
    prt_hashtags = [ re.sub('\t',' ', x) if (x == x) else ' ' for x in df['hashtags'] ]
    
    lk_hashtags = count_hashtags_lk.fit_transform( prt_hashtags )
    df['trend_lk_hashtags']=np.sum(lk_hashtags, axis=1)
    
    rt_hashtags = count_hashtags_rt.fit_transform( prt_hashtags )
    df['trend_rt_hashtags']=np.sum(rt_hashtags, axis=1)
    
    #### domains
    prt_domains = [ re.sub('\t',' ', x) if (x == x) else ' ' for x in df['present_domains'] ]
    
    lk_domains = count_domains_lk.fit_transform( prt_domains )
    df['trend_lk_present_domains']=np.sum(lk_domains, axis=1)
    
    rt_domains = count_domains_rt.fit_transform( prt_domains )
    df['trend_rt_present_domains']=np.sum(rt_domains, axis=1)
    
    #### links
    prt_links = [ re.sub('\t',' ', x) if (x == x) else ' ' for x in df['present_links'] ]
    
    lk_links = count_links_lk.fit_transform( prt_links )
    df['trend_lk_present_links']=np.sum(lk_links, axis=1)
    
    rt_links = count_links_rt.fit_transform( prt_links )
    df['trend_rt_present_links']=np.sum(rt_links, axis=1)
    
    #### bert
    lk_bert = count_bert_lk.fit_transform( df['text_tokens'] )
    df['trend_lk_text_tokens']=np.sum(lk_bert, axis=1)
    
    rt_bert = count_bert_rt.fit_transform( df['text_tokens'] )
    df['trend_rt_text_tokens']=np.sum( rt_bert, axis=1)
    
    if init:
        df.to_csv(out_directory+'{}'.format(doc_name), header=df.columns, sep=',')

    else:
        df.to_csv(out_directory+'{}'.format(doc_name), header=df.columns, sep=',')

    return True

###############################################################################
#################### PHASE 4
###############################################################################

def author_match(global_author_tastes, user, author, mode): 
    return global_author_tastes[user][mode].get(author,0)

def author_match2(global_author_tastes, user, author):
    return global_author_tastes[user].get(author,0)


def trend_match(global_trend_tastes, user, trend, mode):
    trend = trend.split(' ')  #ast.literal_eval(trend) 
    if len( trend  )==0:
        res=0   
    else:
        res = len( set(trend).intersection( set(global_trend_tastes[user][mode]) ))  
    return res

def trend_match2(global_lk_trends, user, trend):
    trend = trend.split(' ')  #ast.literal_eval(trend) #
    if len( trend )==0:
        res=0
    else:

        res = len( set(trend).intersection( set(global_lk_trends[user]) ) )     
    return res


def modify_batch2(out_directory,directory, doc_name,author_multimedia, user_multimedia,  global_lk_author,global_author_tastes,
                  global_lk_hashtags, global_hashtag_tastes, init):


    df = pd.read_csv(directory+doc_name, encoding="utf-8", sep=',', header=0, index_col=False,low_memory=False)
    df = df.drop(df.columns[[0]], axis=1)
    df.index = df.tweet_id+'-'+df.engaging_user_id

    substr1 = {k:global_author_tastes.get(k, {'rt_authors':{}, 'rtc_authors':{},'rpl_authors':{} } ) for k in df.engaging_user_id }
    df['taste_rt_author'] = [ author_match(substr1, user, author, 'rt_authors') for user, author in zip(df.engaging_user_id, df.engaged_with_user_id) ]
    df['taste_rtc_author'] = [ author_match(substr1, user, author, 'rtc_authors') for user, author in zip(df.engaging_user_id, df.engaged_with_user_id) ]
    df['taste_rpl_author'] = [ author_match(substr1, user, author, 'rpl_authors') for user, author in zip(df.engaging_user_id, df.engaged_with_user_id) ]

    substr2 = {k:global_lk_author.get(k, {}) for k in df.engaging_user_id }
    df['taste_lk_author'] = [ author_match2(substr2, user, author) for user, author in zip(df.engaging_user_id, df.engaged_with_user_id) ]

    prt_hashtags = [ re.sub('\t',' ', x) if (x == x) else ' ' for x in df['hashtags'] ]
    
    substr3 = {k:global_hashtag_tastes.get(k, {'rt_hashtags':{}, 'rtc_hashtags':{}, 'rpl_hashtags':{} } ) for k in df.engaging_user_id }
    df['taste_rt_hashtag'] = [ trend_match(substr3, user, hashtag, 'rt_hashtags') for user, hashtag in zip(df.engaging_user_id, prt_hashtags ) ]
    df['taste_rtc_hashtag'] = [ trend_match(substr3, user, hashtag, 'rtc_hashtags') for user, hashtag in zip(df.engaging_user_id, prt_hashtags) ]
    df['taste_rpl_hashtag'] = [ trend_match(substr3, user, hashtag, 'rpl_hashtags') for user, hashtag in zip(df.engaging_user_id, prt_hashtags) ]


    substr4 = {k:global_lk_hashtags.get(k, {}) for k in df.engaging_user_id }
    df['taste_lk_hashtag'] = [ trend_match2(substr4, user, hashtag) for user, hashtag in zip(df.engaging_user_id, prt_hashtags) ]

    # user proportion
    substr5 = {k:user_multimedia[k] for k in df.engaging_user_id }
    df['user_link_prop'] = [ substr5[x1][0]/x2 for x1,x2 in zip(df.engaging_user_id,df.user_presence)  ]
    df['user_photo_prop'] = [ substr5[x1][1]/x2 for x1,x2 in zip(df.engaging_user_id,df.user_presence)  ]
    df['user_video_prop'] = [ substr5[x1][2]/x2 for x1,x2 in zip(df.engaging_user_id,df.user_presence)  ]
    df['user_gif_prop'] = [ substr5[x1][3]/x2 for x1,x2 in zip(df.engaging_user_id,df.user_presence)  ]
    
    ### author proportion
    substr6 = {k:author_multimedia[k] for k in df.engaged_with_user_id }
    df['author_link_prop'] = [ substr6[x1][0]/x2 for x1,x2 in zip(df.engaged_with_user_id,df.author_presence)  ]
    df['author_photo_prop'] = [ substr6[x1][1]/x2 for x1,x2 in zip(df.engaged_with_user_id,df.author_presence)  ]
    df['author_video_prop'] = [ substr6[x1][2]/x2 for x1,x2 in zip(df.engaged_with_user_id,df.author_presence)  ]
    df['author_gif_prop'] = [ substr6[x1][3]/x2 for x1,x2  in zip(df.engaged_with_user_id,df.author_presence)  ]
    
    if init:
        df.to_csv(out_directory+'{}'.format(doc_name), header=df.columns, sep=',')

    else:
        df.to_csv(out_directory+'{}'.format(doc_name), header=df.columns, sep=',')
        
    return True

#########################################################
################### PHASE 5 
#######################################################

import ast 

def link_info(links, link, author, tweet, time):
    if len(link)==0:
        res = (0, 0, 0, 0, -1)
    else:
        request = [ links[li] for li in link]
        l_author = 1 if author in [ li['author_id']  for li in request] else 0
        l_tweet = 1 if tweet in [ li['tweet_id']  for li in request] else 0
        l_week = sum( [ li['total_week']  for li in request] )
        l_hour = sum( [ li['total_hour'][time]  for li in request] )
        l_age = min( [ li['chain'][tweet]  for li in request] )
        res = (l_author, l_tweet, l_week, l_hour, l_age)
    return res


def info_extraction(trend, ref, timestamp):
    if len(ref)==0:
        res = (0,0)
    else:
        week = sum( [ trend[el]['total_week'] for el in ref] )
        hour = sum( [ trend[el]['total_hour'][timestamp] for el in ref] )
        res = (week,hour)
    return res

def modify_batch4(out_directory, directory, doc_name,  taste_content,tweet_views,links,domains,hashtags, init): 
    df = pd.read_csv(directory+doc_name, encoding="utf-8", sep=',', header=0, index_col=False,low_memory=False)
    df = df.drop(df.columns[[0]], axis=1)
    df.index = df.tweet_id+'-'+df.engaging_user_id
    
    info_t = np.array( [ taste_content.get(k,[ float("nan") ]*4) for k in df.engaging_user_id ] )
    df['content_link'],df['content_text']  = info_t[:,0], info_t[:,1]
    df['content_photo_video'],df['content_gif'] = info_t[:,2], info_t[:,3]
    
    substr = { key:tweet_views[key] for key in df.tweet_id }
    df['tweet_presence']=[ substr[x] for x in df.tweet_id ]
    
    present_links = [ re.sub('\t',' ', x) if (x == x) else ' ' for x in df['present_links'] ]
    present_links = [ x.split(' ') if x!=' ' else []  for x in present_links ]
    substr1 = {k:links[k] for k in np.unique( list( itertools.chain.from_iterable(present_links) ) ) } 
    info_l = np.array( [ link_info(substr1,link,author,tweet,time[:13]) for link,author,tweet,time in zip(present_links,df.engaged_with_user_id,df.tweet_id,df.tweet_utctimestamp) ] ) 
    df['link_author'] = info_l[:,0] 
    df['link_tweet'] = info_l[:,1]
    df['link_week'] = info_l[:,2] 
    df['link_hour'] = info_l[:,3]
    df['link_age'] =  info_l[:,4]

    # x.split(' ') if x!=' ' else [] ast.literal_eval(x)
    present_hashtags = [ re.sub('\t',' ', x) if (x == x) else ' ' for x in df['hashtags'] ]
    present_hashtags = [ x.split(' ') if x!=' ' else [] for x in present_hashtags ]
    substr2 = {k:hashtags[k] for k in np.unique( list( itertools.chain.from_iterable(present_hashtags) ) ) } 
    info_h = np.array( [ info_extraction(substr2, ref,timestamp[:13]) for ref, timestamp in zip(present_hashtags,df.tweet_utctimestamp)])
    df['hashtag_week'] = info_h[:,0]
    df['hashtag_hour'] = info_h[:,1]  

    present_domains = [ re.sub('\t',' ', x) if (x == x) else ' ' for x in df['present_domains'] ]
    present_domains  = [ x.split(' ') if x!=' ' else [] for x in present_domains ]
    substr3 = {k:domains[k] for k in np.unique( list( itertools.chain.from_iterable(present_domains) ) ) } 
    info_d = np.array( [ info_extraction(substr3,ref,timestamp[:13]) for ref, timestamp in zip(present_domains,df.tweet_utctimestamp)] )
    df['domain_week'] = info_d[:,0] 
    df['domain_hour'] =  info_d[:,1] 
    
    if init:
        df.to_csv(out_directory+'{}'.format(doc_name), header=df.columns, sep=',')

    else:
        df.to_csv(out_directory+'{}'.format(doc_name), header=df.columns, sep=',')

    return True



