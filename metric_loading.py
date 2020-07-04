
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

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
from datetime import datetime
import matplotlib.pyplot as plt

import fonctions
import itertools
from tqdm.notebook import tqdm

from os import listdir
from os.path import isfile, join

import random
random.seed(0)

###############################################################################################
########################## CONTENT TASTE:
###############################################################################################

def get_user_content_taste():
    
    global_taste_content_user = {}
    print('present_user_content_taste')

    for chunk_id in range(12):
        print(chunk_id)

        with gzip.open('/home/maxime/Desktop/RecSys2020/trends/taste_content_user_{}.pkl.gz'.format(chunk_id), 'rb') as f:
            taste_content_user = pkl.load(f)

        
        select = {k:v for k,v in taste_content_user.items() if k in global_taste_content_user.keys() }
        reste = {k:v for k,v in taste_content_user.items() if k not in global_taste_content_user.keys() }
        
        { update_taste_content_user(global_taste_content_user, k, v) for k,v in select.items() }     
        global_taste_content_user.update(reste)
                
    print('global_taste_content_user formating')

    return global_taste_content_user

def update_taste_content_user(multimedias, k, v):
    multimedias[k]= np.add( multimedias[k],v)

###############################################################################################
########################## TWEET VIEWS:
###############################################################################################

def get_tweet_views(init):
    
    tweet_views = {}
    print('tweet_views aggregation')

    for chunk_id in range(8):
        print(chunk_id)
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/tweet_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                view = pkl.load(f)
        else:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_tweet_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                view = pkl.load(f)

        
        select = {k:v for k,v in view.items() if k in tweet_views.keys() }
        reste = {k:v for k,v in view.items() if k not in tweet_views.keys() }
        
        tweet_views.update(reste)
        { update_views(tweet_views, k, v) for k,v in tqdm(select.items()) }     

        
    print('tweet_views formating')

    return tweet_views

def update_views(tweet_views, k, v):
    tweet_views[k] = tweet_views[k] + v

###############################################################################################
########################## USER MULTIMEDIA:
###############################################################################################

def get_user_multimedia(init):
    
    global_user_multimedia = {}
    print('present_user_multimedia')

    for chunk_id in range(8):
        print(chunk_id)
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/multimedia_user_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                user_multimedia = pkl.load(f)
        else:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_multimedia_user_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                user_multimedia = pkl.load(f)

        
        select = {k:v for k,v in user_multimedia.items() if k in global_user_multimedia.keys() }
        reste = {k:v for k,v in user_multimedia.items() if k not in global_user_multimedia.keys() }
        
        { update_user_multimedia(global_user_multimedia, k, v) for k,v in select.items() }     
        global_user_multimedia.update(reste)
        
    print('present_user multimedia formating')

    return global_user_multimedia

def update_user_multimedia(multimedias, k, v):
    multimedias[k]=[ val+v[idx] for idx,val in enumerate( multimedias[k] ) ]


###############################################################################################
########################## AUTHOR MULTIMEDIA:
###############################################################################################


def get_author_multimedia(init):
    
    global_author_multimedia = {}
    print('present_user_multimedia')

    for chunk_id in range(8):
        print(chunk_id)
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/multimedia_author_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                author_multimedia = pkl.load(f)
        else:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_multimedia_author_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                author_multimedia = pkl.load(f)

        
        select = {k:v for k,v in author_multimedia.items() if k in global_author_multimedia.keys() }
        reste = {k:v for k,v in author_multimedia.items() if k not in global_author_multimedia.keys() }
        
        { update_author_multimedia(global_author_multimedia, k, v) for k,v in select.items() }     
        global_author_multimedia.update(reste)
        
    print('present_author multimedia formating')

    return global_author_multimedia

def update_author_multimedia(multimedias, k, v):
    multimedias[k]=[ val+v[idx] for idx,val in enumerate( multimedias[k] ) ]


###############################################################################################
########################## LANGUAGES:
###############################################################################################


def user_langue_update_agg(user_langue, k, v):
    user_langue[k]=user_langue[k] + v

def get_global_user_langue(init):
    
    global_user_langue = {}
    nb = 16 if init==True else 8
    print(nb)
    
    print('agregating user langue')
    for chunk_id in range(nb):
        print(chunk_id)
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/user_langue_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                user_langue = collections.Counter( pkl.load(f) )
        else:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_user_langue_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                user_langue = collections.Counter( pkl.load(f) )

        
        select = {k:v for k,v in user_langue.items() if k in global_user_langue.keys() }
        reste = {k:v for k,v in user_langue.items() if k not in global_user_langue.keys() }
        { user_langue_update_agg(global_user_langue, k, v) for k,v in select.items() }     
        global_user_langue.update(reste)
        user_langue={}

    print('formating user presence')
    global_user_langue = { k:user_langue_extraction(v)   for k,v in tqdm(global_user_langue.items()) }
 
    return global_user_langue

def user_langue_extraction(v):
    
    langues = v.most_common(2)
    
    if len(langues)>1:
        langue1 = langues[0][0]
        langue2 = langues[1][0]
        prop_langue1 = round (langues[0][1] / ( langues[0][1] + langues[1][1]),3)
        prop_langue2 = round( langues[1][1] / ( langues[0][1] + langues[1][1]),3)
    else:
        langue1 = langues[0][0]
        langue2 = None
        prop_langue1 = 1
        prop_langue2 = 0
    
    return [prop_langue1, prop_langue2,langue1, langue2 ]

###############################################################################################
########################## HASHTAGS:
###############################################################################################

def get_global_hashtag_presence(init):
    
    global_hashtag_presence = {}
    print('hashtag_presence aggregation')

    for chunk_id in range(8):
        print(chunk_id)
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/hashtag_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                hashtag_presence = pkl.load(f)
        else:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_hashtag_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                hashtag_presence = pkl.load(f)

        
        select = {k:v for k,v in hashtag_presence.items() if k in global_hashtag_presence.keys() }
        reste = {k:v for k,v in hashtag_presence.items() if k not in global_hashtag_presence.keys() }
        { update_hashtags(global_hashtag_presence, k, v) for k,v in select.items() }     
        global_hashtag_presence.update(reste)
        
    print('hashtag_presence formating')
    global_hashtag_presence = {k:formatage_d(v) for k,v in tqdm( global_hashtag_presence.items() ) }
    
    return global_hashtag_presence

def update_hashtags(domains, k, v):
    domains[k].extend( v )

def formatage_d(v):
    v = set(v)
    res = {'total_week': len(v),
           'total_hour': collections.Counter( [ ref[1] for ref in v ] ) }
    return res
    


###############################################################################################
########################## DOMAINS:
###############################################################################################

def get_global_present_domains_presence(init):
    
    global_present_domains_presence = {}
    print('present_domains_presence aggregation')

    for chunk_id in range(8):
        print(chunk_id)
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/present_domains_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                present_domains_presence = pkl.load(f)
        else:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_present_domains_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                present_domains_presence = pkl.load(f)

        
        select = {k:v for k,v in present_domains_presence.items() if k in global_present_domains_presence.keys() }
        reste = {k:v for k,v in present_domains_presence.items() if k not in global_present_domains_presence.keys() }
        { update_domains(global_present_domains_presence, k, v) for k,v in select.items() }     
        global_present_domains_presence.update(reste)

    print('present_domains_presence formating')
    global_present_domains_presence = { k:formatage_h(v)   for k,v in tqdm(global_present_domains_presence.items()) }
    
    return global_present_domains_presence

def update_domains(domains, k, v):
    domains[k].extend( v )

def formatage_h(v):
    v = set(v)
    res = {'total_week': len(v),
           'total_hour': collections.Counter( [ ref[1] for ref in v ] ) }
    return res
    

###############################################################################################
########################## LINKS:
###############################################################################################

def sorting_l(v):
    unique = set(v)
    res = sorted(unique, key=lambda x: datetime.strptime(str(x[0]), '%Y-%m-%d %H:%M:%S') )
    return res

def formatage_l(v):
    v= sorting_l(v)
    res= {'author_id':v[0][2], 
     'tweet_id':v[0][1],
     'total_week':len( set(v) ),
     'total_hour':collections.Counter( [ ref[0][:13] for ref in set(v) ] ),
     'chain':{ x[1]: (datetime.strptime(str(x[0]), '%Y-%m-%d %H:%M:%S')-datetime.strptime(str(v[0][0]), '%Y-%m-%d %H:%M:%S') ).total_seconds() for idx, x in enumerate(v) } }
    return res

def get_global_present_links_presence(init):
    
    global_present_links_presence = {}
    print('present_links_presence aggregation')

    for chunk_id in range(8):
        print(chunk_id)
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/present_links_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                present_links_presence = pkl.load(f)
        else:
                
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_present_links_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                present_links_presence = pkl.load(f)
        
        select = {k:v for k,v in present_links_presence.items() if k in global_present_links_presence.keys() }
        reste = {k:v for k,v in present_links_presence.items() if k not in global_present_links_presence.keys() }
        
        { update_links(global_present_links_presence, k, v) for k,v in select.items() }     
        global_present_links_presence.update(reste)
        
    print('present_links_presence formating')
    global_present_links_presence = {k:formatage_l(v) for k,v in tqdm(global_present_links_presence.items())}

    return global_present_links_presence

def update_links(links, k, v):
    links[k].extend( v )

    
###############################################################################################
########################## LIKED USER TASTES:
###############################################################################################


def update_taste_agg(global_tastes, k, v):
    global_tastes[k]= global_tastes[k]+v 
    

def get_liked_author_tastes():
    global_author_tastes = {}

    for chunk_id in range(8):
        print(chunk_id)
    
        with gzip.open('/home/maxime/Desktop/RecSys2020/trends/liked_author_tastes_{}.pkl.gz'.format(chunk_id), 'rb') as f:
            author_tastes =  pkl.load(f)
        
        ### authors
        common_id =[ k for k in author_tastes.keys() if k in global_author_tastes.keys() ]
        else_id = [ k for k in author_tastes.keys() if k not in global_author_tastes.keys() ]  
    
        select = {k:author_tastes[k] for k in common_id }
        reste = {k:author_tastes[k] for k in else_id }
    
        global_author_tastes.update(reste)
        { update_taste_agg(global_author_tastes, k, v) for k,v in tqdm(select.items())  }
    
        print(len(global_author_tastes.keys() ))
        
    return global_author_tastes


def get_liked_hashtag_tastes():
    
    global_hashtag_tastes = {}
    for chunk_id in range(8):
        print(chunk_id)
    
        with gzip.open('/home/maxime/Desktop/RecSys2020/trends/liked_hashtag_tastes_{}.pkl.gz'.format(chunk_id), 'rb') as f:
            hashtag_tastes =  pkl.load(f)
        
        ### hashtag
        common_id =[ k for k in hashtag_tastes.keys() if k in global_hashtag_tastes.keys() ]
        else_id = [ k for k in hashtag_tastes.keys() if k not in global_hashtag_tastes.keys() ]  
    
        select = {k:hashtag_tastes[k] for k in common_id }
        reste = {k:hashtag_tastes[k] for k in else_id }
    
        global_hashtag_tastes.update(reste)
        { update_taste_agg(global_hashtag_tastes, k, v) for k,v in tqdm(select.items())  }
    
        print(len(global_hashtag_tastes.keys() ))
        
    return global_hashtag_tastes

##############################################################################################
########################## USER TASTES:
###############################################################################################


def update_taste_agg(global_tastes, k, v):
    global_tastes[k]= {  column: global_tastes[k][column]+v[column] for column in global_tastes[k].keys() }
    

def get_author_tastes():
    global_author_tastes = {}

    for chunk_id in range(8):
        print(chunk_id)
    
        with gzip.open('/home/maxime/Desktop/RecSys2020/trends/author_tastes_{}.pkl.gz'.format(chunk_id), 'rb') as f:
            author_tastes =  pkl.load(f)
        
        ### authors
        common_id =[ k for k in author_tastes.keys() if k in global_author_tastes.keys() ]
        else_id = [ k for k in author_tastes.keys() if k not in global_author_tastes.keys() ]  
    
        select = {k:author_tastes[k] for k in common_id }
        reste = {k:author_tastes[k] for k in else_id }
    
        global_author_tastes.update(reste)
        { update_taste_agg(global_author_tastes, k, v) for k,v in tqdm(select.items())  }
    
        print(len(global_author_tastes.keys() ))
        
    return global_author_tastes


def get_hashtag_tastes():
    
    global_hashtag_tastes = {}
    for chunk_id in range(8):
        print(chunk_id)
    
        with gzip.open('/home/maxime/Desktop/RecSys2020/trends/hashtag_tastes_{}.pkl.gz'.format(chunk_id), 'rb') as f:
            hashtag_tastes =  pkl.load(f)
        
        ### hashtag
        common_id =[ k for k in hashtag_tastes.keys() if k in global_hashtag_tastes.keys() ]
        else_id = [ k for k in hashtag_tastes.keys() if k not in global_hashtag_tastes.keys() ]  
    
        select = {k:hashtag_tastes[k] for k in common_id }
        reste = {k:hashtag_tastes[k] for k in else_id }
    
        global_hashtag_tastes.update(reste)
        { update_taste_agg(global_hashtag_tastes, k, v) for k,v in tqdm(select.items())  }
    
        print(len(global_hashtag_tastes.keys() ))
        
    return global_hashtag_tastes

###############################################################################################
########################## GLOBAL AUTHOR:
###############################################################################################

def get_global_author_presence(init):
    
    engagements = ['like_timestamp','retweet_timestamp','retweet_with_comment_timestamp','reply_timestamp']
    global_author_presence = {}
    print('author presence aggregation')

    for chunk_id in range(8):
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/author_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                author_presence = collections.Counter( pkl.load(f) )
        else:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_presence_metric_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                author_presence = collections.Counter( pkl.load(f) )
        
        select = {k:v for k,v in author_presence.items() if k in global_author_presence.keys() }
        reste = {k:v for k,v in author_presence.items() if k not in global_author_presence.keys() }
        { author_presence_update_agg(global_author_presence, k, v) for k,v in select.items() }     
        global_author_presence.update(reste)
        author_presence={}
        
    print('author presence formating')
    global_author_presence = { k:author_presence_extraction(v)   for k,v in tqdm(global_author_presence.items()) }
    
    return global_author_presence

def author_presence_extraction(v):

    total_presence = sum( v ) 
    daily_mean = round( total_presence / len( v ), 3)
    daily_std = round( np.std( v ), 3)

    return [total_presence, daily_mean, daily_std ]

def author_presence_update_agg(author_presence, k, v):
    author_presence[k]= [x + y for x, y in zip(author_presence[k], v )]
    
###############################################################################################
########################## GLOBAL USER 2:
###############################################################################################

def get_global_user_ratio2():
    print('agregating user ratio')
    
    
    engagements = ['like_timestamp','retweet_timestamp','retweet_with_comment_timestamp','reply_timestamp']
    global_user_ratio = {}

    for chunk_id in range(16):
        print(chunk_id)
        with gzip.open('/home/maxime/Desktop/RecSys2020/trends/2user_ratio_{}.pkl.gz'.format(chunk_id), 'rb') as f:
            user_ratio = pkl.load(f) 
                
        select = {k:v for k,v in user_ratio.items() if k in global_user_ratio.keys() }
        reste = {k:v for k,v in user_ratio.items() if k not in global_user_ratio.keys() }
        { user_ratio_update(global_user_ratio, k, v) for k,v in select.items() }     
        global_user_ratio.update(reste)

    print('formating user ratio')
    global_user_ratio = { k:np.round( np.divide(v,[sum(v)]*12),3)   for k,v in tqdm(global_user_ratio.items()) if sum( [ v[0] ]  + list(v[2:]) ) > 7 }
    
    return global_user_ratio

def user_ratio_update(multimedias, k, v):
    multimedias[k] = np.add(multimedias[k], v)
    
###############################################################################################
########################## GLOBAL USER:
###############################################################################################

def user_presence_update_agg(user_presence, k, v):
    user_presence[k]=[x + y for x, y in zip(user_presence[k], v )]

def get_global_user_presence(init):
    
    engagements = ['like_timestamp','retweet_timestamp','retweet_with_comment_timestamp','reply_timestamp']
    global_user_presence = {}
    print('agregating user presence')
    for chunk_id in range(8):
        
        
        if init:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/user_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                user_presence = collections.Counter( pkl.load(f) )
        else:
            with gzip.open('/home/maxime/Desktop/RecSys2020/trends/update1_user_presence_{}.pkl.gz'.format(chunk_id), 'rb') as f:
                user_presence = collections.Counter( pkl.load(f) )
        
        select = {k:v for k,v in user_presence.items() if k in global_user_presence.keys() }
        reste = {k:v for k,v in user_presence.items() if k not in global_user_presence.keys() }
        { user_presence_update_agg(global_user_presence, k, v) for k,v in select.items() }     
        global_user_presence.update(reste)
        user_presence={}

    print('formating user presence')
    global_user_presence = { k:sum( v ) for k,v in tqdm(global_user_presence.items()) }
 
    return global_user_presence


