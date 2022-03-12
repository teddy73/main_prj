#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset 
import os

def create_features_for_all_trainset(main_data, graph_data,network_of_following):
    current_directory = os.getcwd()
    os.mkdir(current_directory+'/features/')
    main_data=main_data
    graph_data=graph_data
    network_of_following=network_of_following
    #sort the graph dataframe base on user name
    graph_data =graph_data.sort_values('user').reset_index(drop=True)
    #delete the duplicate users in graph data and save in  the new dataframe 
    graph_data_drop_duplicate_users = graph_data.drop_duplicates(
        subset ="user", keep = "first", inplace = False)
    graph_data_drop_duplicate_users = \
        graph_data_drop_duplicate_users.reset_index(drop=True)
    #obtain list of out degree for each user
    graph_data_drop_duplicate_users['out_deg'] = \
        list(dict(network_of_following.out_degree(
            list(graph_data_drop_duplicate_users['user']))).values())
    #obtain list of out degree for each user
    graph_data_drop_duplicate_users['in_deg'] = \
        list(dict(network_of_following.in_degree(
            list(graph_data_drop_duplicate_users['user']))).values())
    #make copy of garph dataframe and dataframe of unique users    
    copy_of_data = graph_data_drop_duplicate_users.copy(deep=True)
    copy_of_graph =graph_data.copy(deep=True)
    #create features for different trainset based on time  
    #like the number of retweet and reply for users and their neighbors
    trainset_time = main_data[main_data['date']<='2012-07-01 12:00:00'][
        'date'].max()
    loop_for_all_different_trainsize = [i for i in range(1,24)]
    for i in loop_for_all_different_trainsize:
        #make new trainset based on next time 
        trainset = main_data[main_data['date']<=trainset_time]  
        #reset both dataframe to set new featurs based on new trainset
        graph_data_drop_duplicate_users = copy_of_data.copy(deep=True)
        graph_data = copy_of_graph.copy(deep=True)
        #create numberofretweet and numberofreply feature based on trainset 
        dataframe_of_users_with_interaction_retweet = trainset[
            trainset['interaction_type']=='RT']
        dataframe_of_users_with_interaction_reply = trainset[
            trainset['interaction_type']=='RE']
        list_of_user_with_interaction_retweet = list(
            dataframe_of_users_with_interaction_retweet['user'].value_counts(
                sort=False).keys())
        number_of_retweet = list(
            dataframe_of_users_with_interaction_retweet['user'].value_counts(
                sort=False))
        temporary_dataframe = pd.DataFrame(
            {'user':list_of_user_with_interaction_retweet,
             'number_of_RT_train':number_of_retweet})   
        list_of_user_with_interaction_reply = list(
            dataframe_of_users_with_interaction_reply['user'].value_counts(
                sort=False).keys())
        number_of_reply = list(
            dataframe_of_users_with_interaction_reply['user'].value_counts(
                sort=False))
        temporary_dataframe_two=pd.DataFrame(
            {'user':list_of_user_with_interaction_reply,
             'number_of_RE_train':number_of_reply})
        temporary_dataframe = temporary_dataframe.merge(
            temporary_dataframe_two, how='outer', on='user')
        #join the results for each node in main dataframe
        graph_data_drop_duplicate_users = \
            graph_data_drop_duplicate_users.set_index('user').join(
            temporary_dataframe.set_index('user'), how='left') 
        graph_data_drop_duplicate_users['number_of_RT_train'] = \
            graph_data_drop_duplicate_users['number_of_RT_train'].fillna(0)
        graph_data_drop_duplicate_users['number_of_RE_train'] = \
            graph_data_drop_duplicate_users['number_of_RE_train'].fillna(0)    
        #create numberofretweet and numberofreply of user' neighbors feature based on trainset 
        graph_data = graph_data.set_index('target').join(
            temporary_dataframe.set_index('user'), how='left') 
        graph_data['number_of_RE_train'] = graph_data['number_of_RE_train'].fillna(0)
        graph_data['number_of_RT_train'] = graph_data['number_of_RT_train'].fillna(0)
        number_of_RT_neighbor = list(
            graph_data.groupby('user').sum()['number_of_RT_train'])
        user =list(graph_data.groupby('user').sum()[
            'number_of_RT_train'].keys())
        temporary_dataframe_two = pd.DataFrame(
            {'user':user,
             'RT_neigh_train':number_of_RT_neighbor}) 
        number_of_RE_neighbor = list(
            graph_data.groupby('user').sum()['number_of_RE_train'])
        user =list(graph_data.groupby('user').sum()[
            'number_of_RE_train'].keys())
        temporary_dataframe = pd.DataFrame(
            {'user':user,
             'RE_neigh_train':number_of_RE_neighbor})         
        temporary_dataframe = temporary_dataframe.merge(
            temporary_dataframe_two, how='outer', on='user')
        graph_data_drop_duplicate_users = graph_data_drop_duplicate_users.join(
            temporary_dataframe.set_index('user'), how='left')
        graph_data_drop_duplicate_users.reset_index(inplace=True)
        #convert the columns of dataframe
        graph_data_drop_duplicate_users=graph_data_drop_duplicate_users.astype(
            {'in_deg': 'int16',
             'out_deg': 'int16',
             'number_of_RT_train': 'int16',
             'number_of_RE_train': 'int16',
             'RT_neigh_train': 'int16',
             'RE_neigh_train': 'int16'})  
        #save the features based on time in trainset
        list_of_all_users_in_graph = list(graph_data_drop_duplicate_users['user'])
        x_train=graph_data_drop_duplicate_users[
            ['in_deg',
             'out_deg',
             'number_of_RT_train',
             'number_of_RE_train',
             'RT_neigh_train',
             'RE_neigh_train']].to_numpy()
        np.save(current_directory+'/features/user%s'%(i),
                list_of_all_users_in_graph)
        np.save(current_directory+'/features/x_train%s'%(i),x_train)
        trainset_time = trainset_time + DateOffset(hours=6)   