#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:10:28 2022

@author: macbook
"""

import numpy as np
import pandas as pd
import networkx as nx
import csv
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset 
#create dataframe 
def create_dataframe_and_graph_data(data_address,graph_address):
    data_address=data_address
    graph_address=graph_address
    #read dataset
    main_data=pd.read_csv(data_address,delim_whitespace=True, header=None)
    #define column name for data
    main_data.columns=['user', 'target_user', 'timestamp', 'interaction_type']
    main_data=main_data.drop(0)
    #convert 'target_user' and 'user' column to string
    main_data['target_user']=(main_data['target_user'].astype(str))
    main_data['user']=(main_data['user'].astype(str))
    #formate the column date
    main_data['date']=(main_data['timestamp'].astype(int)).apply(datetime.fromtimestamp)
    main_data = main_data.drop(['timestamp'], axis='columns')
    #sort dataset based on the date column
    main_data=main_data.sort_values('date').reset_index(drop=True)
    #filter rows by interaction type with retweet and reply
    main_data = main_data.loc[(main_data['interaction_type']=='RT')| 
                              (main_data['interaction_type']=='RE')]
    main_data['date'] = main_data['date'].dt.round('H')
    main_data = main_data.sort_values('date').reset_index(drop=True)
    #read following graph
    read_the_edglist = open(graph_address, 'r')
    convert_edglist_to_csv = csv.reader(read_the_edglist)  
    list_of_thenetwork_of_following = [
        tuple(map(str,row[0].split())) for row in convert_edglist_to_csv]
    read_the_edglist.close()
    list_of_thenetwork_of_following_seperate_user_and_target = \
        [nodeID1+' '+nodeID for nodeID1, nodeID in list_of_thenetwork_of_following]
    network_of_following = nx.parse_edgelist(
        list_of_thenetwork_of_following_seperate_user_and_target, 
        nodetype=str,create_using=nx.DiGraph())
    list_of_users_in_network=\
        [nodeID1 for nodeID1, nodeID in list_of_thenetwork_of_following]
    list_of_neighbor_in_network=\
        [nodeID for nodeID1, nodeID in list_of_thenetwork_of_following]
    #create dataframe based on the users and who are following    
    graph_data = pd.DataFrame({'user':list_of_users_in_network,
                               'target':list_of_neighbor_in_network})
    print(graph_data['user'])
    print(main_data['date'])
    return main_data,graph_data,network_of_following

def add_feature_in_dataset(main_data,graph_data,network_of_following):
    #sort the graph dataframe base on user name
    graph_data =graph_data.sort_values('user').reset_index(drop=True)
    print(graph_data['user'])
    #define two new columns for graph data
    graph_data['active_neighbor']=0
    graph_data['active_user']=0
    #change the type of columns in graph dataframe
    graph_data=graph_data.astype({'active_neighbor': 'int8','active_user': 'int8'})
    #delete the duplicate users in graph data and save in  the new dataframe 
    graph_data_drop_duplicate_users = graph_data.drop_duplicates(
        subset ="user", keep = "first", inplace = False)
    graph_data_drop_duplicate_users = \
        graph_data_drop_duplicate_users.reset_index(drop=True)
    #obtain list of out degree for each user
    graph_data_drop_duplicate_users['out_deg'] = \
        list(dict(network_of_following.out_degree(
            list(graph_data_drop_duplicate_users['user']))).values())
    print(graph_data_drop_duplicate_users['out_deg'])
    #define new columns 
    graph_data_drop_duplicate_users['influence_weight']= 0.0
    graph_data_drop_duplicate_users['outcome']= 0
    graph_data_drop_duplicate_users['login_time']=main_data['date'].min()
    graph_data_drop_duplicate_users['counter']=0
    #change the type of columns in graph dataframe
    graph_data_drop_duplicate_users=graph_data_drop_duplicate_users.astype({
        'out_deg': 'int16','outcome': 'int8','counter':'int8'})
    return graph_data,graph_data_drop_duplicate_users

