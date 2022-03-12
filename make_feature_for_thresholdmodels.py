#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import networkx as nx
import csv
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset 
import math
import os
from set_login import *
def create_inputs_for_thresholdmodel(main_data,
                                     graph_data,
                                     graph_data_drop_duplicate_users,
                                     effective_time,
                                     login_type,
                                     login_time,
                                     k,
                                     b):
    #get the working directory and make new folder named accoriding to effective time
    code_address = os.getcwd()
    os.mkdir(code_address+'/%s/'%(effective_time))
    #create effective time column 
    main_data['effective_time'] = np.array(pd.DataFrame(np.array(
        main_data['date']+DateOffset(hours=effective_time)))[0])
    #set the min time and max time for trainset
    max_time_for_trainset = main_data[
        main_data['date']<='2012-07-07 00:00:00']['date'].max()
    min_time_for_trainset = main_data['date'].min()
    #define a new dataframe for saving influence and outcome of users
    train_dataframe_for_influence = pd.DataFrame()
    trainset_time = main_data[
        main_data['date']<='2012-07-01 12:00:00']['date'].max()
    counter_for_different_trainset_size=1
    #reset the influential neighbor for each user 
    graph_data['active_neighbor']=0
    graph_data_drop_duplicate_users['active_user']=0
    graph_data_drop_duplicate_users['active_neighbor']=0
    #we want to compute the influence for different trainsize
    while min_time_for_trainset <= max_time_for_trainset: 
        window_min = min_time_for_trainset 
        window_max = min_time_for_trainset + DateOffset(hours=1) 
        #update the login time for users who are not become active at all
        mask = (graph_data_drop_duplicate_users['counter']==0)
        graph_data_drop_duplicate_users.loc[
                mask, 'login_time'] = window_min
        #obtain the active node at the moment
        dataframe_ofactive_nodes = main_data.query(
                'date>=@window_min & date<@window_max')
        #get the list of active node
        list_of_active_nodes = list(set(list(
                dataframe_ofactive_nodes['user'])+list(
                    dataframe_ofactive_nodes['target_user'])))
        if(dataframe_ofactive_nodes.shape[0]!=0):
            #obtain list of nodes who are effective at the moment
            list_of_influential_nodes =\
                list(main_data.loc[
                    (main_data['effective_time']>=window_min) & 
                    (main_data['date']<window_min),'user'])+list(
                        main_data.loc[(main_data['effective_time']>=window_min)&
                                      (main_data['date']<window_min),'target_user'])
            list_of_influential_nodes =list(set(
                    list_of_influential_nodes))
            #reset the influential neighbor for each user 
            graph_data['active_neighbor']=0
            graph_data_drop_duplicate_users['active_user']=0
            #set the influential neighbor from previous time
            graph_data.loc[graph_data['target'].isin(
                    list_of_influential_nodes),'active_neighbor']=1
            #identify nodes are active now
            graph_data_drop_duplicate_users.loc[
                    graph_data_drop_duplicate_users['user'].isin(
                        list_of_active_nodes),'active_user']=1
            #create dataframe for nodes who are active from 
            #graph_data_drop_duplicate_users table
            dataframe_ofactive_nodes = graph_data_drop_duplicate_users[
                    graph_data_drop_duplicate_users['user'].isin(
                        list_of_active_nodes)]
            #create dataframe for users and their count of influential neighbors
            list_count_ofinfluentailnodes_foreachuser = \
                list(graph_data.groupby('user').sum()['active_neighbor'])
            list_ofusers =list(graph_data.groupby('user').sum()[
                    'active_neighbor'].keys())
            temporary_dataframe = pd.DataFrame({'user':list_ofusers,
                              'sum_active_neighbor':
                                  list_count_ofinfluentailnodes_foreachuser})
            #merge active node dataframe and count of influenctial neighbor     
            dataframe_ofactive_nodes = dataframe_ofactive_nodes.merge(
                temporary_dataframe, how='left', on='user')
            #compute the influence weight for active nodes        
            out_degree_of_users_are_active_now =\
                1/np.array(dataframe_ofactive_nodes['out_deg'])
            number_of_active_neighbor_for_each_node = np.array(
                    dataframe_ofactive_nodes['sum_active_neighbor'])
            dataframe_ofactive_nodes['influence_weight']  =\
                    number_of_active_neighbor_for_each_node * out_degree_of_users_are_active_now 
            dataframe_ofactive_nodes['outcome'] = 1
            #add dataframe to main dataframe
            train_dataframe_for_influence = pd.concat([
                    train_dataframe_for_influence,
                    dataframe_ofactive_nodes],ignore_index=True) 
            #update login time for active nodes
            graph_data_drop_duplicate_users.loc[
                    graph_data_drop_duplicate_users['user'].isin(
                        list_of_active_nodes),'login_time']=window_min
            graph_data_drop_duplicate_users.loc[
                    graph_data_drop_duplicate_users['user'].isin(
                        list_of_active_nodes),'counter']=1
            #obtain list of nodes that not become active to get their influence
            list_of_nodes_that_not_become_active_atall = list(
                    graph_data_drop_duplicate_users.loc[
                        (graph_data_drop_duplicate_users['active_user']==0)&
                        (graph_data_drop_duplicate_users['active_neighbor']==0),'user'])
            #obtain list of nodes that are not active now to get their influence
            list_of_inactive_now=list(
                    graph_data_drop_duplicate_users.loc[
                        (graph_data_drop_duplicate_users['active_user']==0)&
                        (graph_data_drop_duplicate_users['active_neighbor']==1),'user'])
            #obtain list of nodes that are not online
            temporary_list=list(
                    graph_data_drop_duplicate_users.loc[
                        (graph_data_drop_duplicate_users['user'].isin(
                            list_of_nodes_that_not_become_active_atall))&
                        (graph_data_drop_duplicate_users['login_time']!= window_min),
                        'user'])
            #delete users who are not online then we compute the influence
            list_of_nodes_that_not_become_active_atall= list(
                    set(list_of_nodes_that_not_become_active_atall)-set(
                        temporary_list))
            temporary_list=list(graph_data_drop_duplicate_users.loc[
                    (graph_data_drop_duplicate_users['user'].isin(
                        list_of_inactive_now))&
                    (graph_data_drop_duplicate_users['login_time']!= window_min),
                    'user'])
            list_of_inactive_now= list(
                    set(list_of_inactive_now)-set(temporary_list))
            #
            if(len(list_of_nodes_that_not_become_active_atall)
                   >dataframe_ofactive_nodes.shape[0]):
                #consider inactive nodes equal to active nodes
                sample_inactive = \
                       np.array(list_of_nodes_that_not_become_active_atall
                                )[:dataframe_ofactive_nodes.shape[0]]
                graph_data_drop_duplicate_users.loc[
                       graph_data_drop_duplicate_users['user'].isin(
                           sample_inactive),'active_neighbor']=1
            else:
                sample_inactive = list_of_nodes_that_not_become_active_atall
                graph_data_drop_duplicate_users.loc[
                       graph_data_drop_duplicate_users['user'].isin(
                           sample_inactive),'active_neighbor']=1
            sample_inactive=list(sample_inactive)
            sample_inactive.extend(list_of_inactive_now)  
            dataframe_ofinactive_nodes = graph_data_drop_duplicate_users[
                    graph_data_drop_duplicate_users['user'].isin(sample_inactive)]
            #compute the influential weight for inactive nodes
            dataframe_ofinactive_nodes = \
                dataframe_ofinactive_nodes.merge(temporary_dataframe, how='left', on='user')
            out_degree_of_users_are_inactive_now =\
                1/np.array(dataframe_ofinactive_nodes['out_deg'])
            number_of_active_neighbor_for_each_node = np.array(
                    dataframe_ofinactive_nodes['sum_active_neighbor'])
            dataframe_ofinactive_nodes['influence_weight'] =\
                    number_of_active_neighbor_for_each_node*out_degree_of_users_are_inactive_now 
            dataframe_ofinactive_nodes['outcome'] =0
            #add inactive dataframe to main dataframe
            train_dataframe_for_influence = pd.concat(
                    [train_dataframe_for_influence,dataframe_ofinactive_nodes]
                    ,ignore_index=True)  
            #update the logintime
            graph_data_drop_duplicate_users = set_login_time(
                graph_data_drop_duplicate_users, login_type, 
                login_time, k, b, window_min, 'featureformodels')
            
        if min_time_for_trainset != trainset_time:
            min_time_for_trainset = min_time_for_trainset + DateOffset(hours=1) 
        else:    
            #we want to save features that we made for specific trainsize
            trainset_time = trainset_time + DateOffset(hours=6) 
            #filter dataframe based on nodes who are active
            temporary_dataframe = train_dataframe_for_influence.loc[
                (train_dataframe_for_influence['outcome']==1)]
            #obtain the min influence_weight for nodes with outcome==1
            list_influenctialweight = list(
                temporary_dataframe.groupby('user').min()['influence_weight'])
            user =list(
                temporary_dataframe.groupby('user').sum()['influence_weight'].keys())            
            temporary_dataframe = temporary_dataframe.drop_duplicates(
                subset ="user",keep = "first", inplace = False)
            temporary_dataframe_two = pd.DataFrame(
                {'user':user,'influ':list_influenctialweight})
            temporary_dataframe = temporary_dataframe.merge(
                temporary_dataframe_two, how='left', on='user')
            #filter dataframe based on nodes who are not active
            temporary_dataframe_two = train_dataframe_for_influence.loc[
                (train_dataframe_for_influence['outcome']==0)]
            #obtain the min influence_weight for nodes with outcome==1
            list_influenctialweight = list(
                temporary_dataframe_two.groupby('user').max()['influence_weight'])
            user =list(
                temporary_dataframe_two.groupby('user').sum()['influence_weight'].keys())
            temporary_dataframe_two = temporary_dataframe_two.drop_duplicates(
                subset ="user",keep = "first", inplace = False)
            temporary_dataframe_one =pd.DataFrame(
                {'user':user,'influ':list_influenctialweight})
            temporary_dataframe_two = temporary_dataframe_two.merge(
                temporary_dataframe_one, how='left', on='user')
            if temporary_dataframe.shape[0]<temporary_dataframe_two.shape[0]:
               temporary_dataframe_two = \
                   temporary_dataframe_two[:temporary_dataframe.shape[0]]
               temporary_dataframe_one= pd.DataFrame()
               temporary_dataframe_one=pd.concat(
                   [temporary_dataframe, temporary_dataframe_two]).sort_index()
               temporary_dataframe_one=temporary_dataframe_one.reset_index()
            else:
               temporary_dataframe = \
                   temporary_dataframe[:temporary_dataframe_two.shape[0]]
               temporary_dataframe_one= pd.DataFrame()
               temporary_dataframe_one=pd.concat(
                   [temporary_dataframe, temporary_dataframe_two]).sort_index()
               temporary_dataframe_one=temporary_dataframe_one.reset_index()
            #set the varaible we want to save   
            t_train=temporary_dataframe_one['influ'].to_numpy()  
            user=temporary_dataframe_one['user'].to_numpy()
            outcome=temporary_dataframe_one['outcome'].to_numpy()
            counter=graph_data_drop_duplicate_users['counter']
            login_time= graph_data_drop_duplicate_users['login_time']
            #save the results
            np.save(code_address+'/%s/t_train%s' 
                    %(effective_time,counter_for_different_trainset_size),t_train)
            np.save(code_address+'/%s/user%s' 
                    %(effective_time,counter_for_different_trainset_size),user)
            np.save(code_address+'/%s/outcome%s' 
                    %(effective_time,counter_for_different_trainset_size),outcome)
            np.save(code_address+'/%s/counter%s' 
                    %(effective_time,counter_for_different_trainset_size),counter)
            np.save(code_address+'/%s/login_time%s' 
                   %(effective_time,counter_for_different_trainset_size),login_time)
            counter_for_different_trainset_size+=1
            
def make_input_for_models(effective_time):
    code_address = os.getcwd()
    os.mkdir(code_address+'/%s/'%(effective_time)+'/threshold/')
    loop=[i for i in range(1,24)]
    for i in loop:
       user=np.load(code_address+'/%s/user%s.npy' 
                    %(effective_time,i),allow_pickle=True)
       t_train=np.load(code_address+'/%s/t_train%s.npy'
                       %(effective_time,i),allow_pickle=True)
       y_train=np.load(code_address+'/%s/outcome%s.npy'
                       %(effective_time,i),allow_pickle=True)
       p=pd.DataFrame(np.load(code_address+'/features/x_train%s.npy'
                              %(i)))
       p.columns=['in_deg',
                  'out_deg',
                  'number_of_RT_train',
                  'number_of_RE_train',
                  'RT_neigh_train',
                  'RE_neigh_train']
       p['user']=np.load(code_address+'/features/user%s.npy'
                         %(i),allow_pickle=True)
       df=pd.DataFrame({'user':user})
       df = df.merge(p, how='left', on='user')   
       x_train=df[['in_deg',
                   'out_deg',
                   'number_of_RT_train',
                   'number_of_RE_train',
                   'RT_neigh_train',
                   'RE_neigh_train']].to_numpy()
       x_test=p[['in_deg',
                 'out_deg',
                 'number_of_RT_train',
                 'number_of_RE_train',
                 'RT_neigh_train',
                 'RE_neigh_train']].to_numpy()
       np.save(code_address+'/%s/threshold/y_train%s'
               %(effective_time,i),y_train)
       np.save(code_address+'/%s/threshold/t_train%s'
               %(effective_time,i),t_train)
       np.save(code_address+'/%s/threshold/x_train%s'
               %(effective_time,i),x_train)
       np.save(code_address+'/%s/threshold/x_test%s'
               %(effective_time,i),x_test) 