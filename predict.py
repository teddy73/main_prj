#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset 
import os
from set_login import set_login_time
def prediction(main_data, 
               graph_data,
               graph_data_drop_duplicate_users, 
               effective_time, 
               login_type,
               login_time,
               k,
               b):
    MainDataFrame=pd.melt(
        main_data,id_vars=['date'],value_vars=['user','target_user'],value_name='user')
    MainDataFrame=MainDataFrame.sort_values('date').reset_index(drop=True)
    list_of_time_in_trainset=[]
    list_of_bithreshold_error_for_all_trainsize=[]
    list_of_linearthreshold_error_for_all_trainsize=[] 
    #create new folder to save error results of models
    code_address = os.getcwd()
    os.mkdir(code_address+'/%s/'%(effective_time)+'/result/')
    trainset_time=main_data[main_data['date']<='2012-07-01 12:00:00'][
            'date'].max()
    loop = [i for i  in range(1,24)]
    #predict number of active node for all 23 trainsize based bi and linear threshold
    for i in loop:
        list_of_hours_compute_error=[] 
        list_of_active_nodes_indata = []
        list_of_activenodes_predicted_by_linearthreshold = []
        list_of_activenodes_predicted_by_bithreshold = []
        trainset = main_data[main_data['date']<=trainset_time]
        max_time_in_trainset = trainset['date'].max()
        #reset the login time and active neighbor for all trainsize
        graph_data['active_neighbor']=0
        graph_data_drop_duplicate_users['effective_time_for_bi_threshold']=''
        graph_data_drop_duplicate_users['effective_time_for_linear_threshold']=''
        graph_data_drop_duplicate_users['login_time_linearthreshold']= ''
        graph_data_drop_duplicate_users['login_time_bithreshold']= ''
        graph_data_drop_duplicate_users['counter_thre']=0
        graph_data_drop_duplicate_users['counter_bi']=0
        #load threshold, up and down threshold based on trainsize
        graph_data_drop_duplicate_users['threshold']= np.load(
                code_address+'/%s/threshold/threshold%s.npy' %(effective_time,i))
        graph_data_drop_duplicate_users['down_threshold']= np.load(
                code_address+'/%s/threshold/down_threshold%s.npy' %(effective_time,i))
        graph_data_drop_duplicate_users['up_threshold']= np.load(
                code_address+'/%s/threshold/up_threshold%s.npy' %(effective_time,i))   
        #load the login time of users from trainset
        graph_data_drop_duplicate_users['login_time_linearthreshold']=np.load(
                code_address+'/%s/login_time%s.npy' %(effective_time,i))
        graph_data_drop_duplicate_users['login_time_bithreshold']=np.load(
                code_address+'/%s/login_time%s.npy'%(effective_time,i))
        graph_data_drop_duplicate_users['counter_thre']=np.load(
                code_address+'/%s/counter%s.npy' %(effective_time,i))
        graph_data_drop_duplicate_users['counter_bi']=np.load(
                code_address+'/%s/counter%s.npy'%(effective_time,i))                
        # set the start and end time of prediction(24 hours prediction after last trainset time)
        max_time_in_prediction = trainset['date'].max()+DateOffset(hours=24)
        min_time_in_prediction = trainset['date'].max()+DateOffset(hours=1) 
        # set the effective time for nodes who are still effective from trainset
        #df_temp: the temporary dataframe
        df_temp=MainDataFrame.loc[
                (MainDataFrame['date']>=
                min_time_in_prediction-DateOffset(hours=effective_time))&
                (MainDataFrame['date']<min_time_in_prediction)]
        list_of_effective_time_in_trainset = np.array(
                df_temp.groupby('user').max()['date'])
        list_of_users =list(df_temp.groupby('user').max()['date'].keys())
        df_temp =pd.DataFrame({'user':list_of_users,
                   'effective_time':list_of_effective_time_in_trainset})     
        graph_data_drop_duplicate_users = \
                graph_data_drop_duplicate_users.merge(df_temp, how='left', on='user')
        #fill null value in effective time column        
        DataFrameWithDatetimeColumns = \
            graph_data_drop_duplicate_users.select_dtypes(include=['datetime'])
        graph_data_drop_duplicate_users[DataFrameWithDatetimeColumns.columns] = \
            DataFrameWithDatetimeColumns.fillna(
                main_data['date'].min()-DateOffset(hours=45)) 
        #update effective time                  
        graph_data_drop_duplicate_users['effective_time'] =\
            graph_data_drop_duplicate_users[
                'effective_time']+DateOffset(hours=effective_time) 
        #set the effective time for both bi and linear threshold                    
        graph_data_drop_duplicate_users['effective_time_for_linear_threshold']=\
            np.array(graph_data_drop_duplicate_users['effective_time'])
        graph_data_drop_duplicate_users['effective_time_for_bi_threshold']=\
                np.array(graph_data_drop_duplicate_users['effective_time'])
        #delete the effective time column        
        graph_data_drop_duplicate_users=\
                graph_data_drop_duplicate_users.drop(['effective_time'],axis=1)
        """        
        #set active node from last time in trainset        
        list_of_active_nodes_indata.extend(list(main_data.loc[
            (main_data['date']==max_time_in_trainset),'user']))
        list_of_active_nodes_indata.extend(list(main_data.loc[
            (main_data['date']==max_time_in_trainset),'target_user']))
        list_of_active_nodes_indata = list(set(list_of_active_nodes_indata))
        list_of_activenodes_predicted_by_linearthreshold.extend(
            list_of_active_nodes_indata)
        list_of_activenodes_predicted_by_bithreshold.extend(
            list_of_active_nodes_indata)
        """    
        #reset errors for different trainsize
        mean_absolute_persentage_error_linearthreshold=0
        mean_absolute_persentage_error_bithreshold=0   
        #create list of threshold for prediction
        list_ofusers = np.array(graph_data_drop_duplicate_users['user'])
        list_ofout_degreeofusers =1/np.array(graph_data_drop_duplicate_users['out_deg'])
        list_ofthresholds = np.array(graph_data_drop_duplicate_users['threshold'])
        list_ofupthreshold = np.array(graph_data_drop_duplicate_users['up_threshold'])
        list_ofdownthreshold = np.array(graph_data_drop_duplicate_users['down_threshold'])
        window_test = max_time_in_trainset+ DateOffset(hours=6)
        while min_time_in_prediction <= max_time_in_prediction:    
            CurrentTime = min_time_in_prediction 
            EndTime = min_time_in_prediction + DateOffset(hours=1)
            #update the login time for users who are not become active at all
            mask = (graph_data_drop_duplicate_users['counter_bi']==0)
            graph_data_drop_duplicate_users.loc[
                mask, 'login_time_bithreshold'] = CurrentTime  
            #update the login time for users who are not become active at all
            mask = (graph_data_drop_duplicate_users['counter_thre']==0)
            graph_data_drop_duplicate_users.loc[
                mask, 'login_time_linearthreshold'] = CurrentTime  
            #obtain the list of real active node at the moment
            mask = (main_data['date']>=CurrentTime)&(main_data['date']<EndTime)
            list_of_active_nodes_indata.extend(list(set(list(MainDataFrame.loc[mask, 
            'target_user']))))
            list_of_active_nodes_indata.extend(list(main_data.loc[mask,'user']))
## compute the real active nodes
## predict the active node based on linear threhsold model
             #delete the nodes who are not effective
            list_of_effective_node = np.array(
                 graph_data_drop_duplicate_users.loc[
                     (graph_data_drop_duplicate_users[
                         'effective_time_for_linear_threshold'
                         ]>=CurrentTime),'user'])
             #update effective neighbor for each user
            graph_data['active_neighbor']=0
            graph_data.loc[ graph_data['target'].isin(list_of_effective_node),
                            'active_neighbor']=1  
             #obtain the active nodes
            list_of_number_of_activeneighbor = np.array(
                 graph_data.groupby('user').sum()['active_neighbor'])
            list_of_influenceForEachUser=\
                list_of_number_of_activeneighbor*list_ofout_degreeofusers
            mask = np.where(
                 (list_of_influenceForEachUser >= list_ofthresholds) , 
                             True, False)
            list_of_activenode =list_ofusers[mask]
             #delete nodes who are not online
            list_of_offlinenodes=list(
                 graph_data_drop_duplicate_users.loc[
                     (graph_data_drop_duplicate_users['user'].isin(
                         list_of_activenode))&
                     (graph_data_drop_duplicate_users[
                         'login_time_linearthreshold']!=CurrentTime),'user'])
            list_of_activenode= list(
                 set(list_of_activenode)-set(list_of_offlinenodes)) 
             #set the effective time for new active nodes
            mask = (graph_data_drop_duplicate_users['user'].isin(list_of_activenode))
            graph_data_drop_duplicate_users.loc[
                 mask, 'effective_time_for_linear_threshold'] = \
                 CurrentTime+DateOffset(hours=effective_time)    
             #reset login time for new active nodes    
            graph_data_drop_duplicate_users.loc[
                 mask,'login_time_linearthreshold']=CurrentTime
            graph_data_drop_duplicate_users.loc[
                 mask,'counter_thre']=1
             #update login time 
            graph_data_drop_duplicate_users = set_login_time(
                 graph_data_drop_duplicate_users, login_type, 
                 login_time, k, b, CurrentTime, 'threshold')
            list_of_activenodes_predicted_by_linearthreshold.extend(list_of_activenode)
################################################### end of prediction by linear threshold
################################################### epredict the active node based on bithrehsold model        
             #delete the nodes who are not effective
            list_of_effective_node = np.array(
                 graph_data_drop_duplicate_users.loc[
                     (graph_data_drop_duplicate_users[
                         'effective_time_for_bi_threshold'
                         ]>=CurrentTime),'user'])
             #update effective neighbor for each user
            graph_data['active_neighbor']=0
            graph_data.loc[graph_data['target'].isin(list_of_effective_node),
                            'active_neighbor']=1
             #obtain the active nodes
            list_of_number_of_activeneighbor = np.array(
                 graph_data.groupby('user').sum()['active_neighbor'])
            list_of_influenceForEachUser = \
                 list_of_number_of_activeneighbor*list_ofout_degreeofusers 
            mask = np.where(
                 (list_of_influenceForEachUser > list_ofupthreshold) & 
                 (list_of_influenceForEachUser <= list_ofdownthreshold), 
                             True, False)
            list_of_activenode =list_ofusers[mask]
             #delete nodes who are not online
            list_of_offlinenodes=list(
                 graph_data_drop_duplicate_users.loc[
                     (graph_data_drop_duplicate_users['user'].isin(
                         list_of_activenode))&
                     (graph_data_drop_duplicate_users[
                         'login_time_bithreshold']!=CurrentTime),'user'])
            list_of_activenode= list(
                 set(list_of_activenode)-set(list_of_offlinenodes))   
             #set the effective time for new active nodes
            mask = (graph_data_drop_duplicate_users['user'].isin(list_of_activenode))
            graph_data_drop_duplicate_users.loc[
                 mask, 'effective_time_for_bi_threshold'] =  \
                 CurrentTime+DateOffset(hours=effective_time)
             #reset login time for new active nodes    
            graph_data_drop_duplicate_users.loc[
                 mask,'login_time_bithreshold']=CurrentTime
            graph_data_drop_duplicate_users.loc[
                 mask,'counter_bi']=1  
             #update login time           
            graph_data_drop_duplicate_users  = set_login_time(
                 graph_data_drop_duplicate_users, 
                 login_type, login_time, k, b, CurrentTime, 'bi')                
            list_of_activenodes_predicted_by_bithreshold.extend(list_of_activenode) 
################################################### bi_threshold   
             #compute the mean absolute percentage error for both models
            if(min_time_in_prediction==window_test):
                list_of_hours_compute_error.append(window_test)
                window_test = window_test + DateOffset(hours=6)
                mean_absolute_persentage_error_linearthreshold += abs(len(
                    list_of_activenodes_predicted_by_linearthreshold)-len(
                        list_of_active_nodes_indata))/len(list_of_active_nodes_indata)
                mean_absolute_persentage_error_bithreshold += abs(len(
                    list_of_activenodes_predicted_by_bithreshold)-len(
                        list_of_active_nodes_indata))/len(list_of_active_nodes_indata)          
                list_of_active_nodes_indata=[]
                list_of_activenodes_predicted_by_linearthreshold=[]
                list_of_activenodes_predicted_by_bithreshold=[]
            min_time_in_prediction = min_time_in_prediction + DateOffset(hours=1) 
        #save the errors in working directory
        list_of_time_in_trainset.append(max_time_in_trainset)
        list_of_bithreshold_error_for_all_trainsize.append(1/len(
            list_of_hours_compute_error)*mean_absolute_persentage_error_bithreshold)
        list_of_linearthreshold_error_for_all_trainsize.append(1/len(
            list_of_hours_compute_error)*mean_absolute_persentage_error_linearthreshold)
        np.save(code_address+'/%s/result/time_effectivetime%s_logintype%s'
                %(effective_time,effective_time,login_type),list_of_time_in_trainset)
        np.save(code_address+'/%s/result/errorbi_effectivetime%s_logintype%s'
                %(effective_time,effective_time,login_type),
                list_of_bithreshold_error_for_all_trainsize)
        np.save(code_address+'/%s/result/errorre_effectivetime%s_logintype%s'
                %(effective_time,effective_time,login_type),
                list_of_linearthreshold_error_for_all_trainsize)
        trainset_time = trainset_time + DateOffset(hours=6)

