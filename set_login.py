#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 23:33:16 2022

@author: macbook
"""
import pandas as pd
import numpy as np 
import math
from pandas.tseries.offsets import DateOffset 

def set_login_time(graph_data_drop_duplicate_users,login_type,login_time,k,b,window_min,model):
    #check which model wants to update its logintime then identify the corect columns
    if (model=='threshold'):
        col_login='login_time_linearthreshold'
        col_counter = 'counter_thre'
    elif (model=='featureformodels' ):
        col_login='login_time'
        col_counter = 'counter'       
    else:
        col_login='login_time_bithreshold'
        col_counter = 'counter_bi'
    #update login time based the login type   
    if (login_type=='constant'):
        mask = (graph_data_drop_duplicate_users[col_counter]==1)
        p=pd.DataFrame(np.array(graph_data_drop_duplicate_users.loc[
            mask, col_login]+DateOffset(hours=login_time)))
        graph_data_drop_duplicate_users.loc[
            mask, col_login] = np.array(p[0])
    elif (login_type=='linear'):
        for i in range(1,k):
            mask = (graph_data_drop_duplicate_users[col_counter]==i)& (
                     graph_data_drop_duplicate_users[col_login]==window_min)
            p=pd.DataFrame(np.array(
                graph_data_drop_duplicate_users.loc[
                    mask, col_login]+DateOffset(hours=i)))
            graph_data_drop_duplicate_users.loc[
                mask, col_login] = np.array(p[0])
            graph_data_drop_duplicate_users.loc[mask, col_counter] = \
                    graph_data_drop_duplicate_users.loc[
                        mask, col_counter].apply(lambda x: x+1)
        mask = (graph_data_drop_duplicate_users[col_counter]==k)& (
                       graph_data_drop_duplicate_users[col_login]==window_min)
        p=pd.DataFrame(np.array(
                       graph_data_drop_duplicate_users.loc[
                           mask, col_login]+DateOffset(hours=k)))
        graph_data_drop_duplicate_users.loc[
                       mask, col_login] = np.array(p[0]) 
        
    else :
        temp=math.floor(math.log(k, b))
        for i in range(1,temp):
            mask = (graph_data_drop_duplicate_users[col_counter]==i)& (
                     graph_data_drop_duplicate_users[col_login]==window_min)
            p=pd.DataFrame(np.array(graph_data_drop_duplicate_users.loc[
                mask, col_login]+DateOffset(hours=b^(i-1))))
            graph_data_drop_duplicate_users.loc[
                mask, col_login] = np.array(p[0])
            graph_data_drop_duplicate_users.loc[mask, col_counter] = \
                    graph_data_drop_duplicate_users.loc[
                        mask, col_counter].apply(lambda x: x+1)
        mask = (graph_data_drop_duplicate_users[col_counter]==temp)& (
                graph_data_drop_duplicate_users[col_login]==window_min)
        p=pd.DataFrame(np.array(
                graph_data_drop_duplicate_users.loc[
                    mask, col_login]+DateOffset(hours=k)))
        graph_data_drop_duplicate_users.loc[
                mask, col_login] = np.array(p[0])      
       
        
              
    return graph_data_drop_duplicate_users