#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import os
from create_dataframe import create_dataframe_and_graph_data, add_feature_in_dataset
from getfeature import create_features_for_all_trainset
from make_feature_for_thresholdmodels import *
from multiprocessing import Pool
from predict import *
from bi_threshold.CTL.causal_tree_learn import CausalTree_bi_threshold
from CTL.causal_tree_learn import CausalTree
from drawplot import*
#change
#define main variable that use in project
#you can choose which login type or k or b you want to use 
code_address = os.getcwd()
print(code_address)
#make folder for plots
os.mkdir(code_address+'/plots/')
flag_get_feature = True
flag_craete_data = True
effective_time = [5,10,15,20,25]
#login_type=['constant','linear','exponential']
login_type='exponential'
if login_type=='constant':
    login_time = [ i for i in range(1,25) if i%2==0]
else:
    login_time=[]
# you can choose the k from k=[6,8,12,18,24]
k=8
# you can change the base in  login_type='exponential'   
b=2
AddressList=[5,10,15,20,25]
def main_func():   
    #set the data address
    data_address = os.path.join(code_address, 'data/higgs-activity_time.txt')
    graph_address=os.path.join(code_address,'data/higgs-social_network.edgelist')
    print('data_address: ',data_address,'graph_address: ',graph_address)
    #call create_dataframe_and_graph_data function to read data 
    if flag_craete_data==True:
        main_data,graph_data,network_of_following = create_dataframe_and_graph_data(
            data_address, 
            graph_address)
        #call add_feature_in_dataset function to add features to data
        graph_data,graph_data_drop_duplicate_users = add_feature_in_dataset(
            main_data, 
            graph_data, 
            network_of_following)
        print('graph_data_shape: ',graph_data.shape,
              'shape: ',graph_data_drop_duplicate_users.shape)
    #call create_features_for_all_trainset function to obtain features for different trainset 
    #based on the time of trainset    
    if flag_get_feature==True:
        create_features_for_all_trainset(
            main_data, 
            graph_data,
            network_of_following)
        flag_get_feature==False  
    """after we obtain features we should create inputs for 
    both linear threshold and bithreshold model 
    """
#we call computethreshold function parallary for all trainsize    
def paraller():
    pool=Pool(processes=6)
    pool.starmap(compute_thresholds,a)  
         
#for each trainsize, we should compute threshold, up and down thresholds        
def compute_thresholds(b,d):  
    #set the variable that we need for model  
    y_train = np.load(code_address+'/%s/threshold/y_train%s.npy' %(b,d))
    t_train=np.load(code_address+'/%s/threshold/t_train%s.npy'%(b,d))
    x_train=np.load(code_address+'/%s/threshold/x_train%s.npy'%(b,d))
    x_test=np.load(code_address+'/%s/threshold/x_test%s.npy'%(b,d))
    variable_names = []
    for i in range(x_train.shape[1]):
        variable_names.append(f"Column {i}") 
    # regular CTL
    ctl = CausalTree(cont=True)
    ctl.fit(x_train, y_train, t_train)
    ctl_predict = ctl.predict(x_test)
    triggers = ctl.get_triggers(x_test)
    #save thresholds
    np.save(code_address+'/%s/threshold/threshold%s'%(b,d), triggers)
    # bithreshold CTL
    ctl_bi_threshold = CausalTree_bi_threshold(cont=True)
    ctl_bi_threshold.fit(x_train, y_train, t_train)
    ctl_predict_bi_threshold = ctl_bi_threshold.predict(x_test)
    triggers_bi_threshold = ctl_bi_threshold.get_triggers(x_test)
    #save thresholds
    np.save(code_address+'/%s/threshold/down_threshold%s'%(b,d), 
    triggers_bi_threshold[0])
    np.save(code_address+'/%s/threshold/up_threshold%s'%(b,d), 
    triggers_bi_threshold[1]) 

if __name__ == "__main__":
    main_data,graph_data,graph_data_drop_duplicate_users =main_func()  
    for i in effective_time:
        #for each effective time we should get the influence
        create_inputs_for_thresholdmodel(
            main_data, 
            graph_data, 
            graph_data_drop_duplicate_users, 
            i, 
            login_type,
            login_time,
            k,
            b)      
        #for each effective time we should make input 
        make_input_for_models(i)
        #we call computethreshold function parallary for all trainsize
        a = [(i,k)for k in range(1,24) ]
        paraller()
        print('finished')
        #after we get thresholds we call prediction function   
        prediction(main_data,
                   graph_data, 
                   graph_data_drop_duplicate_users, 
                   i ,
                   login_type, 
                   login_time, 
                   k, 
                   b)
        plot(login_type,login_time,k,b,i)
