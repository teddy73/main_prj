import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.dates as dates
import os
def plot(LoginType,LoginTime,K,B,EffectiveTime):
    #get the working directory
    CodeAddress = os.getcwd()
    error_linear= np.load(
            CodeAddress+'/%s/result/errorre_effectivetime%s_logintype%s.npy'
            %(EffectiveTime,EffectiveTime,LoginType))
    error_bi= np.load(CodeAddress+'/%s/result/errorbi_effectivetime%s_logintype%s.npy'
                %(EffectiveTime,EffectiveTime,LoginType))
    list_time = np.load(CodeAddress+'/%s/result/time_effectivetime%s_logintype%s.npy'
                %(EffectiveTime,EffectiveTime,LoginType))
    plt.rcParams.update({'font.size': 50})
    fig,ax= plt.subplots(figsize=(40,20),dpi=100)
    def format_date(x, pos=None):
        return dates.num2date(x).strftime('%B-%d %H:%M') #use FuncFormatter to format dates
    m=[dates.date2num(list_time) for i in range(0,20)]
    m=m[0]
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    ax.xaxis.set_major_locator(dates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(dates.DateFormatter("%B-%d %H:%M"))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(float(x), ',')))
    plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    ax.plot(list_time,error_bi, marker='o', 
            markerfacecolor='blue', markersize=20,lw=10.0,color=color_sequence[14])
    ax.plot(list_time,error_linear,linestyle='--', 
            marker='o', markerfacecolor='blue', markersize=20,linewidth=10.0,color=color_sequence[15])
    ax.xaxis.set_major_locator(ticker.FixedLocator(m)) # I want all the dates on my xaxis
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    plt.ylabel('Mean absolute percentage error ($*100$)')
    plt.xlabel('prediction time')
    plt.legend(['bithreshold with $t_{eff}$=%s'%(EffectiveTime),
    'linear threshold with $t_{eff}$=%s'%(EffectiveTime)])
    plt.xticks(rotation=90)
    plt.savefig(CodeAddress+'plot/effectivetime%s_logintype%s.jpg'
    %(EffectiveTime,LoginType),bbox_inches='tight', dpi=150)