# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 05:15:21 2016

@author: Prajit Dhar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
#import pickle

tags = ['comm_stat','inf','statement','infl_adr_fut_act','comm_spkr_fut_act','oth_frwd_func','agreement','understanding','answer','other']

def main():
    y_preds = np.array(pickle.load(open('data/output_3gram_smallvecs.pkl','rb')))
    y_test = np.array(pickle.load(open('data/output_3gram_smallvecstest.pkl','rb')))
    pdpredict=pd.DataFrame(y_preds,columns=tags)
    pdy_test=pd.DataFrame(y_test,columns=tags)
    label=pdy_test.idxmax(axis=1)
    storeperc=[]
    storeloc=[]
    storeperc,storeloc=data_creator(pdpredict,label)
    #print len(storeperc), len(storeloc)
    #Create Dataframes
    pdstoreperc=pd.DataFrame(storeperc,columns=["Percentages"])
    pdstoreloc=pd.DataFrame(storeloc,columns=["Count"],dtype='category')
    #print pdstoreloc.shape
    #print pdstoreperc.shape
    #pdstoreloc["Count_Percentage"]=pdstoreloc.Count/pdstoreloc.shape[0]
    pdstoreloc.to_pickle('data\location.pkl')
    pdstoreperc.to_pickle('data\percentages.pkl')
    data_analyser(pdstoreperc,pdstoreloc)
    data_plotter(pdstoreperc,pdstoreloc)
    
def data_creator(df,label):
    sc=[]
    sl=[]
    for i in xrange(df.shape[0]):
    
        temp=df.iloc[i].sort_values(ascending=False)
        sc.append((temp[0]-temp[label[i]])/temp[0])
        tempsymbol=np.where(temp.index==label[i])[0]
        sl.append(tempsymbol[0])
    #print len(sl)
    return sc,sl
    
    
def data_analyser(perc,loc):
    print "\n Data description for Percentage difference of the predicted tags \n"
    print perc.describe()
    print "\n Data description for Tag difference of the predicted tags \n"
    print loc.Count.value_counts()
    
    print "\n Normalized values"
    print loc.Count.value_counts(normalize=True)





def data_plotter(perc,loc):
    print "\n Constructing graphs for Percentage change distribution"
    print "\n Box Plots"
    plt.figure()
    perc.plot.box()
    
    #print "\n Overall Histogram"
    perc.plot.hist(bins=100,normed=True,color='green')
    plt.figure()
    #print "\n Histogram to show binned data grained Histogram"  
    perc.plot.hist(bins=10,range= (0.01,1),alpha=0.9,normed=True,color='red')
    plt.figure()
    #print "\n Fine grained Histogram"
    perc.plot.hist(bins=100,range= (0.01,1),alpha=0.9,normed=True,color='yellow')
    plt.figure()
    loc.Count.value_counts().plot(kind='barh',stacked=True,color='violet')
    plt.figure()
    loc.Count.value_counts().cumsum().plot()
    

if __name__ == '__main__':
    main()
    
    
    