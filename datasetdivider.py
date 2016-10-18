# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 14:55:22 2016

@author: Prajit Dhar
"""

from swda_time import CorpusReader
from swda_time import Transcript
import random



numtranscripts=list(range(1,644))
traintranscripts=random.sample(numtranscripts, int(round(0.8*643)))

testtranscripts=list(set(numtranscripts).symmetric_difference(set(traintranscripts)))


i=1
traindata=[]
testdata=[]
corpus = CorpusReader('swda_time', 'swda_time/swda-metadata-ext.csv')
for transcript in corpus.iter_transcripts(display_progress=False):
    if i in traintranscripts:
        traindata.append(transcript)
        i+=1
    else:
        testdata.append(transcript)
        i+=1
    #j=0
        #j+=1
    #print "\n There are ",j," utterenances in transcript",i
    i+=1
    



