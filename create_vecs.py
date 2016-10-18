from swda_time import CorpusReader
import numpy as np
import word2vec
from nltk.tokenize import RegexpTokenizer
from  collections import deque
import pickle

def main():

    n = 5
    global tag_vecs
    global wordvecs
    tags = ['sd','b','sv','aa','%','+','ha','qy','x','ny','fc','qw','nn','bk','h','qy^d','fo_o_fw_by_bc','bh','^q','bf','na','ad', '^2','b^m','qo','qh','^h','ar','ng','br','no','fp','qrr','arp_nd','t3','oo_co_cc','t1','bd','aap_am', '^g','qw^d','fa','ft','ba','fo_o_fw_"_by_bc']
    tag_vecs = dict(zip(tags,np.eye(len(tags))))

    corpus = CorpusReader('swda_time', 'swda_time/swda-metadata-ext.csv')

    #write_text_to_file(corpus)
    #word2vec.word2vec('data/utterances.txt','data/wordvecs.bin',size=100)
    #'''
    wordvecs = word2vec.load('data/wordvecs.bin')

    featvecs = []
    for trans in corpus.iter_transcripts(display_progress=False):
        end_prev_turn = trans.utterances[0].end_turn
        for utt in trans.utterances:
            featvecs.append(create_feature_vec(utt,end_prev_turn))
            end_prev_turn = utt.end_turn

    build_ngram_data(featvecs,n,len(tags))
    #'''
    
def build_ngram_data(featvecs,n,ylen):

    X = []
    y = []
    gram = deque([],n)
    for i in range(len(featvecs)-1):
        if len(gram) == n:
            X.append(np.array(gram))
            y.append(featvecs[i+1][:ylen])
        if None in featvecs[i]:
            gram = deque([],n)
        else:
            gram.append(featvecs[i])
            
    pickle.dump(X, open('data/X.pkl','wb'))
    pickle.dump(y, open('data/y.pkl','wb'))        
            
            
def create_feature_vec(utt,end_prev_turn):
    feat_vec = []

    feat_vec.extend(tag_vecs[utt.damsl_act_tag()])
    feat_vec.extend(utterance_vec(utt.text))

    # add transition time 
    try:
        feat_vec.append(utt.start_turn - end_prev_turn)
    except TypeError:
        feat_vec.append(None)

    # add utterance lengts in milisecond 
    try:
        feat_vec.append(utt.end_turn - utt.start_turn)
    except TypeError:
        feat_vec.append(None)

    return feat_vec


def utterance_vec(utt_text):
    tok = RegexpTokenizer(r'\w+')
    
    utttxt = [wordvecs[t.lower()] for t in tok.tokenize(utt_text) if t in wordvecs.vocab]
    if len(utttxt) > 0:
        utt_vec = np.sum(utttxt,axis=0)/len(utttxt[0])
    else:
        utt_vec = np.zeros(100)
    return utt_vec
    
def write_text_to_file(corpus):
    tok = RegexpTokenizer(r'\w+')
    #tok = RegexpTokenizer('\\{[ACFDE]|\\}|/|-|\\[|\\]|\\(|\\)|<|>|\\+","',gaps=True)
    with open('data/utterances.txt','w') as f:
        for trans in corpus.iter_transcripts(display_progress=False):
            for utt in trans.utterances:
                utttxt = [t.lower() for t in tok.tokenize(utt.text) if len(t) > 1]
                utttxt.append('\n')
                f.write(' '.join(utttxt))
            

if __name__ == '__main__':
    main()
