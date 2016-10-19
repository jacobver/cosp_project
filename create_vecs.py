from swda_time import CorpusReader
import numpy as np
import word2vec
from nltk.tokenize import RegexpTokenizer
from  collections import deque
import pickle
#from gensim.models.word2vec import Word2Vec

def main():

    global tag_vecs
    global wordvecs
    global posvecs
    global wordvec_len
    global posvec_len
    
    n = 5
    wordvec_len = 100
    posvec_len = 20
    
    
    tags = ['sd','b','sv','aa','%','+','ha','qy','x','ny','fc','qw','nn','bk','h','qy^d','fo_o_fw_by_bc','bh','^q','bf','na','ad', '^2','b^m','qo','qh','^h','ar','ng','br','no','fp','qrr','arp_nd','t3','oo_co_cc','t1','bd','aap_am', '^g','qw^d','fa','ft','ba','fo_o_fw_"_by_bc']
    tag_vecs = dict(zip(tags,np.eye(len(tags))))

    corpus = CorpusReader('swda_time', 'swda_time/swda-metadata-ext.csv')

    # create word vecs
    #write_text_to_file(corpus)
    #word2vec.word2vec('data/utterances.txt','data/wordvecs.bin',size=wordvec_len)

    # create pos (word) vecs
    #write_pos_to_file(corpus)
    #word2vec.word2vec('data/utterances_postags.txt','data/posvecs.bin',size=posvec_len)

    wordvecs = word2vec.load('data/wordvecs.bin')
    posvecs =  word2vec.load('data/posvecs.bin')

    #'''
    featvecs = []
    for trans in corpus.iter_transcripts(display_progress=False):
        end_prev_turn = trans.utterances[0].end_turn
        prev_uttvec = np.zeros(wordvec_len)
        for utt in trans.utterances:
            ftv = create_feature_vec(utt,end_prev_turn,prev_uttvec)
            featvecs.append(ftv)
            end_prev_turn = utt.end_turn
            prev_uttvec = ftv[len(tags):len(tags)+wordvec_len]

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
            
    pickle.dump(X, open('data/X%d.pkl'%n,'wb'))
    pickle.dump(y, open('data/y%d.pkl'%n,'wb'))        
            
            
def create_feature_vec(utt,end_prev_turn,prev_uttvec):
    feat_vec = []

    feat_vec.extend(tag_vecs[utt.damsl_act_tag()])
    uttvec = utterance_vec(utt.pos_words())
    feat_vec.extend(uttvec)
    feat_vec.extend(utterance_pos_vec(utt.pos_lemmas()))

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

    # add nr of word in utterance
    feat_vec.append(len(utt.pos_words()))

    # add cosine distance of previous and current utterance vec
    feat_vec.append(cosine_dist(prev_uttvec,uttvec))
    
    return feat_vec

def cosine_dist(u,v):
    return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v))) 

def utterance_vec(utt_words):
    
    utttxt = np.array([wordvecs[w.lower()] for w in utt_words if w in wordvecs.vocab])
    if len(utttxt) > 0:
        utt_vec = np.sum(utttxt,axis=0)/len(utttxt)
    else:
        utt_vec = np.zeros(wordvec_len)
    return utt_vec

def utterance_pos_vec(utt_lemmas):
    
    uttpos = np.array([posvecs[wp[1]] for wp in utt_lemmas if wp[1] in posvecs.vocab])
    if len(uttpos) > 0:
        pos_vec = np.sum(uttpos,axis=0)/len(uttpos)
    else:
        pos_vec = np.zeros(posvec_len)
    return pos_vec
    
def write_text_to_file(corpus):
    with open('data/utterances.txt','w') as f:
        for trans in corpus.iter_transcripts(display_progress=False):
            for utt in trans.utterances:
                utttxt = [w.lower() for w in utt.pos_words()]
                utttxt.append('\n')
                f.write(' '.join(utttxt))

def write_pos_to_file(corpus):
    with open('data/utterances_postags.txt','w') as f:
        for trans in corpus.iter_transcripts(display_progress=False):
            for utt in trans.utterances:
                utttxt = [w_p[1] for w_p in utt.pos_lemmas()]
                utttxt.append('\n')
                f.write(' '.join(utttxt))


if __name__ == '__main__':
    main()
