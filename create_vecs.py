from swda import CorpusReader
import numpy as np
import word2vec
from nltk.tokenize import WordPunctTokenizer
from  collections import deque
import pickle
#from gensim.models.word2vec import Word2Vec

def main():

    global word_toker
    word_toker = WordPunctTokenizer()
    global tag_vecs_1h
    global tagvecs
    global wordvecs
    global posvecs
    global wordvec_len
    global posvec_len
    global tag_classes
    global prev_tag
    
    n = 3
    wordvec_len = 50
    posvec_len = 10
    tagvec_len = 20
    
    tags = ['sd','b','sv','aa','%','ha','qy','x','ny','fc','qw','nn','bk','h','qy^d','fo_o_fw_by_bc','bh','^q','bf','na','ad', '^2','b^m','qo','qh','^h','ar','ng','br','no','fp','qrr','arp_nd','t3','oo_co_cc','t1','bd','aap_am', '^g','qw^d','fa','ft','ba','fo_o_fw_"_by_bc']

    comm_stat = ['%','t1','t3','x','%']
    inf = ['^t','^c']
    statement = ['sd','sv']
    infl_adr_fut_act = ['qy', 'qw', 'qo', 'qr', 'qrr', '^d', '^g','ad','qy^d','qw^d','qh']
    comm_spkr_fut_act = ['oo','co','cc','oo_co_cc']
    oth_frwd_func = ['fp','fc','fx','fe','fo','ft','fw','fa','fo_o_fw_"_by_bc']
    agreement = ['aa','aap','am','arp','ar','^h','aap_am']
    understanding = ['br','br^m','b','bh','bk','b^m','^2','bf','ba','by','bd','bc']
    answer = ['ny','nn','na','ng','no','nn^e','ny^e','sd^e','sv^e','^e','arp_nd']
    other = ['^q','h']

    tag_classes = [comm_stat,inf,statement,infl_adr_fut_act,comm_spkr_fut_act,oth_frwd_func,agreement,understanding,answer,other]
    tag_vecs_1h = dict(zip(tags,np.eye(len(tags))))

    corpus = CorpusReader('swda')

    # create word vecs
    write_text_to_file(corpus)
    word2vec.word2vec('data/utterances.txt','data/wordvecs.bin',size=wordvec_len)

    # create pos (word) vecs
    #write_pos_to_file(corpus)
    #Xword2vec.word2vec('data/utterances_postags.txt','data/posvecs.bin',size=posvec_len)

    # create tag (word) vecs
    #write_tags_to_file(corpus)
    #word2vec.word2vec('data/trans_tag_sens.txt','data/tagvecs.bin',size=tagvec_len)
    
    
    wordvecs = word2vec.load('data/wordvecs.bin')
    posvecs =  word2vec.load('data/posvecs.bin')
    tagvecs = word2vec.load('data/tagvecs.bin')
    
    nrT = 0
    nrUtt = 0
    X = []
    y = []
    
    ylen = len(tags)
    for trans in corpus.iter_transcripts(display_progress=False):
        diag_x = []
        diag_y = []
        prev_uttvec = np.zeros(wordvec_len)
        prev_tag = ''
        nrT += 1
        for utt in trans.utterances:
            tag,ftv = create_feature_vec(utt,prev_uttvec,prev_tag)
            diag_x.append(ftv)
            diag_y.append(np.array(tag_vecs_1h[tag]))
            prev_uttvec = ftv[tagvec_len:tagvec_len+wordvec_len]
            prev_tag = tag
            nrUtt += 1
        X.append(np.array(diag_x[:-1]))
        y.append(np.array(diag_y[1:]))
    pickle.dump([np.array(X),np.array(y)],open('data/Xy_words_tags.pkl','wb'))

    #build_ngram_data(featvecs,n,ylen)

    print(nrT,nrUtt)
    #'''
            
def create_feature_vec(utt,prev_uttvec,prev_tag):
    feat_vec = []

    tag = utt.damsl_act_tag()
    if tag == '+':
        tag = prev_tag
    feat_vec.extend(tagvecs[tag])

    uttvec = utterance_vec(utt.text_words())
    feat_vec.extend(uttvec)

    #feat_vec.extend(utterance_pos_vec(utt.pos_lemmas()))

    # add nr of word in utterance
    #feat_vec.append(len(utt.pos_words()))

    # add cosine distance of previous and current utterance vec
    #feat_vec.append(cosine_dist(prev_uttvec,uttvec))

    if utt.caller == 'A':
        feat_vec.extend([1,0])
    elif utt.caller == 'B':
        feat_vec.extend([0,1])

    return tag,np.array(feat_vec)


def build_ngram_data(featvecs,n,ylen):

    X = []
    y = []
    gram = deque([],n)
    for i in range(len(featvecs)-1):
        if len(gram) == n:
            X.append(np.array(gram))
            y.append(np.array(featvecs[i+1][:ylen]))
        if None in featvecs[i]:
            gram = deque([],n)
        else:
            gram.append(featvecs[i])
            
    pickle.dump(X, open('data/X%d_%dwv_%dpv.pkl'%(n,wordvec_len,posvec_len),'wb'))
    pickle.dump(y, open('data/y%d_%dwv_%dpv.pkl'%(n,wordvec_len,posvec_len),'wb'))        

    
def tag_class_vec(tag):
    global prev_tag
    if tag == '+':
        tag = prev_tag
    vec = np.zeros(len(tag_classes))
    for i in range(len(tag_classes)):
        if tag in tag_classes[i]:
            vec[i]=1
            prev_tag = tag
            return vec
        
    print('not in any class: %s'%tag)
    return vec


def cosine_dist(u,v):
    return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v))) 

def utterance_vec(utt_words):
    words = word_toker.tokenize(' '.join([w.lower() for w in utt_words if w[0] != '{' ]))
    utttxt = np.array([wordvecs[w] for w in words if w in wordvecs.vocab])
    if len(utttxt) > 0:
        utt_vec = np.sum(utttxt,axis=0)/len(utttxt)
    else:
        utt_vec = np.zeros(wordvec_len)
        print('no words!! -- %s'%utt_words)
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
                utttxt = word_toker.tokenize(' '.join([w.lower() for w in utt.text_words() if  w[0] != '{' ]))
                utttxt.append('\n')
                f.write(' '.join(utttxt))

def write_pos_to_file(corpus):
    with open('data/utterances_postags.txt','w') as f:
        for trans in corpus.iter_transcripts(display_progress=False):
            for utt in trans.utterances:
                utttxt = [w_p[1] for w_p in utt.pos_lemmas()]
                utttxt.append('\n')
                f.write(' '.join(utttxt))

def write_tags_to_file(corpus):
    with open('data/trans_tag_sens.txt','w') as f:
        for trans in corpus.iter_transcripts(display_progress=False):
            tag_sen = []
            prev_tag = ''
            for utt in trans.utterances:
                t = utt.damsl_act_tag()
                tag = prev_tag if t == '+' else t
                tag_sen.append(tag)
                prev_tag = tag
                tag_sen.append('\n')
            f.write(' '.join(tag_sen))


if __name__ == '__main__':
    main()
