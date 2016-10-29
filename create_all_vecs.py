from swda3 import CorpusReader
import numpy as np
import word2vec
from nltk.tokenize import WordPunctTokenizer
from  collections import deque
import pickle
from nltk.stem.porter import PorterStemmer
#from gensim.models.word2vec import Word2Vec

def main():

    global word_toker
    word_toker = WordPunctTokenizer()

    global stemmer
    stemmer = PorterStemmer()

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

    comm_stat = ['%','t1','t3','x']
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
    #write_text_to_file(corpus)
    #word2vec.word2vec('data/utterances.txt','data/wordvecs.bin',size=wordvec_len)
    wordvecs = word2vec.load('data/wordvecs.bin')
    vocab = wordvecs.vocab


    # create pos (word) vecs
    #write_pos_to_file(corpus)
    #Xword2vec.word2vec('data/utterances_postags.txt','data/posvecs.bin',size=posvec_len)

    # create tag (word) vecs
    #write_tags_to_file(corpus)
    #word2vec.word2vec('data/trans_tag_sens.txt','data/tagvecs.bin',size=tagvec_len)

    tagvecs = word2vec.load('data/tagvecs.bin')

    
    X = []
    y = []
    
    maxl = 0
    for trans in corpus.iter_transcripts(display_progress=False):
        diag_x = []
        diag_y = []
        prev_tag = ''
        for utt in trans.utterances:
            tag,ftm,l = create_x_vec(utt,prev_tag)
            if l > maxl:
                maxl=l
            diag_x.append(ftm)
            diag_y.append(np.array(tag_class_vec(tag)))
            prev_tag = tag
        X.append(np.array(diag_x[:-1]))
        y.append(np.array(diag_y[1:]))
    pickle.dump([np.array(X),np.array(y)],open('data/Xy_complete_tagclasses.pkl','wb'))

    print(maxl)

            
def create_x_vec(utt,prev_tag):
    feat_mat = []
    feat_vec = []
    
    if utt.caller == 'A':
        feat_vec.extend([1,0])
    elif utt.caller == 'B':
        feat_vec.extend([0,1])

    tag = utt.damsl_act_tag()
    if tag == '+':
        tag = prev_tag
    feat_vec.extend(tagvecs[tag])

    feat_mat.append(np.array(feat_vec))

    utt_vec = utterance_vec(utt.text_words())
    feat_mat.append(utt_vec)

    #feat_mat.append(utterance_pos_mat(utt.pos_lemmas()))

    return tag,feat_mat, len(utt_vec)

    
def tag_class_vec(tag):
    vec = np.zeros(len(tag_classes))
    for i in range(len(tag_classes)):
        if tag in tag_classes[i]:
            vec[i]=1
            return vec
        
    print('not in any class: %s'%tag)
    return vec


def utterance_vec(utt_words):
    words = word_toker.tokenize(' '.join([stemmer.stem(w.lower()) for w in utt_words if  w[0] not in ['[','{','/','}',']','<','>','-','+' ]]))
    utt_vec = np.array([wordvecs.vocab_hash[w]+1 for w in words if w in wordvecs.vocab],dtype='int')
    utt_vec = np.pad(utt_vec,(101-len(utt_vec),0),'constant')
    return utt_vec

def utterance_pos_mat(utt_lemmas):
    pos_mat = np.array([posvecs[wp[1]] for wp in utt_lemmas if wp[1] in posvecs.vocab])
    if len(pos_mat) == 0:
        pos_mat = np.array([np.zeros(posvec_len)])
    return pos_mat



def write_text_to_file(corpus):

    with open('data/utterances.txt','w') as f:
        for trans in corpus.iter_transcripts(display_progress=False):
            for utt in trans.utterances:
                utttxt = word_toker.tokenize(' '.join([stemmer.stem(w.lower()) for w in utt.text_words() if  w[0] not in ['[','{','/','}',']','<','>','-','+' ]]))
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
