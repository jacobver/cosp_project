from keras.layers import Input,Dense,Embedding,merge, Reshape
from keras.layers.recurrent import GRU, LSTM
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers.wrappers import TimeDistributed
import numpy as np
import pickle
from collections import deque



def main():

    voc_size=6507
    max_utt_len = 101
    ngram_n = 5
    
    print('loading data...')

    [X,y] = pickle.load(open('data/Xy_complete_tagclasses.pkl','rb'))

    Xt_train,Xw_train,y_train = build_ngram_data(ngram_n,X[:1116],y[:1116])
    Xt_test,Xw_test,y_test = build_ngram_data(ngram_n,X[1116:],y[1116:])

    n = ngram_n
    in_n = len(Xw_train)
    out_n = y_train.shape[1]
    embedding_n = 50
    utt_seq_n = 40
    
    tag = Input(shape=(n,22))
    
    utt_words = Input(shape=(n,101),dtype='int32')
    embedded = TimeDistributed(Embedding(voc_size+2,embedding_n,input_length=max_utt_len,mask_zero=True))(utt_words)
    uttvec = TimeDistributed(GRU(embedding_n,input_length=101,return_sequences=False))(embedded)

    utt_all = merge([tag,uttvec],mode='concat')
    
    x =  GRU(utt_seq_n,return_sequences=False,input_length=n)(utt_all)

    outputs = Dense(out_n, activation='softmax')(x)

    model = Model(input=[tag,utt_words],output=outputs)
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])


    model.fit([Xt_train,Xw_train],y_train,nb_epoch=1,batch_size=512)

    print(model.evaluate([Xt_test,Xw_test],y_test))
    y_preds = model.predict([Xt_test,Xw_test],verbose=1)

    pickle.dump(y_preds,open('data/output_%dgram_classes_test1.pkl'%n,'wb'))
    pickle.dump(y_test,open('data/test_%dgram_classes_test1.pkl'%n,'wb'))

def build_ngram_data(n,Xin,yin):
    Xt = []
    Xw = []
    y = []
    for diag_x,diag_y in zip(Xin,yin):
        tag_gram = deque([],n)
        word_gram = deque([],n)
        for [tx,wx],single_y in zip(diag_x,diag_y):
            if len(word_gram) == n:
                Xt.append(np.array(tag_gram))
                Xw.append(np.array(word_gram))
                y.append(np.array(single_y))
            else:
                tag_gram.append(tx)
                word_gram.append(wx)
                         
    return np.array(Xt),np.array(Xw),np.array(y)


if __name__ == '__main__':
    main()
