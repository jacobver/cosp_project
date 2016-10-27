import pickle
import numpy as np
from keras.layers import Dense,Input
from keras.layers.recurrent import SimpleRNN,GRU
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from collections import deque

def main():

    print('loading data...')
    #X = np.array(pickle.load(open('data/X3_20wv_5pv.pkl','rb')))
    #Y = np.array(pickle.load(open('data/y3_20wv_5pv.pkl','rb')))
    [X,y] = pickle.load(open('data/Xy_words_tags.pkl','rb'))

    
    
    split = 90 

    X_train,y_train = build_ngram_data(5,X[:1001],y[:1001])
    X_test,y_test = build_ngram_data(5,X[1001:],y[1001:])

    '''
    X_train = X[:1001]
    y_train = y[:1001]
    
    X_test = X[1001:]
    y_test = y[1001:]

    
    for x,y in zip(X,Y):
        if np.random.randint(100) < split:
            X_train.append(x)
            y_train.append(y)
        else:
            X_test.append(x)
            y_test.append(y)
    
    '''
    n = X_train.shape[1]
    in_n = X_train.shape[2]
    out_n = y_train.shape[1]
    hidden_n = in_n-(in_n-out_n)/2
    

    #in_n = X[0].shape[1]
    #out_n = y[0].shape[1]
    #hidden_n = in_n-(in_n-out_n)/2
    
    print('nr of features:\t%d\nnr of hidden n:\t%d\nnr of labels:\t%d\n'%(in_n,hidden_n,out_n))
    
    print('compiling model computational graph...')
    model = build_ngram_model(n,in_n,out_n,hidden_n)
    #model = build_sequence_model(in_n,hidden_n,out_n)
    
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    print('train...')
    model.fit(np.array(X_train),np.array(y_train),nb_epoch=1000)

    print('test...')
    print(model.evaluate(np.array(X_test),np.array(y_test)))
    y_preds = model.predict(np.array(X_test),verbose=1)

    pickle.dump(y_preds,open('data/output_%dgram_newdata.pkl'%n,'wb'))
    pickle.dump(y_test,open('data/test_%dgram_newdata.pkl'%n,'wb'))


def build_sequence_model(in_n,hidden_n,out_n):
    inputs = Input(shape=(1000,))
    x = TimeDistributed(GRU(hidden_n,input_shape=(500,in_n),activation='tanh',return_sequences=True))(inputs)
    outputs = Dense(out_n,W_regularizer=l2(0.01), activation='softmax')(x)
    return Model(inputs=inputs,outputs=outputs)

def build_ngram_model(n,inlen,outlen,hidden_n):
    model = Sequential()
    model.add(GRU(hidden_n,input_shape=(n,inlen),activation='tanh',return_sequences=False))
    model.add(Dense(outlen,W_regularizer=l2(0.01), activation='softmax'))

    return model

def build_ngram_data(n,Xin,yin):
    X = []
    y = []
    for diag_x,diag_y in zip(Xin,yin):
        gram = deque([],n)
        for x,single_y in zip(diag_x,diag_y):
            if len(gram) == n:
                X.append(np.array(gram))
                y.append(np.array(single_y))
            else:
                gram.append(x)
    return np.array(X),np.array(y)
    
if __name__ == '__main__':
    main()
