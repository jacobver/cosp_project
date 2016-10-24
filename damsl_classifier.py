import pickle
import numpy as np
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.regularizers import l2



def main():

    print('loading data...')
    X = np.array(pickle.load(open('data/X3_20wv_5pv.pkl','rb')))
    Y = np.array(pickle.load(open('data/y3_20wv_5pv.pkl','rb')))

    n = X.shape[1]

    split = 90 
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    
    for x,y in zip(X,Y):
        if np.random.randint(100) < split:
            X_train.append(x)
            y_train.append(y)
        else:
            X_test.append(x)
            y_test.append(y)
         
    in_n = X.shape[2]
    out_n = Y.shape[1]
    hidden_n = in_n-(in_n-out_n)/2
    print('nr of features:\t%d\nnr of hidden n:\t%d\nnr of labels:\t%d\n'%(in_n,hidden_n,out_n))
    
    print('compiling model computational graph...')
    model = build_model(n,in_n,out_n,hidden_n)

    
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    print('train...')
    model.fit(np.array(X_train),np.array(y_train),nb_epoch=300)

    print('test...')
    print(model.evaluate(np.array(X_test),np.array(y_test)))
    y_preds = model.predict(np.array(X_test),verbose=1)

    pickle.dump(y_preds,open('data/output_%dgram_20wv_5pv.pkl'%n,'wb'))
    pickle.dump(y_test,open('data/test_%dgram_20wv_5pv.pkl'%n,'wb'))
    
def build_model(n,inlen,outlen,hidden_n):
    model = Sequential()
    model.add(SimpleRNN(hidden_n,input_shape=(n,inlen),activation='tanh',return_sequences=False))
    model.add(Dense(outlen,W_regularizer=l2(0.01), activation='softmax'))

    return model
                  
    
if __name__ == '__main__':
    main()
