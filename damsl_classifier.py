import pickle
import numpy as np
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential


def main():

    print('loading data...')
    X = np.array(pickle.load(open('data/X.pkl','rb')))
    Y = np.array(pickle.load(open('data/y.pkl','rb')))

    n = X.shape[1]

    split = 80 
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
    hidden_n = 80
    out_n = Y.shape[1]

    print('compiling model computational graph...')
    model = build_model(n,in_n,out_n,hidden_n)

    
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    print('train...')
    model.fit(np.array(X_train),np.array(y_train),nb_epoch=1)

    print('test...')
    print(model.evaluate(np.array(X_test),np.array(y_test)))
    y_preds = model.predict(np.array(X_test),verbose=1)
    
def build_model(n,inlen,outlen,hidden_n):
    model = Sequential()
    model.add(SimpleRNN(hidden_n,input_shape=(n,inlen),activation='tanh',return_sequences=False))
    model.add(Dense(outlen,activation='softmax'))

    return model
                  
    
if __name__ == '__main__':
    main()
