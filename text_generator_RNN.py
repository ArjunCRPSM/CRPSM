import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "/home/crpsm/Pycharm/DataSet/wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars=sorted(list(set(raw_text)))
chars_to_int=dict((c,i) for i,c in enumerate(chars))

n_chars=len(raw_text)
n_vocabs=len(chars_to_int)

seq_len=100
x=[]
y=[]
for i in range(0,n_chars-seq_len,1):
	se_in=raw_text[i:i+seq_len]
	seq_out=raw_text[i+seq_len]

	x.append([chars_to_int[chars] for chars in se_in])
	y.append([chars_to_int[seq_out]])

n_patterns=len(x)
print(n_patterns)

x=numpy.reshape(x,(n_patterns,seq_len,1))
y=np_utils.to_categorical(y)

x=x/float(n_vocabs)

model=Sequential()
model.add(LSTM(256,input_shape=(x.shape[1],x.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(x,y,epochs=50,batch_size=64)

model.save('/home/crpsm/Pycharm/Saved_Models/Text_Generator.h5')

start = numpy.random.randint(0, len(x)-1)
pattern = x[start]
print("Seed:")
print("\"", ''.join([chars_to_int[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocabs)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = chars_to_int[index]
	seq_in = [chars_to_int[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")

model.save('/home/crpsm/Pycharm/Saved_Models/Text_Generator_protection.h5')

import matplotlib.pyplot as plt




