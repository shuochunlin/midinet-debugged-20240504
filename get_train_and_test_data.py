import numpy as np
import random
import math
import time

# Note the number of augmentations taken prior in get_data.py
data_aug_count = 8
# create_bar_offset = 4

#load data

data = np.load('data_x.npy')   # size is (training_samples, 1, 128, 16), where 128: midi notes, 16: timestep length
prev_data = np.load('prev_x.npy')

print(np.shape(data))
print(np.shape(prev_data))

chord_data = np.load('data_y.npy')

print(np.shape(chord_data))

# data = np.load('octave2_x_T.npy')
# prev_data = np.load('octave2_prev_x_T.npy')
print('data shape: {}'.format(data.shape))
time.sleep(3)

song_idx = int(data.shape[0]/8) // (data_aug_count + 1)
test_ratial = 0.1
test_song_num = round(song_idx*test_ratial)
train_song_num = data.shape[0] - test_song_num
print('total song number: {}'.format(song_idx))
print('number of test song: {}, \n,number of train song: {}'.format(test_song_num,train_song_num))
time.sleep(3)

#create the song idx for test data

full = np.arange(song_idx)

test_idx= random.sample(range(0,full.shape[0]),test_song_num)
test_idx = np.asarray(test_idx)
print('total {} song idx for test: {}'.format(test_idx.shape[0],test_idx))
time.sleep(3)

#create the song idx for train data
train_idx = np.delete(full,test_idx)
print('total {} song idx for train: {}'.format(train_idx.shape[0],train_idx))
time.sleep(3)

    

def test_data(data,test_idx,chord_data):

    #save the test data and train data separately
    X_te = []
    y_te = []

    for i in range(test_idx.shape[0]):
        # don't augment test data
        # for j in range(data_aug_count+1):
        stp = (test_idx[i])*8*(data_aug_count+1)  # no augmentation but still needs to skip those examples
        edp = stp + 8
        print("Test idx:", train_idx[i], stp, edp)
        song = data[stp:edp,0,:,:]
        song = song.reshape((8,1,128,16))   # modification: divisions per bar to 16
        # song = np.transpose(song, (0,1,3,2))  # do a transpose step here to fit data
        X_te.append(song)
        # added
        chords = chord_data[stp:edp,0,:]
        chords = chords.reshape((8,1,13,1))
        y_te.append(chords)
        # print('i: {}, test_iex: {}, stp: {}, song.shape: {}, song num: {}'.format(i, test_idx[i], stp, song.shape, len(X_te)))
        
    X_te = np.vstack(X_te)
    y_te = np.vstack(y_te)
    return X_te, y_te


def train_data(data,train_idx,chord_data):

    #save the test data and train data separately
    X_tr = []
    y_tr = []

    for i in range(train_idx.shape[0]):
        for j in range(data_aug_count+1):
            stp = (train_idx[i])*8*(data_aug_count+1) + j*8
            edp = stp + 8
            # print("Train idx:", train_idx[i], stp, edp)
            song = data[stp:edp,0,:,:]
            song = song.reshape((8,1,128,16))   # modification: pitch dimension is now 64 instead of 128
            # song = np.transpose(song, (0,1,3,2))  # do a transpose step here to fit data
            X_tr.append(song)
            # added
            chords = chord_data[stp:edp,0,:]
            chords = chords.reshape((8,1,13,1))  # reshape
            y_tr.append(chords)

        # print('i: {}, train_iex: {}, stp: {}, song.shape: {}, song num: {}'.format(i, train_idx[i], stp, song.shape, len(X_tr)))

    X_tr = np.vstack(X_tr)
    y_tr = np.vstack(y_tr)
    return X_tr, y_tr



# test_data
X_te, y_te = test_data(data,test_idx,chord_data) # just added chord_data_into_this
prev_X_te, _ = test_data(prev_data,test_idx,chord_data)
np.save('data/data_X_te.npy',X_te)
np.save('data/prev_X_te.npy',prev_X_te)

np.save('data/data_y_te.npy',y_te)

print('test song completed, X_te matrix shape: {}, y_te matrix shape:{}'.format(X_te.shape, y_te.shape))


#train_data
X_tr, y_tr = train_data(data,train_idx,chord_data)
prev_X_tr, _ = train_data(prev_data,train_idx,chord_data)
np.save('data/data_X_tr.npy',X_tr)
np.save('data/prev_X_tr.npy',prev_X_tr)

np.save('data/data_y_tr.npy',y_tr)

print('train song completed, X_tr matrix shape: {}, y_tr matrix shape:{}'.format(X_tr.shape, y_tr.shape))





