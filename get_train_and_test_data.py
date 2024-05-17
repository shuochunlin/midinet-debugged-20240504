import numpy as np
import random
import math
import time

# Note the number of augmentations taken prior, has to match get_data.py
data_aug_count = 0  # number of pitch shifts done. 0 for no augmentation. 12 for all 12 keys in music.

#load data

data = np.load('data_x.npy')   # size is (training_samples, 1, 128, 16), where 128: midi notes, 16: timestep length
prev_data = np.load('prev_x.npy')

song_lengths = np.load('song_lengths.npy')

chord_data = np.load('data_y.npy')

print("Data shape:", np.shape(data))
print("Prev shape:", np.shape(prev_data))

print("Chord shape:",np.shape(chord_data))
print("Song Lengths:", chord_data)
time.sleep(3)  # comment this out to skip waiting

# song indices hard coded to just the 47 songs we have (I removed one)

"""song_idx = 47  # int(data.shape[0]/8) // (data_aug_count + 1)
test_ratial = 0.1
test_song_num = round(song_idx*test_ratial)
train_song_num = (song_idx - test_song_num) // data_aug_count
print('total song number: {}'.format(song_idx))
print('number of test songs: {}, \nnumber of train songs: {}'.format(test_song_num,train_song_num))
time.sleep(3)"""

#create the song idx for test data

"""full = np.arange(song_idx)

test_idx= random.sample(range(0,full.shape[0]),test_song_num)
test_idx = np.asarray(test_idx)
print('total {} song idx for test: {}'.format(test_idx.shape[0],test_idx))
time.sleep(3)

#create the song idx for train data
train_idx = np.delete(full,test_idx)
print('total {} song idx for train: {}'.format(train_idx.shape[0],train_idx))
time.sleep(3)"""

### REWRITTEN PARTS
# Step 1: Create a list of song indices
start_indices = [0]
for length in song_lengths[:-1]:
    start_indices.append(start_indices[-1] + length ) #* data_aug_count)

# Get the total number of songs
num_songs = len(song_lengths)

# Step 2: Randomly select test songs
test_song_num = int(0.2 * num_songs)  # For example, 20% of the songs
test_idx = random.sample(range(num_songs), test_song_num)

# Step 3: Remove test songs from training set
train_idx = [i for i in range(num_songs) if i not in test_idx]

# Printing results
print('Total {} song indices for train: {}'.format(len(train_idx), train_idx))
print('Total {} song indices for test: {}'.format(len(test_idx), test_idx))

# Step 4: Create segment indices for training and testing sets
train_segments = []
test_segments = []

for idx in train_idx:
    start = start_indices[idx]
    end = start + song_lengths[idx]
    train_segments.extend(range(start, end))

for idx in test_idx:
    start = start_indices[idx]
    end = start + song_lengths[idx]
    test_segments.extend(range(start, end))

train_segments = np.array(train_segments)
test_segments = np.array(test_segments)

print('Total {} segments for train: {}'.format(len(train_segments), train_segments))
print('Total {} segments for test: {}'.format(len(test_segments), test_segments))

# time.sleep(15)
### REWRITTEN

def test_data(data,test_idx,chord_data):

    #save the test data and train data separately
    X_te = []
    y_te = []

    for i in range(test_idx.shape[0]):
        # don't augment test data
        stp = (test_idx[i])*8*(data_aug_count+1)  # no augmentation but still needs to skip those aug examples
        edp = stp + 8
        # print("Test idx:", test_idx[i], stp, edp)
        song = data[stp:edp,0,:,:]
        song = song.reshape((8,1,128,16))   # 8 bars, 1 track, 128 pitch range, 16 subdivisions per bar
        X_te.append(song)

        # added chord handling into original file
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
        for j in range(data_aug_count+1):  # augmentation
            stp = (train_idx[i])*8*(data_aug_count+1) + j*8  # reading original files and the augmented indices
            edp = stp + 8
            # print("Train idx:", train_idx[i], stp, edp)
            song = data[stp:edp,0,:,:]
            song = song.reshape((8,1,128,16)) 
            X_tr.append(song)
            # added chord handling into original file
            chords = chord_data[stp:edp,0,:]
            chords = chords.reshape((8,1,13,1))  # reshape
            y_tr.append(chords)

        # print('i: {}, train_iex: {}, stp: {}, song.shape: {}, song num: {}'.format(i, train_idx[i], stp, song.shape, len(X_tr)))

    X_tr = np.vstack(X_tr)
    y_tr = np.vstack(y_tr)
    return X_tr, y_tr



# test_data
X_te, y_te = test_data(data,test_segments,chord_data)
prev_X_te, _ = test_data(prev_data,test_segments,chord_data)
np.save('data/data_X_te.npy',X_te)
np.save('data/prev_X_te.npy',prev_X_te)

np.save('data/data_y_te.npy',y_te)

print('test song completed, X_te matrix shape: {}, y_te matrix shape:{}'.format(X_te.shape, y_te.shape))


#train_data
X_tr, y_tr = train_data(data,train_segments,chord_data)
prev_X_tr, _ = train_data(prev_data,train_segments,chord_data)
np.save('data/data_X_tr.npy',X_tr)
np.save('data/prev_X_tr.npy',prev_X_tr)

np.save('data/data_y_tr.npy',y_tr)

print('train song completed, X_tr matrix shape: {}, y_tr matrix shape:{}'.format(X_tr.shape, y_tr.shape))





