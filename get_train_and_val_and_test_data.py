import numpy as np
import random
import math
import time

# added handling for validation set

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


### In progress
total_songs = len(song_lengths)

# Determine the number of songs for each split
train_song_num = int(0.8 * total_songs)
val_song_num = int(0.1 * total_songs)
test_song_num = total_songs - train_song_num - val_song_num  # Ensures the total adds up correctly

# Generate an array of song indices
song_indices = np.arange(total_songs)

# Randomly shuffle the song indices
np.random.shuffle(song_indices)

# Split the indices into training, validation, and test sets
train_idx = song_indices[:train_song_num]
val_idx = song_indices[train_song_num:train_song_num + val_song_num]
test_idx = song_indices[train_song_num + val_song_num:]


print(f'Total songs: {total_songs}')
print(f'Training set size: {train_song_num} songs, indices: {train_idx}')
print(f'Validation set size: {val_song_num} songs, indices: {val_idx}')
print(f'Test set size: {test_song_num} songs, indices: {test_idx}')

# To get the start and end indices for each song based on segments
train_indices = []
val_indices = []
test_indices = []

current_index = 0
for i in range(total_songs):
    length = song_lengths[i]
    if i in train_idx:
        train_indices.extend(range(current_index, current_index + length))
    elif i in val_idx:
        val_indices.extend(range(current_index, current_index + length))
    elif i in test_idx:
        test_indices.extend(range(current_index, current_index + length))
    current_index += length

train_segments = np.array(train_indices)
val_segments = np.array(val_indices)
test_segments = np.array(test_indices)

np.random.shuffle(train_segments)
np.random.shuffle(val_segments)
# np.random.shuffle(test_segments)  # debatable if shuffling test idx is needed but hey it's an option

# print(f'Training indices: {train_indices}')
# print(f'Validation indices: {val_indices}')
# print(f'Test indices: {test_indices}')
    

def test_data(data,test_idx,chord_data):

    #save the test data and train data separately
    X_te = []
    y_te = []

    for i in range(test_idx.shape[0]):
        # don't augment test data
        # for j in range(data_aug_count+1):
        stp = (test_idx[i])*8*(data_aug_count+1)  # no augmentation but still needs to skip those examples
        edp = stp + 8
        print("Test idx:", test_idx[i], stp, edp)
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


def val_data(data,val_idx,chord_data):

    #save the test data and train data separately
    X_val = []
    y_val = []

    for i in range(val_idx.shape[0]):
        # don't augment test data
        # for j in range(data_aug_count+1):
        stp = (val_idx[i])*8*(data_aug_count+1)  # no augmentation but still needs to skip those examples
        edp = stp + 8
        # print("Val idx:", val_idx[i], stp, edp)
        song = data[stp:edp,0,:,:]
        song = song.reshape((8,1,128,16))   # modification: divisions per bar to 16
        # song = np.transpose(song, (0,1,3,2))  # do a transpose step here to fit data
        X_val.append(song)
        # added
        chords = chord_data[stp:edp,0,:]
        chords = chords.reshape((8,1,13,1))
        y_val.append(chords)
        # print('i: {}, test_iex: {}, stp: {}, song.shape: {}, song num: {}'.format(i, test_idx[i], stp, song.shape, len(X_te)))
        
    X_val = np.vstack(X_val)
    y_val = np.vstack(y_val)
    return X_val, y_val


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
X_te, y_te = test_data(data,test_segments,chord_data) # just added chord_data_into_this
prev_X_te, _ = test_data(prev_data,test_segments,chord_data)
np.save('data/train-val-test/data_X_te.npy',X_te)
np.save('data/train-val-test/prev_X_te.npy',prev_X_te)

np.save('data/train-val-test/data_y_te.npy',y_te)

print('test song completed, X_te matrix shape: {}, y_te matrix shape:{}'.format(X_te.shape, y_te.shape))



# val_data
X_val, y_val = train_data(data,val_segments,chord_data)  # with aug
prev_X_val, _ = train_data(prev_data,val_segments,chord_data)  # with aug
np.save('data/train-val-test/data_X_val.npy',X_val)
np.save('data/train-val-test/prev_X_val.npy',prev_X_val)

np.save('data/train-val-test/data_y_val.npy',y_val)

print('val song completed, X_val matrix shape: {}, y_val matrix shape:{}'.format(X_val.shape, y_val.shape))


#train_data
X_tr, y_tr = train_data(data,train_segments,chord_data)
prev_X_tr, _ = train_data(prev_data,train_segments,chord_data)

np.save('data/train-val-test/data_X_tr.npy',X_tr)
np.save('data/train-val-test/prev_X_tr.npy',prev_X_tr)

np.save('data/train-val-test/data_y_tr.npy',y_tr)

print('train song completed, X_tr matrix shape: {}, y_tr matrix shape:{}'.format(X_tr.shape, y_tr.shape))





