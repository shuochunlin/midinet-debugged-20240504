import numpy as np
from pypianoroll import Multitrack, Track
import matplotlib
matplotlib.use('Agg')
import datetime


def find_pitch(song,volume=40):   # song shape(128,128), which is (time step, pitch)
    for time in range(song.shape[0]):
        step = song[time,:]
        max_index = np.argmax(step)
        for i in range(len(step)):
            if i ==max_index:
                song[time,i] = volume
            else:
                song[time,i] = 0
    return song

def reshape_bar(song):
    eight_bar = song[0]
    for i in range(7):
        b = song[i+1]
        eight_bar  = np.concatenate((eight_bar,b),axis=0)
    eight_bar = eight_bar.astype(float)
    #print("A bar's shape: {}".format(eight_bar.shape))
    return eight_bar

def make_a_track(eight_bar_binarized,track_name ='melody' ,instrument=0):
    track = Track(pianoroll=eight_bar_binarized, program=instrument, is_drum=False,name=track_name)
    return track

def make_a_demo(track1,track2,song_idx):
    sample_name = 'sample_'+str(song_idx)

    multitrack = Multitrack(tracks=[track1.standardize(),track2.standardize()], tempo=np.array([[120.0]]),resolution=4)
    # pypiano.plot(multitrack, filepath='your file situation', mode='separate', preset='default', cmaps=None, xtick='auto', ytick='octave', xticklabel=True, yticklabel='auto', tick_loc=None, tick_direction='in', label='both', grid='both', grid_linestyle=':', grid_linewidth=0.5)
    # plt.savefig('your file situation'+sample_name+'.png')
    return sample_name, multitrack


def chord_list(chord,idx):

    one_song_chord = chord[idx]
    song_chord = []
    for i in range(len(one_song_chord)):
        bar_idx = []
        one_bar_chord = one_song_chord[i]
        bar_idx.append(int(one_bar_chord[0][12]))
        max_idx = np.argmax(one_bar_chord[:11])
        bar_idx.append(max_idx)
        song_chord.append(bar_idx)
    return song_chord


def build_chord_map():
    c_maj  = [60,64,67]
    c_min  = [60,63,67]
    chord_map = []
    chord_list_maj = []
    chord_list_min = []
    chord_list_maj.append(c_maj)
    chord_list_min.append(c_min)
    for i in range(11):
        chord = [x+1 for x in c_maj] 
        c_maj = chord
        chord_list_maj.append(chord)
        chord_min = [x+1 for x in c_min]
        chord_list_min.append(chord_min)
        c_min = chord_min
    chord_map.append(chord_list_maj)
    
    # our encoding is different and so we just pass chord_list_min as normal - commented out
    # chord_list_min[:] = chord_list_min[9:] + chord_list_min[0:9]
    chord_map.append(chord_list_min)
    return chord_map

def decode_chord(maj_min,which_chord):

    chord_map = build_chord_map()
    chord = chord_map[maj_min][which_chord]

    return chord

def get_chord(song_chord):
    chord_player = []
    for item in song_chord:
        maj_min = item[0]
        which_chord = item[1]
        answer_chord = decode_chord(maj_min,which_chord)
        chord_player.append(answer_chord)
    return chord_player

def make_chord_track(chord,instrument,volume=40):
    pianoroll = np.zeros((128, 128))
    for i in range(len(chord)):
        st = 16*i
        ed = st + 16
        chord_pitch = chord[i]
        pianoroll[st:ed, chord_pitch] = volume
    track = Track(pianoroll=pianoroll, program=instrument, is_drum=False,
                  name='chord')
    return track



def main():
    d_data = np.load('data_x.npy', allow_pickle=True)
    d_chord = np.load('data_y.npy', allow_pickle=True)

    model_id = input("Current Model's ID") 

    data = np.load(f'{model_id}_output_songs.npy', allow_pickle=True)
    chord = np.load(f'{model_id}_output_chords.npy', allow_pickle=True)

    print(np.shape(d_data), np.shape(d_chord),np.shape(data),np.shape(chord))
    d_data = d_data.reshape((-1, 8, 128, 16))
    d_chord = d_chord.reshape((-1, 8, 1, 13))
    print(np.shape(d_data), np.shape(d_chord),np.shape(data),np.shape(chord))
    
    instrument = 0 # int(input('which instrument you want to play? from 0 to 128,default=0:'))  # enter 0 for bass
    volume     = 100 # int(input('how loud you want to play? from 1 to 127,default= 40:'))

    handle_dataset = 0 # int(input('handling dataset? 0 for samples, 1 for dataset'))
    
    if handle_dataset:
        data = np.transpose(d_data, (0, 1, 3, 2))
        chord = d_chord

    for i in range(data.shape[0]):
        one_song = data[i]
        song = []
        for item in one_song:
            item = item #.detach().numpy()
            # print(np.shape(item))
            item = item.reshape(16,128)
            song.append(item)
        eight_bar = reshape_bar(song)
        eight_bar_binarized = find_pitch(eight_bar,volume)
        track = make_a_track(eight_bar_binarized,instrument)
        
        song_chord = chord_list(chord,i)
        chord_player = get_chord(song_chord)
        # np.save('file/chord_'+str(i)+'.npy',chord_player)
        chord_track = make_chord_track(chord_player,instrument,volume)
        sample_name, multitrack = make_a_demo(track,chord_track,i)

        print(str(sample_name))
        #print(str(instrument))
        #print(str(volume))
        now = datetime.datetime.now()

        if handle_dataset:
            multitrack.write('midi_dataset_segmented/dataset_'+str(sample_name)+'.mid')
        else:
            multitrack.write('samples/generated_file'+str(sample_name)+"-"+str(now.strftime("%Y%m%d-%H"))+'h.mid')
        if i % 100 == 0:
            print(str(sample_name)+'saved')



if __name__ == "__main__" :

    main()









