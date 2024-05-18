import numpy as np
from pypianoroll import Multitrack, Track
import matplotlib
matplotlib.use('Agg')
import datetime
import os
import pretty_midi

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
    c_maj  = [60,64,67,70]
    c_min  = [60,63,67,70]
    chord_map = []
    chord_list_maj = []
    chord_list_min = []
    chord_list_maj.append(c_maj)
    chord_list_min.append(c_min)
    for i in range(11):
        chord = [(x+1)%12 + 60 for x in c_maj]   # wrapping
        c_maj = chord
        chord_list_maj.append(chord)
        chord_min = [(x+1)%12 + 60 for x in c_min]  # wrapping
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

def jazzify(file, drum_midi):
    # load track, create output track
    original_midi = pretty_midi.PrettyMIDI(file)
    output_midi = pretty_midi.PrettyMIDI()
    
    # copy melody track from original MIDI
    melody_instrument = original_midi.instruments[0]
    melody_instrument.program = pretty_midi.instrument_name_to_program('Acoustic Bass')
    for note in melody_instrument.notes:
            note.pitch -= 12
            note.start = int(note.start * 6) / 6 
    # print(melody_instrument.program)
    output_midi.instruments.append(melody_instrument)
    
    # chord track
    chord_instrument = original_midi.instruments[1]
    chord_instrument.program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    
    chords = [[], [], [], [], [], [], [], []] # 8 bars always
    for note in chord_instrument.notes:
        bar_start = int(note.start / 2)
        bar_end = int(note.end / 2)
        for i in range(bar_start, bar_end):
            chords[i].append(note)
    
    # arbirtrary setup for the chords played by piano
    chords_timing = [[(0., 0.63), (0.83, 1.)],
                     [(0.33, 0.5), (1.0, 1.5)],
                     [(0.33, 0.5), (0.83, 1.5)],
                     [(0.0, 0.33), (0.33, 0.5), (0.83, 1.)]
                    ]
    
    # convert chords into piano track
    new_piano_instrument = pretty_midi.Instrument(program=0, name='Acoustic Grand Piano')
    for bar_num, chord in enumerate(chords):  # bar number counts from 0
        for note in chord:
            for timing in chords_timing[bar_num%4]:
                chord_note = pretty_midi.Note(
                    velocity=64, pitch=note.pitch, start=timing[0]+bar_num*2, end=timing[1]+bar_num*2)
                new_piano_instrument.notes.append(chord_note)
    # print(new_piano_instrument.notes)
    # print(new_piano_instrument.program)
    
    # add drums
    drum_instrument = drum_midi.instruments[0]
    drum_instrument.is_drum = True

    output_midi.instruments.append(new_piano_instrument)
    output_midi.instruments.append(drum_instrument)
    # print(drum_instrument.program)
    
    return output_midi


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
    export_jazzified = 1 # export the files with jazzified output closer to real performance
    
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

        # print(str(sample_name))
        #print(str(instrument))
        #print(str(volume))
        now = datetime.datetime.now()

        if handle_dataset:
            midi_export_filename = 'midi_dataset_segmented/dataset_'+str(sample_name)+'.mid'
        else: 
            midi_export_filename = 'samples/id_'+str(model_id)+'_generated_file'+str(sample_name)+"-"+str(now.strftime("%Y%m%d-%H"))+'h.mid'

        multitrack.write(midi_export_filename)


        # from the file we then convert that into 
        if export_jazzified:
            # load the drum file
            drum_midi = pretty_midi.PrettyMIDI('MIDI_utils/drum_track_MIDI.mid', initial_tempo=120.0)
            jazzified_track = jazzify(midi_export_filename, drum_midi)

            # save file
            if not os.path.exists('samples/jazzified/'):
                os.makedirs('samples/jazzified/')
            jazzified_filename = 'samples/jazzified/jazzified_'+ os.path.split(midi_export_filename)[-1]
            jazzified_track.write(jazzified_filename)

        if i % 100 == 0:
            print(str(sample_name)+' saved')

    print("Saving complete.")


if __name__ == "__main__" :

    main()









