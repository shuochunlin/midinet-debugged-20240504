import xml.etree.ElementTree as ET 
import xmldataset
import os 
from os.path import basename, dirname, join, exists, splitext
import ipdb
import numpy as np


def get_sample(cur_song, cur_dur,n_ratio, dim_pitch, dim_bar, start_pitch=0, wrapback=128):

    cur_bar =np.zeros((1,dim_pitch,dim_bar),dtype=int)
    idx = 1
    sd = 0
    ed = 0
    song_sample=[]
    
    # modification: take only 36 ~ 84 range is enough
    
    while idx < len(cur_song):
        initial_pitch = cur_song[idx]-1

        # special case for chords
        is_minor_chord = 0
        if wrapback == 12 and initial_pitch >= 12:
            is_minor_chord = 1

        cur_pitch = (cur_song[idx]-1 - start_pitch) % wrapback

        
        # pre-processing already got rid of triplets so don't worry
        ed = int(ed + cur_dur[idx]*n_ratio//1)
        
        # print('pitch: {}, sd:{}, ed:{}'.format(cur_pitch, sd, ed))
        if ed <dim_bar:
            # draws piano roll
            cur_bar[0,cur_pitch,sd:ed]=1

            # special case for minor_chords
            if is_minor_chord:
                cur_bar[0, 12, sd:ed]=1   # code the minor chord to 1 to match author's setup
                # print("Minor_chord_applied")
            
            sd = ed
            idx = idx +1
        elif ed >= dim_bar:
            cur_bar[0,cur_pitch,sd:]=1

            # special case for minor_chords
            if is_minor_chord:
                cur_bar[0, 12, sd:]=1   # code the minor chord to 1 to match author's setup
                # print("Minor_chord_applied")

            song_sample.append(cur_bar)
            cur_bar =np.zeros((1,dim_pitch,dim_bar),dtype=int)
            sd = 0
            ed = 0
            # print(cur_bar)
            # print(song_sample)
        # if idx == len(cur_song)-1 and np.sum(cur_bar)!=0:
        #     song_sample.append(cur_bar)
    # print("Size:", np.shape(song_sample))

    # remove the extra 64 notes we just wouldn't use
    # range is now 24 ~ 72
    return song_sample

def build_matrix(note_list_all_c,dur_list_all_c):
    data_x = []           
    prev_x = []
    zero_counter = 0

    data_aug_count = 8
    for i in range(len(note_list_all_c)):
        
        song = note_list_all_c[i]
        dur = dur_list_all_c[i]

        # create data_augmentation
        # data shifts semitone up each time
        for j in range(data_aug_count+1):

            # song, dur, n_ratio (divisions per beat), dim_pitch, dim_bar (divisions per bar * 8)
            song_sample = get_sample(song,dur,4,128,128, start_pitch= -j, wrapback=128)   # j = 0; j = -1 (using offset hack)
            np_sample = np.asarray(song_sample)    # may try the split of 64 (4 bars) to see if it improves
            if len(np_sample) == 0:
                zero_counter +=1
            if len(np_sample) != 0:
                np_sample =np_sample[0]
                np_sample = np_sample.reshape(1,1,128,128) # modified to 96 from 128

                # since we're dealing with swing and triples, dimensions set to 12

                if np.sum(np_sample) != 0:
                    place = np_sample.shape[3]
                    new=[]
                    for i in range(0,place,16):  
                        new.append(np_sample[0][:,:,i:i+16])
                    new = np.asarray(new)  # (2,1,128,128) will become (16,1,128,16)
                    new_prev = np.zeros(new.shape,dtype=int)
                    new_prev[1:, :, :, :] = new[0:new.shape[0]-1, :, :, :]            
                    data_x.append(new)
                    prev_x.append(new_prev)  

    data_x = np.vstack(data_x)
    prev_x = np.vstack(prev_x)


    return data_x,prev_x,zero_counter

def build_chord_matrix(chord_list_all_c,cdur_list_all_c):
    data_y = []
    data_aug_count = 8
    for i in range(len(chord_list_all_c)):
        chords = chord_list_all_c[i]
        dur = cdur_list_all_c[i]

        for j in range(data_aug_count+1):

            song_sample = get_sample(chords,dur, 0.25, 13, 8, start_pitch= -j, wrapback=12)
            np_sample = np.asarray(song_sample)
            # if len(np_sample) == 0:
            #     zero_counter +=1
            if len(np_sample) != 0:
                np_sample =np_sample[0]
                np_sample = np_sample.reshape(1,1,13,8)
                # print(np_sample)

                if np.sum(np_sample) != 0:
                    place = np_sample.shape[3]
                    new=[]
                    for i in range(0,place,1):
                        new.append(np_sample[0][:,:,i])
                    new = np.asarray(new)  # (2,1,13,8) will become (16,1,13,1)
                    #new_prev = np.zeros(new.shape,dtype=int)
                    #new_prev[1:, :, :, :] = new[0:new.shape[0]-1, :, :, :]            
                    data_y.append(new)
                    #prev_x.append(new_prev) 
                    # print("NEW:", new) 

    data_y = np.vstack(data_y)
    return data_y

def check_melody_range(note_list_all,dur_list_all):
    in_range=0
    note_list_all_c = []
    dur_list_all_c = []
    
    for i in range(len(note_list_all)):
        song = note_list_all[i]
        if len(song[1:]) ==0:
            ipdb.set_trace()
        elif min(song[1:])>= 60 and max(song[1:])<= 83:
            in_range +=1
            note_list_all_c.append(song)
            dur_list_all_c.append(dur_list_all[i])
    np.save('dur_list_all_c.npy',dur_list_all_c)
    np.save('note_list_all_c.npy',note_list_all_c)

    return in_range,note_list_all_c,dur_list_all_c

def transform_note(c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list):#,db_key_list,eb_key_list,gb_key_list,ab_key_list,bb_key_list):
    scale = [48,50,52,53,55,57,59,60,62,64,65,67,69,71,72,74,76,77,79,81,83,84,86,88,89,91,93]
    transfor_list_C1 = scale[0:7]
    transfor_list_C2 = scale[7:14]
    transfor_list_C3 = scale[14:21]

    transfor_list_D1 = scale[1:8]
    transfor_list_D2 = scale[8:15]
    transfor_list_D3 = scale[15:22]

    transfor_list_E1 = scale[2:9]
    transfor_list_E2 = scale[9:16]
    transfor_list_E3 = scale[16:23]

    transfor_list_F1 = scale[3:10]
    transfor_list_F2 = scale[10:17]
    transfor_list_F3 = scale[17:24]

    transfor_list_G1 = scale[4:11]
    transfor_list_G2 = scale[11:18]
    transfor_list_G3 = scale[18:25]

    transfor_list_A1 = scale[5:12]
    transfor_list_A2 = scale[12:19]
    transfor_list_A3 = scale[19:26]

    transfor_list_B1 = scale[6:13]
    transfor_list_B2 = scale[13:20]
    transfor_list_B3 = scale[20:27]

    note_c =[]  
    dur_c =[]
    for file_ in c_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_C1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_C2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_C3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_c.append(note_list)
            dur_c.append(dur_list)

        except:
            print('c key but no melody/notes :{}'.format(file_))

    note_d = []
    dur_d = []
    for file_ in d_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_D1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_D2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_D3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_d.append(note_list)
            dur_d.append(dur_list)

        except:
            print('d key but no melody/notes :{}'.format(file_))

    note_e = []
    dur_e = []
    for file_ in e_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_E1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_E2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_E3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_e.append(note_list)
            dur_e.append(dur_list)

        except:
            print('e key but no melody/notes :{}'.format(file_))

    note_f = []
    dur_f = []
    for file_ in e_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_F1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_F2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_F3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_f.append(note_list)
            dur_f.append(dur_list)

        except:
            print('f key but no melody/notes :{}'.format(file_))


    note_g = []
    dur_g = []
    for file_ in a_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_G1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_G2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_G3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_g.append(note_list)
            dur_g.append(dur_list)

        except:
            print('g key but no melody/notes :{}'.format(file_))

    note_a = []
    dur_a = []
    for file_ in a_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_A1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_A2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_A3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_a.append(note_list)
            dur_a.append(dur_list)

        except:
            print('e key but no melody/notes :{}'.format(file_))


    note_b = []
    dur_b = []
    for file_ in a_key_list:
        note_list = [file_]
        dur_list = [file_]  
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()

            for item in root.iter(tag='note'):
                note = item[4].text
                dur = item[3].text
                octave = item[5].text
                dur = float(dur)
                dur_list.append(dur)

                try:
                    note = int(note)
                    if octave == '-1':
                        h_idx = transfor_list_A1[note-1]
                    elif octave == '0':
                        h_idx = transfor_list_A2[note-1]
                    elif octave == '1':
                        h_idx = transfor_list_A3[note-1]        
                    note_list.append(h_idx)
                    
                except:
                    if len(note_list)==1:
                        note = 0
                        note_list.append(note)

                    else:
                        note = note_list[-1]
                        note_list.append(note)

            if note_list[1]== 0:
                note_list[1] = note_list[2]
                dur_list[1] = dur_list[2]

            note_b.append(note_list)
            dur_b.append(dur_list)

        except:
            print('b key but no melody/notes :{}'.format(file_))
   

    note_list_all = note_c + note_d + note_e + note_f + note_g + note_a + note_b
    dur_list_all = dur_c + dur_d + dur_e  + dur_f + dur_g + dur_a  + dur_b

    return note_list_all,dur_list_all

def get_key(list_of_four_beat):
    key_list =[]
    c_key_list = []
    d_key_list = []
    e_key_list = []
    f_key_list = []
    g_key_list = []
    a_key_list = []
    b_key_list = []
    
    db_key_list = []
    eb_key_list = []
    gb_key_list = []
    ab_key_list = []
    bb_key_list = []
    for file_ in list_of_four_beat:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            key = root.findall('.//key')
            key_list.append(key[0].text)
            if key[0].text == 'C':
                c_key_list.append(file_)
            if key[0].text == 'D':
                d_key_list.append(file_)
            if key[0].text == 'E':
                e_key_list.append(file_) 
            if key[0].text == 'F':
                f_key_list.append(file_)
            if key[0].text == 'G':
                g_key_list.append(file_) 
            if key[0].text == 'A':
                a_key_list.append(file_)  
            if key[0].text == 'B':
                b_key_list.append(file_)  



            # key_fifth = root.findall('.//fifths') # root.findall('.//key')

            # a chart for circle of fifths
            # surprising only accounted for 7 keys when there's 12 technically
            # added this part to fix
            """
            key = None
            if key_fifth[0].text == '0':
                key = 'C'
            if key_fifth[0].text == '1':
                key = 'G'
            if key_fifth[0].text == '2':
                key = 'D'
            if key_fifth[0].text == '3':
                key = 'A'
            if key_fifth[0].text == '4':
                key = 'E'
            if key_fifth[0].text == '5':
                key = 'B'
            if key_fifth[0].text == '6' or key_fifth[0].text == '-6':
                key = 'Gb'
            if key_fifth[0].text == '-5':
                key = 'Db'
            if key_fifth[0].text == '-4':
                key = 'Ab'
            if key_fifth[0].text == '-3':
                key = 'Eb'
            if key_fifth[0].text == '-2':
                key = 'Bb'
            if key_fifth[0].text == '-1':
                key = 'F'

            # and now back to author's original code
            
            key_list.append(key)
            if key == 'C':
                c_key_list.append(file_)
            if key == 'D':
                d_key_list.append(file_)
            if key == 'E':
                e_key_list.append(file_) 
            if key == 'F':
                f_key_list.append(file_)
            if key == 'G':
                g_key_list.append(file_) 
            if key == 'A':
                a_key_list.append(file_)  
            if key == 'B':
                b_key_list.append(file_)       

            if key == 'Db':
                db_key_list.append(file_)
            if key == 'Eb':
                eb_key_list.append(file_) 
            if key == 'Gb':
                gb_key_list.append(file_)
            if key == 'Ab':
                ab_key_list.append(file_) 
            if key == 'Bb':
                bb_key_list.append(file_)  
                """
        except:
            print('file broken')
    # print('A key: {}'.format(key_list.count('A')))
    # print('B key: {}'.format(key_list.count('B')))
    # print('C key: {}'.format(key_list.count('C')))
    # print('D key: {}'.format(key_list.count('D')))
    # print('E key: {}'.format(key_list.count('E')))
    # print('F key: {}'.format(key_list.count('F')))
    # print('G key: {}'.format(key_list.count('G')))
    
    # print('Db key: {}'.format(key_list.count('Db')))
    # print('Eb key: {}'.format(key_list.count('Eb')))
    # print('Gb key: {}'.format(key_list.count('Gb')))
    # print('Ab key: {}'.format(key_list.count('Ab')))
    # print('Bb key: {}'.format(key_list.count('Bb')))

    return c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list#, \
           #db_key_list,eb_key_list,gb_key_list,ab_key_list,bb_key_list

def beats_(list_):
    list_of_four_beat =[]
    for file_ in list_:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            beats = root.findall('.//beats_in_measure')
            print(beats)
            num = beats[0].text
            if num == '4':
                list_of_four_beat.append(file_) 
        except:
            print('cannot open the file')
    return list_of_four_beat

def check_chord_type(list_file):
    list_ = []
    for file_ in list_file:
        try:
            chorus_file = ET.parse(file_)
            root = chorus_file.getroot()
            check_list = []
            counter = 0
            None_counter = 0
            for item in root.iter(tag='fb'):
                check_list.append(item.text)
                counter +=1
                if item.text == None:
                    None_counter +=1
            for item in root.iter(tag='borrowed'):
                check_list.append(item.text)
                counter +=1
                if item.text == None:
                    None_counter +=1
            #print(check_list)
            #print(counter)
            #print(None_counter)
            if counter == None_counter :
                list_.append(file_)
        except:
            print('cannot open')
    return list_

def get_listfile(dataset_path):

    list_file=[]
    print(dataset_path)

    for root, dirs, files in os.walk(dataset_path):    
        for f in files:
            # if splitext(f)[0]=='chorus':                
            fp = join(root, f)
            list_file.append(fp)

    return list_file

def main():
    is_get_data = 0
    is_get_matrix = 1
    if is_get_data == 1:
        a = 'Theorytab_xml/'
        list_file = get_listfile(a)

        # the following step filters out songs that are not 4 beats, but we don't need this
        list_ = check_chord_type(list_file)
        #print("List of files:", list_)
        list_of_four_beat = beats_(list_)
        # list_of_four_beat = list_file
        
        print("List of four_beat:", list_of_four_beat)
        c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list= get_key(list_of_four_beat) #,db_key_list,eb_key_list,gb_key_list,ab_key_list,bb_key_list = get_key(list_of_four_beat)
        note_list_all,dur_list_all = transform_note(c_key_list,d_key_list,e_key_list,f_key_list,g_key_list,a_key_list,b_key_list)#,db_key_list,eb_key_list,gb_key_list,ab_key_list,bb_key_list)
        in_range,note_list_all_c,dur_list_all_c = check_melody_range(note_list_all,dur_list_all)
        # print('total normal chord: {}'.format(len(list_)))
        print('total in four: {}'.format(len(list_of_four_beat)))
        print('melody in range: {}'.format(len(note_list_all)))

    if is_get_matrix == 1:
        note_list_all_c = np.load('note_list_all_c.npy')
        dur_list_all_c = np.load('dur_list_all_c.npy')

        chord_list_all_c = np.load('chord_list_all_c.npy')
        cdur_list_all_c = np.load('cdur_list_all_c.npy')

        print(chord_list_all_c)
        print(cdur_list_all_c)

        data_x, prev_x, zero_counter = build_matrix(note_list_all_c,dur_list_all_c)
        data_y = build_chord_matrix(chord_list_all_c,cdur_list_all_c)
        np.save('data_x.npy',data_x)
        np.save('prev_x.npy',prev_x)
        np.save('data_y.npy',data_y)

        print('final tab num: {}'.format(len(note_list_all_c)))
        print('songs not long enough: {}'.format(zero_counter))
        print('sample shape: {}, prev sample shape: {}'.format(data_x.shape, prev_x.shape))
        print('chords sample shape: {}'.format(data_y.shape))
    
if __name__ == "__main__" :

    main()
