import pretty_midi
import music21
import glob


def group_notes_by_bars(notes, bar_duration):
    bars = [[],[],[],[],[],[],[],[]]  # 8 bars explicitly
    # current_bar = []
    bar_index = 0
    
    for note in notes:
        bar_index = int(note.start / 2)
        bars[bar_index].append(note)
        
        # special case for long notes
        if note.end - note.start > 2 and bar_index < 7:
            bars[bar_index+1].append(note)
        
    return bars


def identify_chord(notes):
    stream = music21.stream.Stream()
    for note in notes:
        m21_note = music21.note.Note(note.pitch)
        stream.append(m21_note)
    chord = stream.chordify().chordify()
    return chord


def is_note_in_chord(note, chord):
    note_pitch_class = note.pitch % 12
    return any(p.midi % 12 == note_pitch_class for p in chord.pitches)


def analyze_chord_tone_ratio(filename):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(filename)

    # Assuming the first instrument is the upper staff and the second is the lower staff
    upper_staff = midi_data.instruments[0]
    lower_staff = midi_data.instruments[1]

    # Extract notes from each staff
    upper_notes = upper_staff.notes
    lower_notes = lower_staff.notes

    # Define the bar duration (assuming 4/4 time signature and tempo of 120 BPM)
    bar_duration = 2.0  # in seconds (4 beats * 0.5 seconds per beat)

    upper_bars = group_notes_by_bars(upper_notes, bar_duration)
    lower_bars = group_notes_by_bars(lower_notes, bar_duration)

    lower_chords = [identify_chord(bar) for bar in lower_bars]
    # print(len(lower_chords), "!")

    note_count = 0
    note_count_first_beat = 0
    part_of_chord = 0
    part_of_chord_first_beat = 0

    for bar_index, upper_bar in enumerate(upper_bars):
        # print(bar_index)
        if bar_index >= 8:
            break  # ugly but needed so that anything beyond 8th bar is not processed
        
        lower_chord = lower_chords[bar_index]
        for note in upper_bar:
            is_first_beat = False
            note_count += 1
            if note.start % 2 < 0.5:
                # print("  FIRST BEAT")
                note_count_first_beat += 1
                is_first_beat = True
            if is_note_in_chord(note, lower_chord):
                part_of_chord += 1
                if is_first_beat:
                    part_of_chord_first_beat += 1
                # print(f"Note {note} is part of chord {[str(p) for p in lower_chord.pitches]}")
            else:
                pass
                # print(f"Note {note} is NOT part of chord {[str(p) for p in lower_chord.pitches]}")

    print("Total notes: {}, Part of Chord: {} ({:.4f}%), First Beat Notes: {}, First Beat is Part of a Chord: {} ({:.4f}%) ".format(
        note_count, part_of_chord, part_of_chord/note_count * 100, 
        note_count_first_beat, part_of_chord_first_beat, part_of_chord_first_beat/note_count_first_beat*100))
    
    return note_count, part_of_chord, note_count_first_beat, part_of_chord_first_beat



def main():
    # midi_files = glob.glob('samples/*.mid')  # generated samples path (replace it with the folder)
    midi_files = glob.glob('midi_dataset_segmented/*.mid')  # dataset path
    for file in midi_files:
        print(file)
        analyze_chord_tone_ratio(file)



if __name__ == "__main__":
    main()

