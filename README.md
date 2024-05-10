

# midinet-debugged-20240504
This is the edited version of [Debugged MidiNet](https://github.com/dongmingli-Ben/MidiNet-by-pytorch-Debugged), which is a debugged implementation of [Implementation of MidiNet by pytorch](https://github.com/annahung31/MidiNet-by-pytorch).

MidiNet paper : https://arxiv.org/abs/1703.10847 

MidiNet code  : https://github.com/RichardYang40148/MidiNet 

Instead of training on melody, we use the FiloBass dataset to train basslines.

Dataset       : https://aim-qmul.github.io/FiloBass/ 

--------------------------------------------------------------------------------------------------
## Prepare the data (Optional)

This step is not necessary as dataset has be prepared in "./data" folder. But just in case.

This step converts xml files into training data in npy files format.

*(to be documented)*

|file to run                     |  function|
|-|-|
|process_notes_and_chords.py     |  (not yet included) \[NEW\]: preprocessing xml files into csv files, with melody and chords quantization. The pre-processing steps for process_notes_and_chords.py requires music21 library. Showing the scores "score.show()" require MuseScore installed. If not desired, simply uncomment them.|
|csv_to_npy.py                   |  (not yet included) \[NEW\]: convert into npy files, while converting chord labels to a range 0 ~ 25 accepted by the next step|
|get_data.py                     |  Preprocessing. Reusing the portion of code that converts note list, duration list (npy format) into melody and chord matrix. Contains data augmentation. Due to previous steps used, only the get matrix part is used.|
|get_train_and_test_data.py      |  seperate the melody data into training set and testing set. Data augmentation is taken into account. I added chord handling by duplicating and re-using the original function.|

The current files in "data" folder are augmented into 8 out of 12 keys, split into training and testing set 9:1. 128 pitch_dim, 16 subdivisions per bar.

If the pre-processing is run, the following files would be created during the process:
|npy files                     |  function|
|-|-|
|note_list_all_c.npy             |  Contains all pitches in order (list), extracted from XML files of the dataset|
|dur_list_all_c.npy              |  Durations of pitches (list) extracted from XML files of the dataset|
|chord_list_all_c.npy            |  Contains all chords in order (list), extracted from XML files of the dataset|
|cdur_list_all_c.npy             |  Durations of chords (list) extracted from XML files of the dataset|
|data_x.npy                      |  Our music data content in piano roll format|
|data_y.npy                      |  Our music labels (chords)  |
|prev_x.npy                      |  Previous bar (used with data_x.npy) of the current bar asked to be generated|

--------------------------------------------------------------------------------------------------
## Training the model

After preprocessing the data (placed in ```data``` folder), there should be 6 files:
* data_X_tr.npy (training data - current bar melody)
* data_y_tr.npy (training data - chord labels)
* prev_X_tr.npy (training data - previous bar melody)
* data_X_te.npy (testing data - current bar melody)
* data_y_te.npy (testing data - chord labels)
* prev_X_te.npy (testin data - previous bar melody)

The following a relevant files for training / testing the model:

|file                  |  purposes|
|-|-|
|requirement.txt                 |  toolkits used in the whole work|
|main.py                         |  training setting, drawing setting, generation setting.|
|ops.py                          |  some functions used in model|
|model.py                        |  Generator and Discriminator.   (Based on model 3 in the MidiNet paper)|
|demo.py                         |  transform matrix into midi. (input : melody and chord matrix, output : midi)|

Generally the steps are:
1. Make sure there are toolkits in the requirement.txt installed
2. Edit the file in main.py.
  is_train = 1 for training,  
  is_draw = 1 for drawing loss,  
  is_sample = 1 for generating music after finishing training.
  The sampling function uses 1 bar of testing data as prompt, and the model generates the next 7 measures.
3. Run main.py.

After running the model the following files would be created:
|files                     |  function|
|-|-|
|"/file/*.png"              | saves fake samples and real samples as png files|
|"/draw_figs/lr*_epoch*.png"| saves a figure of model's training progression|
|"/models/*.pth"            | saves model files (note author did not implement restore training from an epoch)|
|lossD_list.npy             |  Generated after is_train = 1. Used to draw loss during is_draw = 1.|
|lossG_list.npy             |  Generated after is_train = 1. Used to draw loss during is_draw = 1.|
|lossD_list_all.npy         |  Generated after is_train = 1. Used to draw loss during is_draw = 1.|
|lossG_list_all.npy         |  Generated after is_train = 1. Used to draw loss during is_draw = 1.|
|D_x_list.npy               |  Generated after is_train = 1. Used to draw loss during is_draw = 1.|
|D_G_z_list.npy             |  Generated after is_train = 1. Used to draw loss during is_draw = 1.|
|output_songs.npy           |  Generated after is_sample = 1. These will be read by demo.py later.|
|output_chords.npy          |  Generated after is_sample = 1. These will be read by demo.py later.|

--------------------------------------------------------------------------------------------------

## Listening to music samples

To convert music samples into MIDI, Run demo.py.
Type in the desired instrument (default = 0 for piano), and the volume (default 40). These are only for playback purposes.

(Note: setting instrument values other than 0 creates bugs for some reason)

After running the files the samples are stored in "/samples" folder, in MIDI file format. In case higher music quality demo is desired, music software would be needed to further process MIDI files into mp3 files.

