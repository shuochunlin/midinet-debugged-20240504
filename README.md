
This is the edited version of [Debugged MidiNet](https://github.com/dongmingli-Ben/MidiNet-by-pytorch-Debugged), which is a debugged implementation of [Implementation of MidiNet by pytorch](https://github.com/annahung31/MidiNet-by-pytorch).

MidiNet paper : https://arxiv.org/abs/1703.10847 

MidiNet code  : https://github.com/RichardYang40148/MidiNet 

Instead of training on melody, we use the FiloBass dataset to train basslines.

Dataset       : https://aim-qmul.github.io/FiloBass/ 

--------------------------------------------------------------------------------------------------
Prepare the data

This step is not necessary as dataset has be prepared in "./data" folder. But in case 

|file to run                     |  function|
|-|-|
|process_notes_and_chords.py     |  \[NEW\]: preprocessing xml files into csv files, with melody and chords quantization|
|csv_to_npy.py                   |  \[NEW\]: convert into npy files|
|get_data.py                     |  reusing the portion of code that converts note list, duration list (npy format) into melody and chord matrix|
|get_train_and_test_data.py      |  seperate the melody data into training set and testing set, now including chord handling|

The pre-processing steps for process_notes_and_chords.py requires music21 library.
Showing the scores "score.show()" require MuseScore installed. If not desired, simply uncomment them.

--------------------------------------------------------------------------------------------------
After you have the data (placed in ```data``` folder), there should be 6 files:
* data_X_tr.npy (training data - current bar melody)
* data_y_tr.npy (training data - chord labels)
* prev_X_tr.npy (training data - previous bar melody)
* data_X_te.npy (testing data - current bar melody)
* data_y_te.npy (testing data - chord labels)
* prev_X_te.npy (testin data - previous bar melody)

1. Make sure you have toolkits in the requirement.txt
2. Run main.py ,  
  is_train = 1 for training,  
  is_draw = 1 for drawing loss,  
  is_sample = 1 for generating music after finishing training.
  The sampling function uses 1 bar of testing data as prompt, and the model generates the next 7 measures.

3. If you would like to turn the output into real midi for listening  
  Run demo.py
  typein the desired instrument (default = 0 for piano), and the volume (default 40). These are only for playback purposes

--------------------------------------------------------------------------------------------------
|file                  |  purposes|
|-|-|
|requirement.txt                  |  toolkits used in the whole work|
|main.py                         |  training setting, drawing setting, generation setting.|
|ops.py                          |  some functions used in model|
|model.py                        |  Generator and Discriminator.   (Based on model 3 in the MidiNet paper)|
|demo.py                         |  transform matrix into midi. (input : melody and chord matrix, output : midi)|
# midinet-debugged-20240504
