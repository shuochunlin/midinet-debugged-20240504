

# midinet-debugged-20240504
This is the edited version of [Debugged MidiNet](https://github.com/dongmingli-Ben/MidiNet-by-pytorch-Debugged), which is a debugged implementation of [Implementation of MidiNet by pytorch](https://github.com/annahung31/MidiNet-by-pytorch).

MidiNet paper : https://arxiv.org/abs/1703.10847 

MidiNet code  : https://github.com/RichardYang40148/MidiNet 

Instead of training on melody, we use the FiloBass dataset to train basslines.

Dataset       : https://aim-qmul.github.io/FiloBass/ 

--------------------------------------------------------------------------------------------------
## XML files to NPY files (Optional)

This step is not necessary as dataset has been fully prepared in "./data" folder.

The relevant files are not added to repository yet, but basically it's just processing the XML files (melody and chords information) into NPY file formats, so I can re-use the code written by the author.

*(to be documented)*

|file to run                     |  function|
|-|-|
|*process_notes_and_chords.py*     |  (not yet included) \[NEW\]: preprocessing xml files into csv files, with melody and chords quantization. The pre-processing steps for process_notes_and_chords.py requires music21 library. Showing the scores "score.show()" require MuseScore installed. If not desired, simply uncomment them.|
|*csv_to_npy.py*                   |  (not yet included) \[NEW\]: convert into npy files, while converting chord labels to a range 0 ~ 25 accepted by the next step|

Afterwards the following files are created:

|npy files                     |  function|
|-|-|
|note_list_all_c.npy             |  Contains all pitches in order (list), extracted from XML files of the dataset|
|dur_list_all_c.npy              |  Durations of pitches (list) extracted from XML files of the dataset|
|chord_list_all_c.npy            |  Contains all chords in order (list), extracted from XML files of the dataset|
|cdur_list_all_c.npy             |  Durations of chords (list) extracted from XML files of the dataset|

--------------------------------------------------------------------------------------------------

## Prepare the data (Optional)

This step is also not necessary as dataset has be prepared in "./data" folder. But just in case there are model configuration changes, the pre-processing may need to be redone as well. Otherwise, no need to touch them.

This step converts xml files into training data in npy files format.

|file to run                     |  function|
|-|-|
|get_data.py                     |  Preprocessing. Reusing the portion of code that converts note list, duration list (npy format) into melody and chord matrix. Contains data augmentation. Due to previous steps used, only the get matrix part is used.|
|get_train_and_test_data.py      |  seperate the melody data into training set and testing set. Data augmentation is taken into account. I added chord handling by duplicating and re-using the original function.|
|get_train_and_val_and_test_data.py      |  seperate the melody data into training set, validation set, and testing set. Data augmentation is taken into account. I added chord handling by duplicating and re-using the original function.|

Dimensions of data: 128 pitch_dim, 16 subdivisions per bar. 

The current files in "data" folder have two versions: one augmented into 3 out of 12 keys, and the other without. Both are in zip formats, which requires unzipping. 

Both can be split into training, validation set and testing set 8 : 1 : 1. Or if without validation set, 8 : 2.

If the pre-processing is run, the following files would be created during the process:
|npy files                       |  function|
|-|-|
|data_x.npy                      |  Our music data content in piano roll format, pre-split|
|data_y.npy                      |  Our music labels (chords), pre-split  |
|prev_x.npy                      |  Previous bar (used with data_x.npy) of the current bar asked to be generated, pre-split|
|song_lengths.npy                |  A temporary file created so that when splitting train/val/test, they are split by songs correctly.|
|"/data/*_te.npy"                |  Testing set (no validation set) |
|"/data/*_tr.npy"                |  Training set (no validation set) |
|"/data/train-val-test/*_te.npy"                |  Testing set |
|"/data/train-val-test/*_tr.npy"                |  Validation set |
|"/data/train-val-test/*_val.npy"               |  Training set |

A recent bug was fixed from the original code. Originally, only the first 8 bars of the song are in the processed dataset. It is now fixed to include all 8-bar segments.

--------------------------------------------------------------------------------------------------
## Training the model

After preprocessing the data (placed in ```data/train-val-test``` folder), there should be 9 files:
* data_X_tr.npy (training data - current bar melody)
* data_y_tr.npy (training data - chord labels)
* prev_X_tr.npy (training data - previous bar melody)
* data_X_val.npy (validation data - current bar melody)
* data_y_val.npy (validation data - chord labels)
* prev_X_val.npy (validation data - previous bar melody)
* data_X_te.npy (testing data - current bar melody)
* data_y_te.npy (testing data - chord labels)
* prev_X_te.npy (testin data - previous bar melody)

Or 6 files if without validation set in ```data``` folder:
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
  [New] model_id = ? for keeping track of samples created by a model that you set as id
  [New] has_val_test = 0 for not using validation set, 1 for using validation set
3. Run main.py.

Relevant hyperparameters, aside from the model, include:
1. label_smoothing_weight: Author set real labels to 0.9 (instead of 1), and fake labels to 0. Alter this value to affect confidence of discriminator.
2. mean_image_weight: Lambda 1 of author's feature matching function. Setting values higher lead to closer resemblance to real data.
3. feature_matching_weight: Lambda 2 of author's feature matching function. Setting values higher lead to closer resemblance to real data.
4. continuity_weight: [new] Added to penalize model for not holding notes enough. Optimal values still under experimentation.
5. batch_size: The usual batch size.

Metrics to monitor while training:
1. Loss_D: Discriminator loss. Should decrease over time slowly.
2. Loss_G: Generator loss. Should not be too low or too high.
3. D(x): Score for Discriminator's real score, for identifying real data correctly. Should be around 0.9 (label_smoothing_weight) to identify real examples, so generator tries to match it.
4. D(G(z)): Score for Discriminator fake score, for identifying fake data correctly. Should hover around 0.5 after a while if generator's fooling discriminator half the time.

After running the model the following files would be created:
|files                     |  function|
|-|-|
|"/file/*.png"              | saves fake samples and real samples as png files|
|"/draw_figs/lr*_epoch*.png"| saves a figure of model's training progression|
|"/models/*.pth"            | saves model files (note author did not implement restore training from an epoch)|
|lossD_list_{model_id}.npy             |  Generated after is_train = 1. Used to draw loss during is_draw = 1.|
|lossG_list_{model_id}.npy             |  Generated after is_train = 1. Used to draw loss during is_draw = 1.|
|lossD_list_all_{model_id}.npy         |  Generated after is_train = 1. |
|lossG_list_all_{model_id}.npy         |  Generated after is_train = 1. |
|D_x_list_{model_id}.npy               |  Generated after is_train = 1. |
|D_G_z_list_{model_id}.npy             |  Generated after is_train = 1. |
|{model_id}_output_songs.npy           |  Generated after is_sample = 1. These will be read by demo.py later.|
|{model_id}_output_chords.npy          |  Generated after is_sample = 1. These will be read by demo.py later.|

### Notes on Loss Function
As per paper, the GAN learns G and D by solving:

minmax V(D,G) = E_{x~P_{data}(X)}[log(D(X))] + E_{z~P_z(z)}[log(1-D(G(z)))]

While the generator added these two terms as well to increase training stability:

lambda_1||{E X - E G(z)}||^2_2 _ lambda_2||E f(X) - E f(G(z))||^2_2

I also added continuity_loss to penalize not holding notes long enough. Given generated sequence $G(z)$ in shape of $(B,T,P)$ where $B$ is batch size, $T$ is seq length, $P$ is pitch number classes, we compute the pitch differences for consecutive timesteps and then take the absolute value of those differences. In an equation:

$$|\Delta G(z)| = \left| G(z)[:, 1:, :] - G(z)[:, :-1, :] \right|$$

Finally we take the sum of absolute differences to get the penalty:

$$L_{\text{continuity}} = \sum_{b=1}^B \sum_{t=1}^{T-1} \sum_{p=1}^P \left| G(z)[b, t, p] - G(z)[b, t-1, p] \right|$$

With that we simply add this (multiplied by a weight lambda3):

lambda_1||{E X - E G(z)}||^2_2 _ lambda_2||E f(X) - E f(G(z))||^2_2 + lambda_3(L_{continuity})


In regards to chord adherence, we found that the model has bigger problems adapting to learning rhythmic patterns, so the focus has shifted to that instead. 

--------------------------------------------------------------------------------------------------

## Listening to music samples

To convert music samples into MIDI, Run demo.py.

Enter the id of the model prompted, and it will convert {model_id}_output_songs.npy and {model_id}_output_chords.npy into playable MIDI files.

After running the files the samples are stored in "/samples" folder, in MIDI file format. In case higher music quality demo is desired, music software would be needed to further process MIDI files into mp3 files.

--------------------------------------------------------------------------------------------------

## Evaluating the music samples

Customize and run music_analysis_with_mgeval.ipynb. It uses the metrics provided by the same author's later paper: https://link.springer.com/epdf/10.1007/s00521-018-3849-7?author_access_token=Z7YxQv2K9z33nk1_XGlY9_e4RwlQNchNByi7wbcMAY5M_T6iwDlmVavmHfG20IIuk492IRWVj17BK1zhOxg5HA5fo8df4mI0b3U1YbTvprNarTF7BunHbKBquKplW2anwIy_TzUtUKq8g6tZzhCUzQ%3D%3D

Chord note percentage can be analyzed with chord_notes_percentage.py, which checks MIDI notes and see how many notes are part of chord notes. 

|files                     |  function|
|-|-|
|music_analysis_with_mgeval.ipynb     | analyzes music features and prints results|
|chord_notes_percentage.py            | prints the chord notes count and percentages|

Features have yet to be documented, but [https://github.com/RichardYang40148/mgeval documentations are available here].


(Last Update: 2024/5/17)
