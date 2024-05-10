
This is a **Debugged** version of [Implementation of MidiNet by pytorch](https://github.com/annahung31/MidiNet-by-pytorch).

MidiNet paper : https://arxiv.org/abs/1703.10847 

MidiNet code  : https://github.com/RichardYang40148/MidiNet 

dataset is from theorytab : https://www.hooktheory.com/theorytab 

You can find crawler here : https://github.com/wayne391/Symbolic-Musical-Datasets 




--------------------------------------------------------------------------------------------------
Prepare the data

|file to run                     |  function|
|-|-|
|get_data.py                     |  get melody and chord matrix from xml|
|get_train_and_test_data.py      |  seperate the melody data into training set and testing set (chord preparation not included)|

Or get the processed data from [the author](https://drive.google.com/drive/folders/1kQ9nXolLTOw1MNC8nPNguIXsvAFcwYCw).

*Note: The processed data from the author combines the training and testing data together. You may need to re-separate them.*  
*Another Note: Whether ```get_data.py``` and ```get_train_and_test_data.py``` are bug-free is unknown (download the processed data from the author is recommended).*


--------------------------------------------------------------------------------------------------
After you have the data (placed in ```data``` folder), 
1. Make sure you have toolkits in the requirement.txt
2. Run main.py ,  
  is_train = 1 for training,  
  is_draw = 1 for drawing loss,  
  is_sample = 1 for generating music after finishing training.
  
3. If you would like to turn the output into real midi for listening  
  Run demo.py

--------------------------------------------------------------------------------------------------
|file                  |  purposes|
|-|-|
|requirement.txt                  |  toolkits used in the whole work|
|main.py                         |  training setting, drawing setting, generation setting.|
|ops.py                          |  some functions used in model|
|model.py                        |  Generator and Discriminator.   (Based on model 3 in the MidiNet paper)|
|demo.py                         |  transform matrix into midi. (input : melody and chord matrix, output : midi)|
# midinet-debugged-20240504
