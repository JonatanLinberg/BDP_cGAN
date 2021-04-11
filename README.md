# Hyperparameters for GenerativeAdversarial Network Training - Conditional GAN on the EMNIST dataset
Bachelor Degree Project in Information Technology  
IT613G, University of Skövde  
Alexander Gustafsson, Jonatan Linberg

### Contents
 * [Code Sources](#Code-Sources)
 * [Training a cGAN](#Training-a-cGAN)
 * [Experiment Data](#Experiment-Data)
 * [Tools](#Tools)
   * [Latent Space Explorer and Recorder (LaSER)](#latent-space-explorer-and-recorder-laserpy)
   * [Character Generator](#character-generator-char_genpy)
   * [Result Plotting](#result-plotting-plot_resultspy)

### Code Sources
 * The project is built around the **EMNIST** dataset, available from:
   * https://github.com/sorki/python-mnist
 * The initial **cGAN** and **FID** code (train_cgan.py) was developed by Jason Brownlee, available from: 
   * https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/ (cGAN)
   * https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/ (FID)

### Training a cGAN
Training a cGAN can be accomplished by running the train_cgan code, `python train_cgan.py <save_path>`. The user can enter runtime parameters when prompted or load them from a file, `python train_cgan.py <save_path> <RTP file>`. Alternatively, one can run a multi-model experiment using the run_experience.py-script, however the variables in the script will need to be adjusted for each experiment. 

### Experiment Data
-- _To be added at a later date_ --

### Tools
#### Latent Space Explorer and Recorder ([LaSER.py](/LaSER.py))
The program features one slider for each latent dimension, for exploration the latent space. The user can define a vector in the space to traverse and even create an animation. 

##### Usage:
 * `python LaSER.py` and enter the model file when prompted or
 * `python LaSER.py <model.h5>`


#### Character Generator ([char_gen.py](/char_gen.py))
The program can be used to generate characters and text using a generator model. A full list of options can be found by using the "-H" option. Can also evaluate the euclidean distance between the generated images and present them as a box plot.

##### Usage:
* `python char_gen.py -[options]`


#### Result plotting ([plot_results.py](plot_results.py))
The program can present various graphs of the data collected from a training run. 

##### Usage:
* `python plot_results.py <results_csv file>`
