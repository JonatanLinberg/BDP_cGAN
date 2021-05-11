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
 * The project is built around the **EMNIST** dataset, available in [python-mnist](/python-mnist/) or from:
   * https://github.com/sorki/python-mnist
 * The initial **cGAN** and **FID** code (train_cgan.py) was developed by Jason Brownlee, available from: 
   * https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/ (cGAN)
   * https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/ (FID)

### Training a cGAN
Training a cGAN can be accomplished by running the [train_cgan](/train_cgan.py) code, `python train_cgan.py <save_path>`. The user can enter runtime parameters when prompted or load them from a file, `python train_cgan.py <save_path> <RTP file>`. Alternatively, one can run a multi-model experiment using the run_experience.py-script, however the variables in the script will need to be adjusted for each experiment. 

### Experiment Data
Raw data from the experiment is available in the [final experiment folder](/final_experiment), grouped by EXMO (00-28) and run (0-2). The loss, accuracy and FID measurements are in a csv plain text file and are most easily viewed with the [result plotting](#result-plotting-plot_resultspy) tool. 

Derived and calculated data (including a Mann-Whitney U-test) was generated with [u_test](/u_test.py) and [calc_experiment_data](/calc_experiment_data.py) and can be found as csv-files in the [data folder](/data). The csv-files have columns for measurements in epochs, these are empty in the csv-files, but can easily be calculated by dividing the batch measurement with 881 (1762 for EXMO27 and 440 for EXMO28).

### Tools
#### Latent Space Explorer and Recorder ([LaSER.py](/LaSER.py))
The program features one slider for each latent dimension, for exploration the latent space. The user can define a vector in the space to traverse and even create an animation. 

##### Usage:
 * `python LaSER.py` and enter the model file when prompted or
 * `python LaSER.py <model.h5>`


#### Character Generator ([char_gen.py](/char_gen.py))
The program can be used to generate characters and text using a generator model. A full list of the many features can be found by using the "-H" option. Can also evaluate the euclidean distance between the generated images and present them as a box plot.

##### Usage:
* `python char_gen.py -[options]`


#### Result plotting ([plot_results.py](plot_results.py))
The program can present various graphs of the data collected from a training run. The "window size" prompt refers to the size (in batches) of the window used for a rolling average over the data. 

##### Usage:
* `python plot_results.py <results_csv file>`




```                                                                                    
                                                                                    
             ░▒▒▒░                                                                  
            ▒▓▓▓▓▒▒░                         ░▒▒░                                   
           ░▓▓▓▓▓▓▓▒░             ░░░        ▒▓▓░                   ░░░░░           
          ▒▒▓▓▓▓▓▓▓▓▒             ░▒▒       ░▒▓▓░                 ░▒▒▒▒▒▒░          
         ▒▓▓▓▓▒▒▒▓▓▓▓░            ▒▓▓░      ░▒▓▒░               ░░▒▒▓▓▓▓▓▒░         
         ▒▓▓▓░   ░▓▓▓▒            ░▓▓▒░     ░▓▓░                ░▒▓▓▓▓▓▓▓▓▒░        
        ░▓▓▓░    ░▓▓▓▒            ░▒▓▓▒░    ▒▓▓░               ░▒▓▓▓▓▓▓▓▓▓▓░        
        ▒▓▓▒░    ░▓▓▓▒             ▒▒▓▓▒░  ░▒▓▓▒               ░▒▓▓▓▒▒▒▓▓▓▓▒        
        ▒▓▓▒    ░▒▓▓▓▒              ▒▓▓▓▒  ▒▓▓▓░              ░▒▓▓▓▒░░▒▓▓▓▓▒        
        ▒▓▓░    ▒▓▓▓▒░              ░▒▓▓▓░░▒▓▓▓               ░▒▓▓▓░░░▒▓▓▓▓▒        
       ░▒▓▓░  ░▒▓▓▓▓▒                ░▒▓▓▒▓▓▓▓▒              ░▒▓▓▓▓▒░▒▓▓▓▓▓░        
       ░▓▓▒░░▒▒▓▓▓▓▒                  ░▓▓▓▓▓▓▓▒             ░▒▓▓▓▓▓▓▓▓▓▓▓▓▒░        
       ▒▓▓░ ░▒▓▓▓▓▓▒░                 ░▒▓▓▓▓▓▓░             ░▒▓▓▓▓▓▓▓▓▓▓▓▒░         
      ░▒▓▓░░▒▓▓▓▓▓▓▓▒▒░                 ▒▓▓▓▓▓░             ░▒▓▓▓▓▓▓▓▓▓▒░           
       ▒▓▓▒▒▓▓▓▓▓▓▓▓▓▓▒░                ░░▓▓▓▒              ░▒▓▓▓▓▓▓▓▒░░            
      ░▒▓▓▓▓▓▓▓▒▒▒▒▒▓▓▓▒░                ░▓▓▓▒              ░▒▓▓▓▓▓▓▒░░             
      ░▓▓▓▓▓▒▒░░   ░▒▓▓▓░                ▒▓▓▓▒              ░▒▓▓▓▓▓▒░░       ░░░    
      ▒▓▓▓▓▒░      ░▒▓▓▓▒                ▒▓▓▓▒              ░▒▓▓▓▓▓▓▒▒░░░░  ░▒▒▒░   
      ▒▓▓▓▓▒       ▒▓▓▓▒▒                ▒▓▓▓░              ░▒▒▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▓▓▒   
      ▒▓▓▓▒░░    ░▒▓▓▓▓▒░                ▒▓▓▒░                ░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░  
      ▒▓▓▓▓▓▒▒▒▒▒▒▓▓▓▓▒░                 ▒▓▓▒░                  ░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▒▒░  
      ░▒▓▓▓▓▓▓▓▓▓▓▓▓▓▒░                  ▒▓▓▒                    ░░▒▒▒▒▒▒▓▒▓▓▓▒░░   
      ░░▒▓▓▓▓▓▓▓▓▓▓▓▒░░                  ▒▒▒░                      ░░░▒▒░▒▒▒▒░░     
        ░░░░░░░░░░░░                      ░░                              ░░░       
```
