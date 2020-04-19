# Kaggle-silver-bengaliai FASTAI2

- It ended up at [61th place, top 3%](https://www.kaggle.com/c/bengaliai-cv19/leaderboard) in the competition. Team Name
 is **Russell Ershov** (was Dead AI but for some unexplained reasons it's not).
- This is my 5th real competition in ML and the second in computer vision area. Everything was done solo. 

## Directory layout

```
.
├── bin           # Scripts to perform various tasks such as `setup_env`, `train`.
├── conf          # Configuration files for classification models.
├── input         # Input files provided by kaggle. 
├── model         # Where classification model outputs are saved.
├── src           # Where all main scripts are located.
├── ipynbs        # Where all notebooks with solution are located.
└── submission    # Where submission files are saved.
```
Missing directories will be created with `./bin/setup_env.sh` once you run it. For custom directories names you should specify everything in `./bin/setup_env.sh file` (but after this you should to change these names inside of the config file).

Everything inside of `config.py` can be replaced by the custom `paths/functions` etc. Once you've changed everything inside of the `config.py`, you need to add your custom functions and augmentations in `src/train_utils/bengali_funcs.py` and `src/train_utils/bengali_augs` respectively. 

## How to run

First of all, you need to run `./bin/setup_env.sh` which will create all folders you need. Second, you need to place your files in directories created by `./bin/setup_env.sh`. You can do it by simply running: mv `path_to_your_files` `created_directories`. 

Please make sure you run each of the scripts from parent directory of `./bin`.

### Preprocessing
Please notice that there are no preprocessing functions in this repository because I haven't done any preprocessing on my own. Images have taken from *https://www.kaggle.com/iafoss/image-preprocessing-128x128*.
The same comes for folded_df. I have taken one from *https://www.kaggle.com/yiheng/iterative-stratification*.

All notebooks you can find in ipynbs directory

### Training (classification model)

~~~
$ sh ./bin/train.sh
~~~

- Here I present only one model xresnet50, but you can choose any model you want.


### Predicting (classification model)

- There is no special script for predicting. Everything you need is just `bengali-ensemble-inference.ipynb`

### Trained Weights

- Trained model weights can be found in `model/your_model_name/`
