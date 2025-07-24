Hey!

Here are the instructions for running this project (at least until now) and is by running all commands in your terminal:

1. Be sure you have a system that allows you to develop and choose environments, in my case, I use MAMBA (highly recommend).

1. Download this directory as a zip and unzip it, or if you know how to use GIT, then clone this repository in your computer.

1. Once you have this, you first have to generate the environment. First, go to your terminal and locate yourself in the directory that you just added in your computer using the command `cd`. When you are inside the folder called "ML_HETEROGENEOUS_EFFECTS", you can generate the environment using the .yml file. If you are using mamba, you should be able to use this command:

```bash
conda env create -f env_ml.yml
```
(This step should take some minutes...)

1. After this is done, you should activate your environment by typing:

```bash
conda activate ml_project
```
(This command is if you are using mamba. The environment should appear on the left of your cd now)

1. Now, we need to create two folders, for this you should type in your terminal these commands:
```bash
python cd ml_heterogeneity_project/
python generate_folders.py
```
1. After these commands, two new folders should be generated, "bld" (which will be used for storing the files generated) and "data".

1. The next step is to upload the data. In the folder that was just generated ("data") you should add the data file "waves123_augmented_consent.dta".

1. Now we are ready to get some results! There are different results files and they are all starting with the word "running".

    - If you want to run the Causal Forest and obtain all the results that comes with it, you only need to run: 
    ```bash
    python running_causal_forest.py
    ```
    - If you want some basic statistics, the file you need to run is:
    ```bash
    python running_some_extra_stats.py
    ```
    - There is a file for tuning the forest. For this you should just run:
    ```bash
    python running_tuning_forest.py
    ```

As a summary, I will explain here what you have in each file.
- *running_causal_forest*: In this file you will create a Causal Forest and also different graphs (better explained in the file itself). It will generate also tests, the DRTest will be generating also graphs that will be stored in the bld folder and also it will print some results in your terminal. The RScorer will also print results in your terminal. There are extra functions that are not used but feel free to add them.
- *running_some_extra_stats*: This file will run different graphs and they will also appear in the bld folder. Here you will have treated vs. not treated distributions (to check the randomization), a heatmap between covariates (to check the correlation between them), and a placebo test that also save the graphs in this bld folder.

You can also go to the notebooks in the notebooks folder, for example "notebook_4every1.ipynb" and if you select the environment created ("ml_project") you can run it little by little. You just need to make sure you follow at least until the step you save the data in the "data" folder.

Please let me know if there's  any mistake or if something is unclear :)