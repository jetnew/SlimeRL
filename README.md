# Slime-RL

## Guide for running experiment on AWS

1. Login into an available AWS instance using the credentials found [here](https://docs.google.com/spreadsheets/d/1PcbQiIeJtteGoNbuYQsMinYbKFchufB-ml94mOgabKw/). 
2. After successful login, login to our user on AWS. Type `su 3244-2010-0008` and use the same password that you used in step 1. 
3. If the conda environment is already setup (refer to the AWS instance spreadsheet), you can go ahead and run your experiment. 
4. If the conda environment is not setup or if you're having issues with the environment, run `bash setup.sh` from the root of the codebase. This will setup the `slime-rl` environment and install all the required dependencies for you. 
5. Activate the environment if not already activated by step 4 (if you skipped it) by typing `conda activate slime-rl`. 
6. Open a linux screen session by typing `screen`. 
7. Run the experiment

If you want to login to AWS and resume again, use `screen -r` and you will be able to see ur experiment running. This assumes you only have one screen. If not, you will be prompted to add the id of the screen you want to resume to from the displayed result. 

Note: if conda is not installed, please don't use the instance. We need admin rights to install conda and the TA assigned for AWS hasn't responded to our email so we don't know when he can install it for us. 

If you face any issues, please feel free to message on the Telegram group or DM me (@raivat). 
