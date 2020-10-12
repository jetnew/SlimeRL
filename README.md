# Slime-RL

## Guide for running experiment on AWS

1. Login into an available AWS instance using the credentials found [here](https://docs.google.com/spreadsheets/d/1PcbQiIeJtteGoNbuYQsMinYbKFchufB-ml94mOgabKw/). 
2. After successful login, login to our user on AWS. Type `su 3244-2010-0008` and use the same password that you used in step 1. 
3. If the conda environment is already setup (refer to the AWS instance spreadsheet), you can go ahead and run your experiment. 
4. If the conda environment is not setup or if you're having issues with the environment, run `bash setup.sh` from the root of the codebase. This will setup the `slime-rl` environment and install all the required dependencies for you. 

Note: *If you find that conda is not installed (`conda: command not found`). Follow the guide [here](https://devopsmyway.com/install-anaconda-on-amazon-linuxec2/) to first install conda and then go to step 4.*

If you face any issues, please feel free to message on the Telegram group or DM me (@raivat). 
