# DLCV-Fall-2022-HW3

Please click [this link](https://docs.google.com/presentation/d/1tza5rtruvOkoComWRS79Yb7IgM_R0pGLyrI4m8Jk5Xg/edit#slide=id.g10278b72a69_0_448) to view the slides of HW3

# Usage

To start working on this assignment, you should clone this repository into your local machine by using the following command.
    
    git clone https://github.com/DLCV-Fall-2022/hw3-<username>.git


Note that you should replace `<username>` with your own GitHub username.

# Submission Rules
### Deadline
2022/11/21 (Mon.) 23:59

### Packages
This homework should be done using python3.8. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt
    conda create --name dlcv-hw3 python=3.8
    conda activate dlcv-hw3
    
### Check the list of modules

    conda list

### Install module

    pip3 install --target=/home/zhongwei/.conda/envs/dlcv-hw3/bin/ --upgrade -r requirements.txt
    pip3 install --target=/home/zhongwei/.conda/envs/dlcv-hw3/bin/ git+https://github.com/openai/CLIP.git
    pip3 install --target=/home/bl530/anaconda3/bin git+https://github.com/openai/CLIP.git
    pip3 install --target=/home/zhongwei/.conda/envs/dlcv-hw3/bin/ git+https://github.com/bckim92/language-evaluation.git

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

# Q&A
If you have any problems related to HW3, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw3 FAQ section in FB group.(But TAs won't answer your question on FB.)

# Reminder:
inference: 

1. device = "cuda"
2. torch.load(model_name, map_location='cuda')

Example:

    if torch.cuda.is_available():
        if torch.cuda.device_count()==2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device)
    ...
    netG.load_state_dict(torch.load(ckpt_path, map_location='cuda'))

