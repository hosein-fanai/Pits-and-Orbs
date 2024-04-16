# Pits and Orbs Environment as a Multi-Agents System
## Description
A simple game written from scratch in python that has two modes: Terminal Printing and PyGame Window. To switch between these modes simply change this argument as following: ```pygame_mode=True``` to have a nice PyGame enabling you to play the game manually. The environment only allows players (or agents) to take 4 types of actions: ```Turn Right - Move Forward - Pick Orb Up - Put Orb Down```. It is obvious that ```Move Forward``` action means that the player moves to the next cell in the direction that it is in without crossing the boundaries. This argument is available in various files, classes and functions, such as ```game.pits_and_orbs.PitsAndOrbs```, ```environment.pits_and_orbs_env.PitsAndOrbsEnv``` and ```utils.make_env```. To build the game simply use the following:

```bash
python3 ./game/setup.py build
```

or create an installer while building the game:
```bash
python3 ./game/setup.py bdist_msi
```

For illustration purposes, if the ```pygame_with_help``` argument is set to ```True```, the PyGame window will have extra height to show players' movements and a guid to what every possible coloring and shapes mean.

![](https://github.com/hosein-fanai/Pits-and-Orbs/blob/main/materials/screenshot.jpg?raw=true "A sample screenshot of the starting point of the PyGame Window mode with help showed.")

As a reminder, the ```utils.make_env``` function is the best and easiest way to create a gym environment out of this game since it provides decent default parameters to train the environment for an RL algorithm.

## Loading the Project
To load the project use the following logic:

```bash
git clone https://github.com/hosein-fanai/Pits-and-Orbs.git
cd "Pits-and-Orbs"
```

If you are going to use the project in headless server like Google Colab, try installing a virtual display as the following (this is only the case if you are trying to run an agent in the environment and create a .gif file from the frames of the PyGame window since PyGame needs a display to render its frames):

```bash
apt-get install swig cmake ffmpeg
apt install python-opengl
apt install ffmpeg
apt install xvfb
pip3 install pyvirtualdisplay
```

Then, use the following command in python (like colab) to create a virutal display before any of the project's codes:

```python
from pyvirtualdisplay import Display


virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```


## Installation
To use the game and its environment you can install below neccessary dependancies in a new virtual environment (all the materials are tested in python=3.10.10), but these do not include the usages for RL agents.

```bash
python3 venv -env pae_env
source pae_env/bin/activate

python3 -m pip install -r requirements.txt
```

To use RL models you should either install TensorFlow or Stable Baselines3 which uses PyTorch. As a result, tensorflow models require you to:

```bash
python3 -m pip install tensorflow==2.10.0
```
And stable-baselines3 require you to:

```bash
python3 -m pip install stable-baselines3==1.8.0
```

## CLI Usage
The below CLI is used for running a certain RL model on the environment and saving the PyGame frames to a .gif file.

```bash
python3 ./ "path to the model" "path to save the gif file"
```

### .gif file example
If you had installed the project properly, and used the CLI the right way, you should have a similar .gif file as below:

![](https://github.com/hosein-fanai/Pits-and-Orbs/blob/main/gifs/First%20phase%20(single%20agent%20with%2010mil-iters-A2C%20model).gif?raw=true "First phase (single agent with 10mil-iters-A2C model")

## Sample Notebook
The is a sample notebook in the root directory of the repository: ```./notebook.ipynb```. Further use cases are implemented there. You can check it out.

