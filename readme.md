# Pits and Orbs Environment as a Multi-Agents System
## Description
A simple game written from scratch in Python having two modes: Terminal Printing and PyGame Window Rendering. To switch between these modes simply change this argument as follows: ```pygame_mode=True``` to have a nice PyGame enabling you to play the game manually. This argument is available in various files, classes, and functions, such as ```game.pits_and_orbs.PitsAndOrbs```, ```environment.pits_and_orbs_env.PitsAndOrbsEnv``` and ```utils.make_env```. The environment only allows players (or agents) to take 4 types of actions: ```Turn Right - Move Forward - Pick Orb Up - Put Orb Down```. It is obvious that ```Move Forward``` action means that the player moves to the next cell in the direction that it is in without crossing the boundaries. The challenging part of training an agent is that this environment is partially observable which means that the agent only has access to its eight neighbors, but not the whole state board (a 2D array). A memory is implemented to appease this matter. The memory tries to save each new cells that the agent sees after getting partial observation. To build the game simply use the following:

```bash
python3 game/setup.py build
```

or create an installer while building the game:
```bash
python3 game/setup.py bdist_msi
```

For illustration purposes, if the ```pygame_with_help``` argument is set to ```True```, the PyGame window will have extra height to show players' movements and a guide to what every possible coloring and shapes mean.

![](https://github.com/hosein-fanai/Pits-and-Orbs/blob/main/materials/screenshot.jpg?raw=true "A sample screenshot of the starting point of the PyGame Window mode with help showed.")

As a reminder, the ```utils.make_env``` function is the best and easiest way to create a gym environment out of this game since it provides decent default parameters to train the environment for an RL algorithm.

## Loading the Project
To load the project use the following logic:

```bash
git clone --depth 1 https://github.com/hosein-fanai/Pits-and-Orbs.git
cd "Pits-and-Orbs"
```

If you are going to use the project in a headless server like Google Colab, try installing a virtual display as the following (this is only the case if you are trying to run an agent in the environment and create a .gif file from the frames of the PyGame window since PyGame needs a display to render its frames):

```bash
apt-get install swig cmake ffmpeg
apt install python-opengl
apt install ffmpeg
apt install xvfb
pip3 install pyvirtualdisplay
```

Then, use the following piece of code in Python (like Colab) to create a virtual display before any of the project's codes:

```python
from pyvirtualdisplay import Display


virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```

## Installation
To use the game and its environment you can install the below necessary dependencies in a new virtual environment (all the materials are tested in Python=3.10.10), but these do not include the usages for RL agents. First create and activate a new environment or just skip this part (if you are using a headless server like Colab make sure to go through with the last section's guide in your dev environment).

```bash
python3 venv -env pae_env
source pae_env/bin/activate
```
Then, install project's dependencies for rendering it on a PyGame window.

```bash
python3 -m pip install -r requirements.txt
```

To use RL models you should either install TensorFlow or Stable Baselines3 which uses PyTorch. As a result, TensorFlow models require you to:

```bash
python3 -m pip install tensorflow==2.10.0
```

And stable-baselines3 requires you to:

```bash
python3 -m pip install stable-baselines3==1.8.0 tensorboard
```

## Making the Environment

There two options to make an instance of gym-environemnt of the game. First, as mentioned above, you can use ```utils.make_env``` to create a new instance of the enironment. This option gives you access to changing various game class arguments, plus a couple of observation wrappers to appease environment usages for RL algorithms. Secondly, just use ```gym.make``` api as below considering that kwargs give you access to game class arguments:

```python
import gym

import environment


env = gym.make("EchineF/PitsAndOrbs-v0", **kwargs)
# or
env = gym.make("EchineF/PitsAndOrbs-two-players-v0", **kwargs)
```

## CLI Usage
The CLI is used for running or training RL models on the environment. Be aware that ```-r``` or ```--run``` means to run the model on the environment according to the ./configs/run.yaml, and save the frames in a .gif file. And, ```-t``` or ```--train``` means to train a new model according to the ./configs/train.yaml settings.

```bash
python3 . [-r or -t] --mpath="path to the model"  --gpath="path to save the gif file"
```

For headless servers write this before the CLI above in the same line (make sure you have installed xfvb thoroughly):
```bash
xvfb-run -s "-screen 0 1400x900x24"
```
For more information use CLI's help command:
```bash
python3 . -h
```

### .gif file example
If you had installed the project properly, and used the CLI the right way, you should have a similar .gif file as below:

![](https://github.com/hosein-fanai/Pits-and-Orbs/blob/main/gifs/First%20phase%20(single%20agent%20with%2010mil-iters-A2C%20model).gif?raw=true "First phase (single agent with 10mil-iters-A2C model")

## Sample Notebook
There is a sample notebook in the root directory of the repository: ```./notebook.ipynb```. Further use cases are implemented there. You can check it out.

