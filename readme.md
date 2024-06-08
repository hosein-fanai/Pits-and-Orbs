# Pits and Orbs Environment as a Multi-Agent System
## Description
A simple game written from scratch in Python having two modes: Terminal Printing and PyGame Window Rendering. To switch between these modes simply change the following argument: ```pygame_mode=True``` to have a nice PyGame window enabling you to play the game manually; this argument can be changed from ```game.pits_and_orbs.PitsAndOrbs``` initializer ```pygame_mode``` argument. Or, ```render_mode="human"``` argument from ```environment.pits_and_orbs_env.PitsAndOrbsEnv``` and ```utils.make_env``` function.

The ```game``` directory contains files and codes only for game object and is not suitable for RL projects. On the contrary, ```environment``` directory contains gym environment version of the game which has certain properties to be used for Reinforcement Learning algorithms. You can install the game on windows with the installer located in the ```dist``` directory.

The goal of the game is for the player to move the orbs to the pits in order to fill them all; that's just it. Be aware that the game is a non-deterministic environment since there is 10 percent chance of a random, not filling orb is thrown to a random direction into an empty cell after the player fills a pit with an orb. The game's properties can be set to values other than their default to have a distinct game. For example, the size of the game (its grid shape), the number of orbs (```orb_num```), the number of pits (```pit_num```), the number of teams (```team_num```) to devide the players, and, even, the number of players (```player_num```) can be changed by the game's initializer. The ```player_num > 1``` means the game will be a multi-agent system that the agents will try to cooperate with eachother to fill the pits, or they will be competing with eachother if the ```team_num > 1``` is true. Note that when there are more than one player in a team, it means that the players in a team need to cooperate while competing against other team(s). The only thing that makes the environment a bit hard to solve is the limited number of movements that each player can take; the default value of ```max_movements``` is 30. In this environment movements is counted only through actual position changes for a player, but not the direction changes, picking orb up, putting orb down, or throwing orb away since these are called steps in RL.

The challenging part of training an agent is that this environment is partially observable which means that the agent only has access to its eight neighbors, but not the whole state board (a 2D array). A memory is implemented to appease this matter (```game.memory.Memory```). The memory tries to update each new cells that the agent sees (it sees them because they're its neighbors), and this memory is unique for each team. That being said, the argument ```return_obs_type``` (for the game's initializer, environment's initializer, or utils.make_env function) can take three inputs to change the output (the observation): ```"partial obs"```, ```"full state"```, or ```"neighbors"```. The first one makes the game or environment return the memory of the agent(s) from ```reset``` or ```step``` functions. The second one makes them to return the entire table (array) of the game. The third one makes them to return only the 8 neighbors of the player.

The environment only allows players (or agents) to take 5 types of actions: ```Turn Right - Move Forward - Pick Orb Up - Put Orb Down - Throw Orb Away```; the last action is only available when ```team_num > 1``` since it makes the environemtn competitive. It is obvious that ```Move Forward``` action means the player moves to the next cell in the direction that it is in without crossing the boundaries. The action ```Throw Orb Away``` means the player to throw an orb out of a pit.
Different values for the actions are:
| Action | Value |
| :--- | :---: |
| Turn Right | 0 |
| Move Forward | 1 |
| Pick Orb Up | 2 |
| Put Orb Down | 3 |
| Throw Orb Away | 4 |

Different values for player direction are:
| Direction | Value |
| :--- | :---: |
| West | 0 |
| North | 1 |
| East | 2 |
| South | 3 |

Finally, different values for cell type are (valid values that are stored in a numpy array of a certain size that indicate existence of different situations in that position):
| Cell Type | Value |
| :--- | :---: |
| Nothing | 0 |
| Player | 1 |
| Orb | 2 |
| Pit | 3 |
| Player & Orb | 4 |
| Player & Pit | 5 |
| Orb & Pit | 6 |
| Player & Orb & Pit | 7 |

The environment outputs an observation object as an OrderedDict from ```step``` function and ```reset``` function; its structure is as follows:

```python
{
    "board": ..., 
    f"player{i}_movements_": ..., 
    f"player{i}_direction": ..., 
    f"player{i}_has_orb": ..., 
    f"player{i}_position": ..., 
    .
    .
    .
    f"filled_pits_positions": ..., 
    "player_turn": ..., 
}
```

Index ```i``` refer to the index of the player in the current team playing. This is the sharing mechanism in each team, and the other meachanism is their shared memory. The key ```board``` is a numpy array depending on the return_obs_type argument. ```Player0_movements```, ```player0_direction```, ```player0_has_orb```, and ```player0_position``` are characteristics of the player index 0 in the current team. There would be more keys if the current team has more than one player in it. ```player0_position``` is only available when the team has more than one player, the environment has more than one team, or ```return_board_type``` is set to ```positions```. ```filled_pits_positions``` is a list of all pits positions filled by the current team playing, and if a pit is not filled by the current team, its postion is returned by a constant value of grid_size. This key is only available when there are more than one team. At last, ```player turn``` determines which player has to play next (the index of the player in the current team). This key is available when there are more than one player in the current team.

To build the game simply use the following:

```bash
python game/setup.py build
```

or create an installer while building the game:
```bash
python game/setup.py bdist_msi
```

If the ```pygame_with_help``` argument is set to ```True```, the PyGame window will have extra height to show players' movements and a guide to what every possible coloring and shapes mean. For illustration purposes, ```pygame_with_help=True``` is set and shown below:

![](https://github.com/hosein-fanai/Pits-and-Orbs/blob/main/materials/screenshot.jpg?raw=true "A sample screenshot of the starting point of the PyGame Window mode with help showed.")

## Wrappers
In the ```./wrappers/``` directory, five distinct wrappers are implemented to appease the usage of the environment for rl algorithms. ```wrappers.steps_limit.StepsLimit``` truncates the episode if the agent exceeds a certain limit, and it can return a negative reward for that matter. The wrappers ```wrappers.onehot_observation.OnehotObservation```, ```wrappers.normalize_observation.NormalizeObservation```, and ```wrappers.concat_observation.ConcatObservation``` get a list of strings as the keys for the observation dictionary in order to know what values to manipulate. ```OnehotObservation``` onehotes the parts of the observation dictionary that their keys are passed in and doesn't take any action on other parts. ```NormalizeObservation``` normalizes the parts of the observation dictionary that their keys are passed in to a value between 0 and 1 and doesn't take any action on other parts. ```ConcatObservation``` concatenates the parts of the observation dictionary that their keys are passed in and blocks other parts. ```wrappers.self_play_wrapper.SelfPlayWrapper``` comes in handy only when the environment is competitive, and we are trying to solve it with the help of self-play algorithm implemented in the ```rl.agent.Agent``` class. This wrapper sets an agent as the opponent of the first team which helps us to interact with this environment.

## Reward Function
Two distinct reward functions are implemented, one of which is sparse and the other isn't. By the arguments ```reward_function_type``` and ```reward_function```, the reward function of the game can be changed. ```reward_function``` can get any dictionary having keys like the table. The situations (keys) and their values for the two implemented reward functions are as follows:

| Situation Flag | Reward Function Type 0 Values | Reward Function Type 1 Values |
| :--- | :---: | :---: |
| cumulative reward coef | 1. | 1.
| turned right | 0. | -0.1 |
| tried to move forward and moved forward | 0. | 0. |
| tried to move forward but stayed | 0. | 0. |
| tried to move forward | 0. | -0.1 |
| tried to move forward to cell type 1 | 0. | 0. |
| tried to move forward to cell type 2 | 0. | 0. |
| tried to pick orb up with already having another orb | 0. | -0.1 |
| tried to pick orb up in cell type 4 | 0. | 0. |
| tried to pick orb up in cell types other than 4 | 0. | -0.1 |
| tried to put orb down without having an orb | 0. | -0.1 |
| tried to put orb down on cell type 1 | 0. | -0.1 |
| tried to put orb down on cell type 4 | 0. | -0.1 |
| tried to put orb down on cell type 5 | 1. | 1. |
| tried to put orb down on cell type 7 | 0. | -0.1 |
| tried to throw another team's orb away in cell type 7 | 1. | 1.
| tried to throw its team's orb away in cell type 7 | -1. | -1.
| tried to throw orb away in cell types other than 7 | 0. | -0.1
| the player has depleted its movements | 0. | -0.1
| episode is done successfully | 1. | 1.
| episode is done unsuccessfully | 0. | -1.
| episode is not done | 0. | 0.

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
python venv -env pao_env
source pao_env/bin/activate
```
Then, install project's dependencies for rendering it on a PyGame window.

```bash
python -m pip install -r requirements.txt
```

To use RL models you should either install TensorFlow or Stable Baselines3 which uses PyTorch. As a result, TensorFlow models require you to check this link (https://www.tensorflow.org/install/pip) or:

```bash
python -m pip install tensorflow==2.10.0
```

And stable-baselines3 requires you to firstly install PyTorch (https://pytorch.org/get-started/locally/) then:

```bash
python -m pip install stable-baselines3==1.8.0 tensorboard
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
In the ```wrappers``` directory, there various useful observation wrappers to make the returned observations more valueable for RL algorithms. You can use them manually or just set the appropriate argument for the ```utils.make_env``` function. For instance, ```wrappers.onehot_observation.OnehotObservation``` onehots each of the arrays in the ordereddict output; this wrapper makes it easier for a neural network to understand observations. 

## CLI Usage
The CLI is used for running or training RL models on the environment. Be aware that ```-r``` or ```--run``` means to run the model on the environment according to the ./configs/#/run.yaml, and save the frames in a .gif file. Also, ```-t``` or ```--train``` means to train a new model according to the ./configs/#/train.yaml settings.

```bash
python . [--run or --train] "config file path"
```

For more information use CLI's help command:
```bash
python . -h
```

For headless servers write this line of command before the CLI above in the same line (make sure you have installed xfvb thoroughly):
```bash
xvfb-run -s "-screen 0 1400x900x24"
```

After using the ```--train``` flag to train a model, you can trace training metrics via tensorbaord if the the model is training by an algorithm written in stable-baselines3.

```bash
tensorboard --logdir=./logs
```

### .gif file example
If you had installed the project properly, and used the CLI the right way, you should have a similar .gif file as below:

![](https://github.com/hosein-fanai/Pits-and-Orbs/blob/main/gifs/First%20phase%20+%20agent%20has%20a%20memory%20+%20rew%20func%201%20(single%20agent%20with%20470mil-iters-A2C%20model).gif?raw=true "First phase + agent has a memory + rew func 1 (single agent with 470mil-iters-A2C model).gif")

## Available Deep RL Models
For the first phase of the project (single agent playing in the 5x5 grid with 30 movement limit and a memory of its past confrontations with its 8 neighbors), there are three different models in ```./models/``` directory and their correspondig config files are in ```./configs/```.

## Sample Notebooks
There are sample notebooks in ```./notebooks/```. Further use cases are implemented there. You can check them out.

## TODO List
Temporarily positions option is deactivated for furthur debugging.

Add documentation to the entire project.

Add new models.

Add the support for gymnasium api.

Write about rl directory in readme.md

