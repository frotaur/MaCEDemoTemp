# MaCEDemoTemp
Repository holding a small demo of MaCE-based models. Temporary for ALIFE 2025!

To play with MaCE and other Alife models, see [the PyCA repo](), which is the basis of this repository. For more links on MaCE, [see this link](https://vassi.life/research/mace). 

## Installation
You need python 3.11 or later (ideally, might work with earlier versions).

<font color="red"> FOR WINDOWS PC WITH GPUS :  If you are running windows, and have a NVIDIA GPU, please run the following command before proceeding : `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`. If something doesn't work afterwards, visit https://pytorch.org/ and choose an earlier version.  Linux and Mac users can safely skip this step.</font>


Install the pyca package by running `pip install -e .` from the projects directory (i.e., the same directory as `README.md`). You are all set!

## Run the simulation
To run the main user interface that allows you to interact with the implementend automata, run : 
```[python]
python simulate.py
```

If you have a cuda GPU, run : 
```
python simulate.py -d cuda
```

More generally you can change the Screen size with the options `simulate.py [-h] [-s SCREEN SCREEN] [-w WORLD WORLD] [-d DEVICE]`
```options:
  -s SCREEN SCREEN, --screen SCREEN SCREEN
                        Screen dimensions as width height (default: 1280 720)
  -w WORLD WORLD, --world WORLD WORLD
                        World dimensions as width height (default: 200 200)
  -d DEVICE, --device DEVICE
                        Device to run on: "cuda" or "cpu" or "mps" (default: cpu)
```

<font color="red"> NOTE : 'mps' device is known to behave very differently. Other devices untested </font>


## Documentation
Documentation is under construction. In the meantime, the code is heavily documented with docstrings

## Code structure

```python
├─pyca
│  ├──automata
│  │   ├──models/ # All implemented automata
│  │   ├──utils/
│  │   ├──automaton.py # Base Automaton class
│  ├──interface/ # Utils for the PyCA GUI
├─demo_data # Cool demo parameters for MaCE/Lenia automata
├─simulate.py # Main entry script for PyCA  
```

