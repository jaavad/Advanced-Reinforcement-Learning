# Exercise 4: Monte-Carlo Methods

## Setup

* Python 3.5+ (for type hints)

Install requirements with the command:
```bash
pip install -r requirements.txt
```

Besid requirements.txt, you might need to install pygame 

Q3-b will take long time for training 

## Complete the code

Fill in the sections marked with `TODO`. The docstrings for each function should help you get started. The starter code is designed to only guide you and you are free to use it as-is or modify it however you like.

### Code structure

- Envs: Blackjack is contained within the `gym` library, and a custom Four Rooms gym environment is given in `env.py`. Complete the `step()` function. To use the custom environment, call `register_env()` once.
- Algorithms are given in `algorithms.py`.
- Policies for the blackjack and Four Rooms environments are given in `policy.py`. For these simple policies, we represent policies as closures so that the policies are updated as soon as the Q-values are updated.
- Create your own files to plot and run the algorithms for each question.
- For the racetracks problem, create your own environment using the grids given in `racetracks.py`.
