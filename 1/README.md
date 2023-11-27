# Exercise 1: Multi-Armed Bandits



### Code structure

- The multi-armed bandit environment is defined in `BanditEnv.py`
- Epsilon-greedy/UCB agents are given in `agents.py`. Both agents inherit the abstract class `BanditAgent`.
- To break ties randomly, we define our own argmax function in `agents.py`.
- The main file to run code is `main.py`, with functions for each question.
- I installed "tqdm" but when runing the code, it could not find it so I copied its file in the code file 
- "tqdm" is a Python library that allows you to output a smart progress bar by wrapping around any iterable. A tqdm progress bar not only shows you how much time has elapsed, but also shows the estimated time remaining for the iterable