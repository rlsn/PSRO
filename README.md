# PSRO

A python implementation of the Policy-Space Response Oracles (PSRO) algorithm as described in [Balduzzi, David, et al. "Open-ended learning in symmetric zero-sum games." International Conference on Machine Learning. PMLR, 2019.](https://arxiv.org/abs/1901.08106). Some minor modifications have been made for performance purposes. The algorithm computes an approximate Nash equilibrium of a zero-sum two-player imperfect information game.

## Quick Start
run `python train.py` to see an exmple run with the toy rock-paper-scissor game with a modified payoff matrix. To configure the training settings, run `python train.py -h` to see available options.
