# CMPE591 Homework 3

This repository contains the implementation of reinforcement learning algorithms for solving the `Pusher-v5` environment. The project includes training and testing scripts for two main approaches: Soft Actor Critic (SAC) and Vanilla Policy Gradient (REINFORCE).

---

## Repository Structure

```
cmpe591_homework_3/
│
├── soft_actor_critic/
│   ├── train.py  # Training script for SAC
│   ├── test.py   # Testing script for SAC
│   ├── ...
│
├── vanilla_policy_gradient/
│   ├── train.py  # Training script for Vanilla Policy Gradient
│   ├── test.py   # Testing script for Vanilla Policy Gradient
│   ├── ...
│
└── sac_plots/
    ├── final_reward_plot.png  # Final reward plot for SAC training
    ├── ...
```

---

## Training and Testing

### Training
- **Soft Actor Critic (SAC)**
  - The training script for SAC is located in `soft_actor_critic/train.py`.

- **Vanilla Policy Gradient (REINFORCE)**
  - The training script for Vanilla Policy Gradient is located in `vanilla_policy_gradient/train.py`.

### Testing
- **Soft Actor Critic (SAC)**
  - The testing script for SAC is located in `soft_actor_critic/test.py`.
  - Example output for the test script:
    ```python
    Episode 1: Reward = 100.0, Steps = 200
    Average reward over 10 episodes: 98.5
    ```

- **Vanilla Policy Gradient (REINFORCE)**
  - The testing script for Vanilla Policy Gradient is located in `vanilla_policy_gradient/test.py`.
  - Example output for the test script:
    ```python
    Test Episode 1: Steps=150, Reward=95.0
    Average Reward over 10 episodes: 94.8 +/- 1.2
    ```

---

## Final Reward Plots

The training process for the algorithms is visualized through reward plots. The final reward plots can be found in the following locations:

- **Soft Actor Critic (SAC)**: `sac_plots/final_reward_plot.png`  
  ![SAC Final Reward Plot](sac_plots/final_reward_plot.png)

- **Vanilla Policy Gradient (REINFORCE)**: `reinforce_plots/final_reward_plot.png`  
  ![REINFORCE Final Reward Plot](reinforce_plots/final_reward_plot.png)

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/DenizBilgeAkkoc/cmpe591_homework_3.git
   cd cmpe591_homework_3
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training scripts:
   ```bash
   python soft_actor_critic/train.py
   python vanilla_policy_gradient/train.py
   ```

4. Run the testing scripts:
   ```bash
   python soft_actor_critic/test.py
   python vanilla_policy_gradient/test.py
   ```

---

## Acknowledgments

This project was developed as part of the CMPE591 course. Special thanks to the course instructors for their guidance.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
