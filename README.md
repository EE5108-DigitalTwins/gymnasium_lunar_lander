# EE5108 Assignment: Lunar Lander with Deep Q-Networks

## Overview

In this assignment, you will implement a Deep Q-Network (DQN) agent to master the LunarLander-v3 environment. The lunar lander must learn to navigate from the top of the screen to land safely between two flags on the moon surface, using engine thrusts efficiently.

**Deadline** Midnight, March 15th \
**Percentage of Final Grade** 15%

![Episode 1](images/episode_1.gif)

## Problem Statement
The Lunar Lander environment simulates landing a small rocket on the moon surface. The environment for testing the algorithm is freely available on the Gymnasium web site (it's an actively maintained fork of the original OpenAI Gym developed by Oleg Klimov.


## Environment Details
In the simulation, the spacecraft has a main engine and two lateral boosters that can be used to control its descent and the orientation of the spacecraft. The spacecraft is subject to the moon's gravitational pull, and the engines have an unlimited amount of fuel. The spacecraft must navigate to the landing spot between two flags at coordinates (0,0) without crashing. Landing outside of the landing pad is possible. The lander starts at the top center of the viewport with a random initial force applied to its center of mass. T

### State Space (8 dimensions)
- x, y coordinates of the lander
- x, y velocity components
- Angle and angular velocity
- Left and right leg contact (boolean)

### Action Space (4 discrete actions)
- 0: Do nothing
- 1: Fire left orientation engine
- 2: Fire main engine
- 3: Fire right orientation engine

### Reward Structure
- Moving toward/away from landing pad: +/- small reward
- Crashing: -100 points
- Coming to rest: +100 points
- Each leg contact: +10 points
- Firing main engine: -0.3 points per frame
- Firing side engine: -0.03 points per frame

**Solved threshold:** Average reward of 200+ over 100 consecutive episodes

## Assignment Structure

### Part A: Environment Setup and Baseline (15 points)
Note: https://gymnasium.farama.org/introduction/basic_usage/ is a good place to start

**Tasks:**
1. Install gymnasium and create the LunarLander-v3 environment
2. Implement a random policy agent as baseline
3. Collect statistics over 100 episodes: mean reward, episode lengths, success rate
4. Visualize and save 3-5 episodes as videos/GIFs

**Deliverables:**
- Code for random agent
- Statistics report with plots
- Sample episode recordings

**Grading:**
- Correct environment setup (5 pts)
- Baseline statistics (5 pts)
- Visualizations (5 pts)

### Part B: DQN Implementation (40 points)
note: https://medium.com/@hkabhi916/mastering-deep-q-learning-with-pytorch-a-comprehensive-guide-a7e690d644fc is a good resource
**Tasks:**
1. Implement a neural network Q-function approximator
2. Implement experience replay buffer
3. Implement epsilon-greedy exploration
4. Implement target network with periodic updates
5. Train for at least 500 episodes

**Deliverables:**
- Complete DQN implementation
- Training loop with proper episode management
- Saved model checkpoints

**Grading:**
- Replay buffer implementation (8 pts)
- Q-network architecture (8 pts)
- Epsilon-greedy exploration (8 pts)
- Target network updates (8 pts)
- Training loop (8 pts)

### Part C: Training and Analysis (30 points)

**Tasks:**
1. Train your DQN agent and track key metrics:
   - Episode rewards over time
   - Epsilon decay curve
   - Loss values
   - Average Q-values
2. Plot learning curves with moving averages
3. Test the trained agent (no exploration) for 100 episodes
4. Record and submit 3-5 test episodes showing learned behavior

**Deliverables:**
- Training plots (rewards, loss, epsilon, Q-values)
- Test performance statistics
- Videos/GIFs of trained agent

**Grading:**
- Successful training (agent improves over time) (10 pts)
- Comprehensive plots and analysis (10 pts)
- Test performance documentation (10 pts)

### Part D: Experimentation and Report (15 points)

**Tasks:**
1. Experiment with at least 3 hyperparameter variations:
   - Learning rate (try: 1e-3, 5e-4, 1e-4)
   - Batch size (try: 32, 64, 128)
   - Epsilon decay rate
   - Network architecture (depth/width)
   - Target network update frequency
   - Replay buffer size

2. Write a 2-3 page report answering:
   - How does epsilon decay affect learning speed vs final performance?
   - What happens if you update the target network too frequently or too rarely?
   - Which hyperparameter had the biggest impact on performance?
   - What failure modes did you observe during training?
   - How does the learned policy compare to your intuition about optimal landing?

**Deliverables:**
- Comparison plots for different hyperparameters
- Written report with insights


## Starter Code
A template is provided, check "lunar_lander.py". In addition, I have also supplier a utils.py file with useful functions for completing this assignment.

**Recording Videos**

This snippet sets up the LunarLander environment with video recording. Note: `episode_trigger` to record only every 50th episode. This keeps disk usage manageable during long training runs.

```python
# record videos 
env = gym.make('LunarLander-v3', render_mode='rgb_array') 
env = RecordVideo(env, 'videos/', episode_trigger=lambda x: x % 50 == 0)
```

## Training notes

1. **Expect instability:** DQN training can be noisy; use moving averages to see trends
2. **Save checkpoints:** Save models every 50-100 episodes so you can recover from training collapse
3. **Debug with visualization:** Watch your agent to understand what it's learning
4. **Be patient:** It may take 300-500 episodes to see good performance

## Learning Objectives

- Understand the fundamental components of reinforcement learning: states, actions, rewards, and episodes
- Implement experience replay and target networks for stable DQN training
- Master the exploration-exploitation tradeoff using epsilon-greedy policies
- Analyze agent behaviour and learning curves
- Debug and tune hyperparameters for optimal performance


## Additional Reading
- [Gymnasium fork](https://gymnasium.farama.org/)
- [Gymnasium Tutorials](https://gymnasium.farama.org/introduction/basic_usage/)
- [Gymnasium LunarLander Docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Geeks for Geeks Tutorial](https://www.geeksforgeeks.org/deep-learning/deep-q-learning/)

## Submission Requirements

Submit a ZIP file containing:
1. All Python code files (`.py`)
2. Trained model weights (`.pt` or `.pth`)
3. Plots and figures (`.png` or `.pdf`)
4. Videos/GIFs of agent behavior (3-5 episodes)
5. Written report (`.pdf`)

## Grading Summary

| Component | Points |
|-----------|--------|
| Part A: Baseline | 15 |
| Part B: Implementation | 40 |
| Part C: Training & Analysis | 30 |
| Part D: Experimentation & Report | 15 |
| **Total** | **100** |
