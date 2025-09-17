
# CartPole RL Agents

This project implements and compares multiple Reinforcement Learning (RL) algorithms using neural networks to solve the classic CartPole-v1 environment from OpenAI Gymnasium. It demonstrates deep learning, RL, and experimentation with Q-Learning, SARSA, and Deep Q-Networks (DQN) using TensorFlow/Keras.

## Features
- **Multiple RL Algorithms:**
  - Q-Learning with Neural Networks
  - SARSA with Neural Networks
  - Deep Q-Network (DQN) with Experience Replay and Target Networks
- **Custom Policy Functions:** Epsilon-greedy exploration and policy improvement
- **Model Saving/Loading:** Trained models are saved for later evaluation
- **Visualization:** Optionally renders the CartPole environment for qualitative evaluation

## Algorithms Implemented
- `Q_learning.py`: Q-Learning with a neural network function approximator
- `sarsa.py`: SARSA with a neural network
- `dqn_cartpole.py`: Deep Q-Network with experience replay and target network
- `better.py`: Improved Q-Learning variant
- `random_agent.py`: Baseline random agent for comparison
- `evalutaor.py`: Evaluates saved models visually

## Technologies Used
- Python 3.x
- TensorFlow / Keras
- OpenAI Gymnasium
- NumPy
- (Optional) OpenCV for rendering

## How to Run
1. Install dependencies:
	```bash
	pip install tensorflow gymnasium keras opencv-python
	```
2. Run any agent script, e.g.:
	```bash
	python Q_learning.py
	python sarsa.py
	python dqn_cartpole.py
	python better.py
	```
3. To evaluate a trained model visually:
	```bash
	python evalutaor.py
	```

## Project Structure
- `Q_learning.py`, `sarsa.py`, `dqn_cartpole.py`, `better.py`: RL agent implementations
- `random_agent.py`: Baseline random agent
- `evalutaor.py`: Model evaluation/visualization
- `*.keras`: Saved neural network models

## About CartPole
CartPole is a classic RL benchmark where the goal is to balance a pole on a moving cart by applying left/right forces. [CartPole-v1 Gymnasium Docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## Author
Achyut Pandey

---
This project is ideal for your resume to showcase skills in deep learning, reinforcement learning, and practical experimentation with neural network-based agents.

