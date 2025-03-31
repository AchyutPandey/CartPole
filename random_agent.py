import tensorflow as tf
import gymnasium as gym
import cv2

env = gym.make("CartPole-v1", render_mode="rgb_array")
for episode in range (5):
    terminated=False
    truncated=False
    env.reset()
    while not (terminated or truncated):
        frame = env.render()
        cv2.imshow("CartPole", frame)
        cv2.waitKey(100)
        action=tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        env.step(action)
        state, reward, terminated, truncated, _ = env.step(action)

env.close()

