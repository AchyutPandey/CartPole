import tensorflow as tf
import gymnasium as gym
import keras

env = gym.make("CartPole-v1", render_mode="rgb_array")

# Q Network
net_input = keras.Input(shape=(4,))
x = keras.layers.Dense(64, activation="relu")(net_input)
x = keras.layers.Dense(32, activation="relu")(x)
#x = keras.layers.Dense(16, activation="relu")(x)
output = keras.layers.Dense(2, activation="linear")(x)

q_net = keras.Model(inputs=net_input, outputs=output)
# Parameters
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 1.001
GAMMA = 0.99
NUM_EPISODES = 500


def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action


for episode in range(NUM_EPISODES):

    done = False
    total_reward = 0
    episode_length = 0
    obs, _ = env.reset()  # Extract only the observation
    state = tf.convert_to_tensor([obs])
    action = policy(state, EPSILON)
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        next_state = tf.convert_to_tensor([next_state])
        next_action = policy(next_state, EPSILON)
        done=terminated or truncated
        target = reward + GAMMA * q_net(next_state)[0][next_action]
        if done:
            target = reward

        with tf.GradientTape() as tape:
            current = q_net(state)

        grads = tape.gradient(current, q_net.trainable_weights)
        delta = target - current[0][action]
        for j in range(len(grads)):
            q_net.trainable_weights[j].assign_add(ALPHA * delta * grads[j])

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1
    print("Episode:", episode, "Length:", episode_length, "Rewards:", total_reward, "Epsilon:", EPSILON)
    EPSILON /= EPSILON_DECAY
q_net.save("better.keras")
env.close()