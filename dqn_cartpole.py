import tensorflow as tf
import gymnasium as gym
import keras

env = gym.make("CartPole-v1", render_mode="rgb_array")
net_input = keras.Input(shape=(4,))
x = keras.layers.Dense(32, activation="relu")(net_input)
x = keras.layers.Dense(16, activation="relu")(x)
output = keras.layers.Dense(2, activation="linear")(x)
q_net = keras.Model(inputs=net_input, outputs=output)
q_net.compile(optimizer = "adam")
loss_fn = keras.losses.MeanSquaredError()

target_net = tf.keras.models.clone_model(q_net)


# parametres
ALPHA=0.001
EPSILON=1.0
EPSILON_DECAY=1.005
GAMMA=0.99
NUM_EPISODES=300
REPLAY_BUFFER = []
MAX_TRANSITION = 100000
BATCH_SIZE=16
STEP_COUNTER=0
UPDATE_FREQ=4
def insert_transition(transition):
    if(len(REPLAY_BUFFER)>=MAX_TRANSITION):
        REPLAY_BUFFER.pop(0)
    REPLAY_BUFFER.append(transition)

def sample_transition(batch_size=16):
    random_indices = tf.random.uniform(shape=(batch_size, ), minval=0, maxval=len(REPLAY_BUFFER), dtype=tf.int32)
    sampled_current_state=[]
    sampled_current_action=[]
    sampled_current_reward=[]
    sampled_next_state = []
    sampled_current_terminal = []
    for index in random_indices:
        sampled_current_state.append(REPLAY_BUFFER[index][0])
        sampled_current_action.append(REPLAY_BUFFER[index][1])
        sampled_current_reward.append(REPLAY_BUFFER[index][2])
        sampled_next_state.append(REPLAY_BUFFER[index][3])
        sampled_current_terminal.append(REPLAY_BUFFER[index][4])

    return tf.convert_to_tensor(sampled_current_state), tf.convert_to_tensor(sampled_current_action
            ), tf.convert_to_tensor(sampled_current_reward), tf.convert_to_tensor(sampled_next_state
            ), tf.convert_to_tensor(sampled_current_terminal)
def policy(state, explore=0.0):
    state_input = tf.expand_dims(tf.convert_to_tensor(state, dtype = tf.float32), axis=0)  # (1, 4)
    q_values = q_net(state_input)
    action = tf.argmax(q_values[0], output_type=tf.int32)
    # action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if(tf.random.uniform(shape=(), maxval=1)<=explore):
        action=tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action



for episode in range(NUM_EPISODES):
    done=False
    #state=env.reset()
    total_rewards=0
    episode_len=0
    obs, _ = env.reset()  # Extract only the observation
    state = tf.convert_to_tensor(obs, dtype=tf.float32)
    action = policy(state, EPSILON)
    while not done:
        action = policy(state, EPSILON)
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        done=terminated or truncated

        insert_transition([state, action, reward, next_state, done])
        state=next_state
        sample_curr_state, sample_curr_action, sample_curr_reward, sample_next_state, sample_curr_terminal = sample_transition(BATCH_SIZE)
        STEP_COUNTER+=1
        sample_curr_state = tf.reshape(sample_curr_state, (16, 4))
        next_action_values = tf.reduce_max(target_net(sample_next_state), axis=1)
        targets = tf.where(sample_curr_terminal, sample_curr_reward, sample_curr_reward+ GAMMA*next_action_values)
        with tf.GradientTape() as tape:
            pred = q_net(sample_curr_state)
            batch_nums = tf.range(0, limit = BATCH_SIZE)
            indices = tf.stack((batch_nums, sample_curr_action), axis=1)
            current_values = tf.gather_nd(pred, indices)
            loss = loss_fn(targets, current_values)
        grads = tape.gradient(loss, q_net.trainable_weights)
        q_net.optimizer.apply_gradients(zip(grads, q_net.trainable_weights))
        if STEP_COUNTER % UPDATE_FREQ == 0:
            target_net.set_weights(q_net.get_weights())

        total_rewards += reward
        episode_len += 1
    print("Episode: ", episode, "Reward: ", total_rewards, "Ep_len: ", episode_len, "Epsilon: ", EPSILON)
    EPSILON /= EPSILON_DECAY

env.close()
q_net.save("dqn_q_net.keras")
print("Q_net saved...")


