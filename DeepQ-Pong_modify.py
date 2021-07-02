import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import skimage
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
import os.path
import os, time

def gettime(style="D%m-%d_T%H-%M-%S"):
    timestr=time.strftime(style)
    return timestr

# Define Hyperparameters
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float('optimiser_learning_rate', 0.0001, help='learning rate for optimiser')
tf.flags.DEFINE_integer('observe_step_num', 10000, help='number of steps to observe before choosing actions')
tf.flags.DEFINE_integer('batch_size', 32, help='size of batch')
tf.flags.DEFINE_float('initial_epsilon', 1.0, help='initial value for epsilon')
tf.flags.DEFINE_integer('epsilon_anneal_num', 500000, help='number of steps over to anneal epsilon')
tf.flags.DEFINE_float('final_epsilon', 0.01, help='final value for epsilon')
tf.flags.DEFINE_float('gamma', 0.99, help='decay rate for future reward')
tf.flags.DEFINE_integer('replay_memory', 200000, 'number of previous transitions to remember')  # 200000 = 10GB RAM
tf.flags.DEFINE_integer('n_episodes', 1000, 'number of episodes to let the model train for')
tf.flags.DEFINE_integer('no_op_steps', 2, 'number of steps to do nothing at start of each episode')
tf.flags.DEFINE_integer('update_target_model_steps', 10000, 'update target Q model every n episodes')
tf.flags.DEFINE_string('train_dir', 'train_data_' + gettime(), 'location for training data and logs')
# tf.flags.DEFINE_boolean('render', True, 'whether to render the image')
tf.flags.DEFINE_boolean('render', False, 'whether to render the image')

Input_shape = (84, 84, 4)  # input image size to model
Action_size = 3

# Create a pre-processing function
# Converts colour image into a smaller grey-scale image
# Converts floats to integers
def pre_processing(observation):
    processed_observation = rgb2gray(observation) * 255
    # Convering to grey converts it to normalised values [0,255] -> [0,1]
    # We need to store the values as integers in the next step to save space, so we need to undo the normalisation
    processed_observation = skimage.transform.resize(processed_observation, (84, 84), mode='constant')
    # Convert from floating point to integers ranging from [0,255]
    processed_observation = np.uint8(processed_observation)
    return processed_observation

# Define the huber loss function
# Used for the Keras model compile function
# Needs to take y and yhat and output loss
# Needs investigation
def huber_loss(y, q_value):
    error = tf.abs(y - q_value)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
    return loss

# Create the model to approximate Qvalues for a given set of states and actions
def atari_model(lr=0.001, resume=False, path_model=None):
    # Define the inputs
    frames_input = keras.Input(shape=Input_shape, name='frames')
    actions_input = keras.layers.Input(shape=(Action_size,), name='action_mask')
    # Normalise the inputs from [0,255] to [0,1] - to make processing easier
    normalised = keras.layers.Lambda(lambda x: x/255.0, name='norm')(frames_input)
    # Conv1 is 16 8x8 filters with a stride of 4 and a ReLU
    conv1 = keras.layers.Conv2D(16, 8, 4, activation='relu')(normalised)
    # Conv2 is 32 4x4 filters with a stride of 2 and a ReLU
    conv2 = keras.layers.Conv2D(32, 4, 2, activation='relu')(conv1)
    # Flatten the output from Conv2
    conv2_flatten = keras.layers.Flatten()(conv2)
    # Then a fully connected layer with 256 ReLU units
    dense1 = keras.layers.Dense(256, activation='relu')(conv2_flatten)
    # Then a fully connected layer with a unit to map to each of the actions and no activation
    output = keras.layers.Dense(Action_size)(dense1)
    # Then we multiply the output by the action mask
    # When trying to find the value of all the actions this will be a mask full of 1s
    # When trying to find the value of a specific action, the mask will only be 1 for a single action
    filtered_output = keras.layers.Multiply(name='Qvalue')([output, actions_input])

    # Create the model
    # Create a model that maps frames and actions to the filtered output
    model = keras.Model(inputs=[frames_input, actions_input],
                        outputs=filtered_output)
    # Print a summary of the model
    model.summary()
    # Define optimiser
    # optimiser = tf.train.AdamOptimizer(learning_rate=lr) lr=0.0001/0.00025 score=-21
    optimiser = tf.train.AdadeltaOptimizer(learning_rate=1.0) # score = 21
    # Compile model
    model.compile(optimizer=optimiser, loss=huber_loss)
    # Return the model
    if resume == True:
        model.load_weights(path_model)
    return model

# get action from model using epsilon-greedy policy
def get_action(history, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= FLAGS.observe_step_num:
        return random.randrange(Action_size)
    else:
        q_value = model.predict([history,
                                 np.ones(Action_size).reshape(1, Action_size)])
        return np.argmax(q_value[0])

# save sample <s,a,r,s'> to the replay memory
def store_memory(memory, history, action, reward, next_history):
    memory.append((history, action, reward, next_history))

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

# train model by random batch
def train_memory_batch(memory, model, model_target):
    # Sample a minibatch
    mini_batch = random.sample(memory, FLAGS.batch_size)
    # Create empty arrays to load our minibatch into
    # These objects have multiple values hence need defined shapes
    state = np.zeros((FLAGS.batch_size, Input_shape[0], Input_shape[1],
                      Input_shape[2]))
    next_state = np.zeros((FLAGS.batch_size, Input_shape[0], Input_shape[1],
                           Input_shape[2]))
    # These objects have a single value, so we can just create a list that we append later
    action = []
    reward = []
    # Create an array that will carry what the target q values will be - based on our target networks weights
    target_q = np.zeros((FLAGS.batch_size,))

    # Fill up our arrays with our minibatch
    for id, val in enumerate(mini_batch):
        state[id] = val[0]
        # print(val[0].shape)
        next_state[id] = val[3]
        action.append(val[1])
        reward.append(val[2])

    # We want the model to predict the q value for all actions hence:
    actions_mask = np.ones((FLAGS.batch_size, Action_size))
    # Get the target model to predict the q values for all actions
    next_q_values = model.predict([next_state, actions_mask])
    next_q_actions = np.argmax(next_q_values, axis=1)
    next_q_values_target = model_target.predict([next_state, actions_mask])

    # Fill out target q values based on the max q value in the next state
    for i in range(FLAGS.batch_size):
        # Standard discounted reward formula
        # q(s,a) = r + discount * cumulative future rewards
        next_action = next_q_actions[i]
        target_q[i] = reward[i] + FLAGS.gamma * next_q_values_target[i][next_action]

    # Convert all the actions into one hot vectors
    action_one_hot = get_one_hot(action, Action_size)
    # Apply one hot mask onto target vector
    # This results in a vector that has the max q value in the position
    # corresponding to the action
    target_one_hot = action_one_hot * target_q[:, None]

    # Then we fit the model
    # We map the state and the action from the memory bank to the q value
    # of that state action pair
    # s,a -> q(s,a|w)
    h = model.fit([state, action_one_hot], target_one_hot, epochs=1,
                  batch_size=FLAGS.batch_size, verbose=0)

    # Return the loss
    # It's just for monitoring progress
    return h.history['loss'][0]

def train(resume=False,path_model=None):
    # Define which game to play
    env = gym.make('PongDeterministic-v4')

    # Create a space for our memory
    # We will use a deque - double ended que
    # This will result in a max size so that once all the space is filled,
    # older entries will be removed to make room for new
    memory = deque(maxlen=FLAGS.replay_memory)

    # Start episode counter
    episode_number = 0
    # Set epsilon
    epsilon = FLAGS.initial_epsilon
    # Define epsilon decay
    epsilon_decay = (FLAGS.initial_epsilon - FLAGS.final_epsilon) / FLAGS.epsilon_anneal_num

    # Start global step
    global_step = 0

    lr = FLAGS.optimiser_learning_rate
    # Define model
    model = atari_model(lr=lr, resume=resume, path_model=path_model)

    # Define target model
    model_target = atari_model(lr=lr, resume=resume, path_model=path_model)
    # use the same weight from the beginning
    model_target.set_weights(model.get_weights())

    # Define where to store logs
    log_dir = "{}/run-{}-log".format(FLAGS.train_dir, 'MK10')
    # Pass graph to TensorBoard
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    # Start the optimisation loop
    while episode_number < FLAGS.n_episodes:
        # Initialisise done as false
        done = False
        step = 0
        score = 0
        loss = 0.0

        # Initialise environment
        observation = env.reset()

        # For the very start of the episode, we will do nothing but observe
        # This way we can get a sense of what's going on
        for _ in range(random.randint(1, FLAGS.no_op_steps)):
            observation, _, _, _ = env.step(1)

        # At the start of the episode there are no preceding frames
        # So we just copy the initial states into a stack to make the state history
        state = pre_processing(observation)
        state_history = np.stack((state, state, state, state), axis=2)
        state_history = np.reshape([state_history], (1, 84, 84, 4))

        # Perform while we still have lives
        while not done:
            # Render the image if selected to do so
            if FLAGS.render:
                env.render()

            # Select an action based on our current model
            action = get_action(state_history, epsilon, global_step, model)
            # Convert action from array numbers to real numbers
            real_action = action + 1

            # After we're done observing, start scaling down epsilon
            if global_step > FLAGS.observe_step_num and epsilon > FLAGS.final_epsilon:
                epsilon -= epsilon_decay

            # Record output from the environment
            observation, reward, done, info = env.step(real_action)

            # Process the observation
            next_state = pre_processing(observation)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            # Update the history with the next state - also remove oldest state
            state_history_w_next = np.append(next_state, state_history[:, :, :, :3], axis=3)

            # Update score
            score += reward
            # Save the (s, a, r, s') set to memory
            store_memory(memory, state_history, action, reward, state_history_w_next)

            # Train model
            # Check if we are done observing
            if global_step > FLAGS.observe_step_num:
                loss_batch = train_memory_batch(memory, model, model_target)
                loss = loss + loss_batch
                # Check if we are ready to update target model with the model we have been training
                if global_step % FLAGS.update_target_model_steps == 0:
                    model_target.set_weights(model.get_weights())
                    print("UPDATING TARGET WEIGHTS")
            state_history = state_history_w_next

            #print("step: ", global_step)
            global_step += 1
            step += 1

            # Check if episode is over - lost all lives in breakout
            if done:
                # Check if we are still observing
                if global_step <= FLAGS.observe_step_num:
                    current_position = 'observe'
                # Check if we are still annealing epsilon
                elif FLAGS.observe_step_num < global_step <= FLAGS.observe_step_num + \
                        FLAGS.epsilon_anneal_num:
                    current_position = 'explore'
                else:
                    current_position = 'train'
                # Print status
                print(
                    'current position: {}, epsilon: {} , episode: {}, score: {}, global_step: {}, '
                    'avg loss: {}, step: {}, memory length: {}'.format(current_position, epsilon,
                     episode_number, score, global_step, loss / float(step), step, len(memory)))

                # Save model every 100 episodes and final episode
                if episode_number % 100 == 0 or (episode_number + 1) == FLAGS.n_episodes:
                    file_name = "pong_model_{}.h5".format(episode_number)
                    model_path = os.path.join(FLAGS.train_dir, file_name)
                    model.save(model_path)

                # Add loss and score  data to TensorBoard
                loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="loss", simple_value=loss / float(step))])
                file_writer.add_summary(loss_summary, global_step=episode_number)

                score_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="score", simple_value=score)])
                file_writer.add_summary(score_summary, global_step=episode_number)

                # Increment episode number
                episode_number += 1

    file_writer.close()

if __name__ == "__main__":
    t_start = time.time()
    resume = False
    path_model = None
    os.environ["CUDA_VISIBLE_DEVICES"] = '4' # score =21 after 1000 epsides
    # resume = True
    # path_model = 'pong_model_checkpoint.h5'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    train(resume=resume, path_model=path_model)
    t_end = time.time()
    t_all = t_end - t_start
    print('train.py: whole time: {:.2f} h ({:.2f} min)'.format(t_all / 3600., t_all / 60.))
