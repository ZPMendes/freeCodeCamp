import random
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np



def reset_game():  # Reset all the variables to play with a new bot
    global data_dict, trained_model, last_guess
    data_dict = {'bot_move': [], 'player_move': []}
    trained_model = None
    last_guess = 'R'
    opponent_history = []


reset_game()

opponent_history = []
categoric_transformer = OneHotEncoder(sparse=False)

def player(prev_play, opponent_history=[]):
    global trained_model
    global last_guess
    guess = 'R'  # Default value if something fails
    sequence_length = 10

    if not prev_play:
        reset_game()

    if prev_play:
        opponent_history.append(prev_play)
        # Only add to bot_move if we're also adding a player move
        data_dict['bot_move'].append(prev_play)
        data_dict['player_move'].append(last_guess)

    if len(opponent_history) < 300:
        # Make a random choice to collect data
        guess = random.choice(['R', 'P', 'S'])

    elif len(opponent_history) == 300:
        combined_sequences, bot_sequences = preprocess_data(data_dict, sequence_length)
        # Train the model
        input_shape = (sequence_length, 6)  # One-hot encoded with 6 features (3 player, 3 bot)
        model = create_model(input_shape)
        model.fit(combined_sequences, bot_sequences, epochs=10, batch_size=32)
        trained_model = model
        guess = random.choice(['R', 'P', 'S'])

    if len(opponent_history) > 300:
        #Get the last sequences of moves:
        last_player_moves = data_dict['player_move'][-sequence_length:]
        last_bot_moves = data_dict['bot_move'][-sequence_length:]

        #Encode the moves
        last_player_moves_encoded = categoric_transformer.transform(np.array(last_player_moves).reshape(-1,1))
        last_bot_moves_encoded = categoric_transformer.transform(np.array(last_bot_moves).reshape(-1,1))
        
        #Stack those sequences of moves to pass to the model
        input_sequence = np.hstack([last_player_moves_encoded, last_bot_moves_encoded])
        input_sequence = input_sequence.reshape(1, sequence_length, 6)
        
        # Add a random element to the counter-move selection
        random_factor = 0.2  # Tune this to add some randomness (20% randomness in this case)
        if np.random.rand() < random_factor:
            guess = random.choice(['R', 'P', 'S'])  # Add randomness to avoid being predictable
        else:
            # Predict bot's next move using the model
            prediction = trained_model.predict(input_sequence)
            prediction_index = np.argmax(prediction)  # Get the index of the highest probability
            prediction_decoded = ['R', 'P', 'S'][prediction_index]

            # Choose a move to beat the bot's predicted move
            if prediction_decoded == 'R':
                guess = 'P'
            elif prediction_decoded == 'P':
                guess = 'S'
            else:
                guess = 'R'

    last_guess = guess

    return guess


def preprocess_data(data_dict, sequence_length):
    global categoric_transformer
    categoric_transformer = OneHotEncoder(sparse=False)

    bot_moves = np.array(data_dict['bot_move']).reshape(-1, 1)
    player_moves = np.array(data_dict['player_move']).reshape(-1, 1)

    categoric_transformer.fit(np.array(['R', 'P', 'S']).reshape(-1, 1))
    
    # One-hot encode the moves
    player_moves_encoded = categoric_transformer.transform(player_moves)
    bot_moves_encoded = categoric_transformer.transform(bot_moves)
    
    #Create sequences for LSTM input
    combined_sequences = []
    bot_sequences = []
    
    for i in range(len(player_moves_encoded) - sequence_length):
        player_sequence = player_moves_encoded[i:i + sequence_length]
        bot_sequence = bot_moves_encoded[i:i + sequence_length]
        
        # Concatenate player and bot moves for each time step in the sequence
        combined_sequence = np.hstack([player_sequence, bot_sequence])
        combined_sequences.append(combined_sequence)
        
        # The next bot move (the move to predict)
        bot_sequences.append(bot_moves_encoded[i + sequence_length])

    return np.array(combined_sequences), np.array(bot_sequences)

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
