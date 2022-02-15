import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
from ckpt_manager import CheckpointManager

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gets all console-printable characters
all_characters = string.printable

# Number of characters
n_characters = len(all_characters)

# Opens the text file
data = unidecode.unidecode(open('data/city_names.txt').read())

# The RNN model
class RNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # input_size, hidden_size
        # Creates the trainable "dictionary" of sorts
        self.embed = nn.Embedding(
            num_embeddings=embedding_size,
            embedding_dim=hidden_size
        )

        # hidden_size, hidden_size, num_layers, batch_first=True
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Final dense layer
        self.dense = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )
    
    # The idea of forward propagation in RNNs is to output the prediction AND the hidden state
    # With LSTM, the cell is also an output since the cell states are trainable (the input gate, forget gate, output gate)
    def forward(self, x, hidden, cell):
        # Pass through embedding layer
        out = self.embed(x)

        # Pass the embedding output to the LSTM layers along with the hidden and cell states
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))

        # Pass only the output to the dense layer (the hidden and cell states aren't trained by the dense layer)
        out = self.dense(out.reshape(out.shape[0], -1))

        # Return the output and the hidden and cell states
        # The output is a set of probabilities for each printable character
        return out, (hidden, cell)
    
    # Method for initializing the hidden and cell states
    def init_hidden(self, batch_size):
        # First three arguments are for the size
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # Returns the initialized hidden and cell states
        return hidden, cell

# Class for training and creating predictions
class Generator():
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_interval = 50
        self.save_interval = 200
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.001
    
    # Tokenizes the string into numbers
    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for i, c in enumerate(string):
            tensor[i] = all_characters.index(c)
        
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(data)-self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1

        # Gets a random section of the data of length self.chunk_len
        text_str = data[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len, device=device)
        text_target = torch.zeros(self.batch_size, self.chunk_len, device=device)

        # Tokenizes the data chunk into integers
        for i in range(self.batch_size):
            # Populates each batch with a substring (although it doesn't really matter since we use a batch size of 1)
            # Hello -> Hell
            text_input[i,:] = self.char_tensor(text_str[:-1])

            # Hello -> ello
            text_target[i,:] = self.char_tensor(text_str[1:])
        
        # Returns the input and the target
        return text_input.long(), text_target.long()

    def generate(self, initial_str='A', prediction_len = 100, temperature=0.85):
        # Initializes empty cell states
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

        # Creates a root character for the generator to go off of
        initial_inp = self.char_tensor(initial_str)

        # Creates the string that will be appended over time
        predicted = initial_str

        # Build up the hidden and cell states until the last initial string character
        for p in range(len(initial_str) - 1):
            # Actual prediction doesn't matter since we're only building the hidden and cell states
            _, (hidden, cell) = self.rnn(initial_inp[p].view(1).to(device), hidden, cell)
        
        last_char = initial_inp[-1]

        for p in range(prediction_len):
            # Get the output and states using the last generated character as input
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)

            # Creates a temperature-adjusted output distribution
            output_dist = output.data.view(-1).div(temperature).exp()

            # Samples a single element from the output distribution (this is what produces the randomness in the final output)
            top_char = torch.multinomial(output_dist, 1)[0]

            # torch.multinomial returns an index, so we get the character from the character list
            predicted_char = all_characters[top_char]

            # Append the character to the final string
            predicted += predicted_char

            # Set the generated character as the last character
            last_char = self.char_tensor(predicted_char)
        
        # Return the compiled string
        return predicted
    
    def create_manager(self, model, optim, directory, file_name, maximum=3, file_format='pt'):
        self.manager = CheckpointManager(
            assets={
                'model' : model.state_dict(),
                'optimizer' : optim.state_dict()
            },
            directory=directory,
            file_name=file_name,
            maximum=maximum,
            file_format=file_format
        )
    
    def train(self):
        # Creates the RNN
        # n_characters, self.hidden_size, self.num_layers, n_characters
        self.rnn = RNN(
            embedding_size=n_characters,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=n_characters
        ).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        loss_function = nn.CrossEntropyLoss()

        self.create_manager(
            model=self.rnn,
            optim=optimizer,
            directory='saved_models',
            file_name='text-generator'
        )

        load_data = self.manager.load()
        self.rnn.load_state_dict(load_data['model'])
        optimizer.load_state_dict(load_data['optimizer'])

        print('Starting training...')

        # Starts the training loop
        for epoch in range(self.num_epochs):
            # Gets a random input and its respective target
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            # Zeroes the gradient
            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            # Get the loss for each character in the data chunk
            for i, c in enumerate(inp[0]):
                prediction, (hidden, cell) = self.rnn(c.unsqueeze(0), hidden, cell)
                loss += loss_function(prediction, target[0][i].unsqueeze(0))
            
            # Do backprop on the cumulative loss
            loss.backward()
            optimizer.step()

            # Get the average loss for logging
            loss = loss.item() / self.chunk_len

            # Print the stats and generate samples every so often
            if (epoch + 1) % self.print_interval == 0:
                print(f'Epoch: {epoch+1} || Loss: {loss}')
                print(self.generate())
            
            if (epoch + 1) % self.save_interval == 0:
                self.manager.save()

generator = Generator()
generator.train()