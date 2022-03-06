import torch
from torch import nn
from torch import optim
from torch import argmax
import numpy as np
import random
import string
from collections import Counter
from torch.nn.functional import one_hot
import random
import math
import shutil

def experiment(use_checkpoint, nb_epochs, message_length, sample_position):
    '''
    args:
    use checkpoint is a bool that gives the user the option to load a checkpoint
    if one exists
    sample_position is the position in the message at which the LSTM takes over
    from the actual message in generating a sensible conclusion to the rest of
    the message. If sample_position is larger than the message_length the code
    won't work.
    '''
    if sample_position > message_length:
        return """This won't predict any code as the sample_position is larger
                 than the message length"""
    no_of_test_batches = 100
    no_of_train_batches = 1000
    batch_size = 32
    no_of_test_sequences = batch_size * no_of_test_batches
    dataset = Dataset(use_checkpoint, message_length, no_of_test_sequences)
    starting_epoch = 0
    test_loss = 99
    loss = nn.CrossEntropyLoss()
    latent_size = 128
    input_size = len(dataset.list_of_characters)
    n_layers = 1
    no_of_samples = 50
    character_predictor = Model(input_size, latent_size, n_layers)
    optimiser = optim.Adam(character_predictor.parameters(), lr= 1e-3)
    if use_checkpoint:
        try:
            checkpoint = torch.load('LSTM_checkpoints/LSTM_model.pt')
            character_predictor.load_state_dict(checkpoint['model_state_dict'])
            optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            starting_epoch = checkpoint['epoch']
            test_loss = checkpoint['test_loss']
        except:
            pass

    for epoch in range(nb_epochs):
        new_test_loss = test(character_predictor, loss,
                             dataset, batch_size, no_of_test_batches)
        print(f'Epoch {starting_epoch+ epoch+ 1},test loss:{new_test_loss:.2f}')
        if new_test_loss > test_loss:
            print('This is now overfitting to the training data')
            shutil.copyfile('LSTM_checkpoints/LSTM_model.pt',
                            'LSTM_checkpoints/Super_trained_LSTM.pt')
            break
        test_loss = new_test_loss
        average_loss = train(character_predictor, loss, optimiser, dataset,
                             batch_size, no_of_train_batches)
        print(f'Epoch {starting_epoch+ epoch+ 1},train loss:{average_loss:.2f}')
        torch.save({
                'epoch': epoch+ 1+ starting_epoch,
                'model_state_dict': character_predictor.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'loss': loss,
                'test_loss': test_loss },
                'LSTM_checkpoints/LSTM_model.pt')
    sample(character_predictor, sample_position, no_of_samples, dataset)

class Dataset:
    def __init__(self, use_checkpoint, message_length, no_of_test_sequences):
        self.no_of_characters, self.list_of_characters, self.txt = get_data()
        self.num_of_unique_characters = len(self.list_of_characters)
        self.message_length = message_length
        loaded = False
        if use_checkpoint:
            try:
                dataset = torch.load('LSTM_checkpoints/LSTM_dataset.pt')
                self._sequence = dataset['test_batches']
                self._forbidden_indices = dataset['test_indices']
                loaded = True
            except:
                pass
        if not loaded:
            one_hot_sequence = []
            for j in range(no_of_test_sequences):
                random_indices = []
                one_hot_indices, random_index_test = self._get_sequence()
                random_indices.append(random_index_test)
                one_hot_tensor = torch.FloatTensor(one_hot_indices)
                one_hot_sequence.append(one_hot(one_hot_tensor.to(torch.int64),
                                                len(self.list_of_characters)))
            batch = torch.stack(one_hot_sequence)
            self._sequence = batch
            self._forbidden_indices = random_indices
        torch.save({
                'test_batches': self._sequence,
                'test_indices': self._forbidden_indices },
                'LSTM_checkpoints/LSTM_dataset.pt')

    def get_batch(self, batch_type, batch_size):
        one_hot_sequence = []
        for i in range(batch_size):
            if batch_type == 'test':
                test_index = random.choice(self._forbidden_indices)
                one_hot_indices, test_index = self._get_sequence(test_index)
            else:
                self._forbidden_indices.sort()
                incomplete = True
                while incomplete:
                    incomplete = False
                    one_hot_indices, random_index_train = self._get_sequence()
                    for i in self._forbidden_indices:
                        if abs(random_index_train- i) < self.message_length:
                            incomplete = True
            one_hot_tensor = torch.FloatTensor(one_hot_indices)
            one_hot_sequence.append(one_hot(one_hot_tensor.to(torch.int64),
                                            len(self.list_of_characters)))
        batch = torch.stack(one_hot_sequence)
        return batch

    def _get_sequence(self, index= None):
        if index is None:
            index = np.random.choice(self.no_of_characters- self.message_length)
        one_hot_indices = []
        for i in range(self.message_length):
            random_character = self.txt[index+ i]
            one_hot_indices.append(
                self.list_of_characters.index(random_character)
            )
        return one_hot_indices, index

def get_data():
    txt_name = 'CompleteShakespeare.txt'
    with open(txt_name) as f:
        txt = f.read()
    txt = remove_undesirables(txt)
    no_of_characters = len(txt)
    list_of_characters = get_vocab(txt)
    return no_of_characters, list_of_characters, txt

def remove_undesirables(txt):
    replacements = ['\n', '\r', '\xa0', '_', '*', '--', '\ufeff']
    digits = set(string.digits)
    for digit in digits:
        txt = txt.replace(digit, ' ')
    for replacement in replacements:
        txt = txt.replace(replacement, ' ')
    while '  ' in txt:
        txt = txt.replace('  ', ' ')
    return list(txt)

def get_vocab(text):
    character_freq = Counter(text)
    sorted_character_freq = sorted(character_freq, key=character_freq.get,
                                   reverse=True)
    return sorted_character_freq

class Model(nn.Module):
    def __init__(self, input_size, latent_size, n_layers):
        super().__init__()

        self.encoder = nn.Linear(input_size, latent_size)
        self.LSTM = nn.LSTM(latent_size, latent_size,
                            n_layers, batch_first= True)
        self.decoder = nn.Linear(latent_size, input_size)
        self.n_layers = n_layers
        self.latent_size = latent_size

    def forward(self, batch, state):
        batch = batch.type(torch.FloatTensor)
        b, t, n = batch.shape
        batch = batch.reshape(b*t, n)
        encoded_batch = self.encoder(batch)
        encoded_batch = encoded_batch.reshape(b, t, self.latent_size)
        LSTM_batch, state = self.LSTM(encoded_batch, state)
        LSTM_batch = LSTM_batch.reshape(b*t, self.latent_size)
        decoded_batch = self.decoder(LSTM_batch)
        decoded_batch = decoded_batch.reshape(b, t, n)
        return decoded_batch, state

    def initial_state(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.latent_size),
                torch.zeros(self.n_layers, batch_size, self.latent_size))

def test(model, loss, dataset, batch_size, no_of_test_batches):
    model.eval()
    losses_test = list()
    for i in range(no_of_test_batches):
        x = dataset.get_batch('test', 32)
        targets = x[:, 1:, :] #inputs
        b, t, n = targets.shape
        targets = targets.reshape(b*t, n)
        targets = torch.argmax(targets, dim= 1)
        initial_state = model.initial_state(batch_size)
        x_hat, state = model(x, initial_state)
        predictions = x_hat[:, :-1, :]  #outputs
        predictions = predictions.reshape(b*t, n)
        J = loss(predictions, targets)
        losses_test.append(J.item())
    average_loss = torch.tensor(losses_test).mean()
    return average_loss

def train(model, loss, optimiser, dataset, batch_size, no_of_train_batches):
    model.train()
    losses_train = list()
    for i in range(no_of_train_batches):
        x = dataset.get_batch('train', 32)
        targets = x[:, 1:, :] #inputs
        b, t, n = targets.shape
        targets = targets.reshape(b*t, n)
        targets = torch.argmax(targets, dim= 1)
        initial_state = model.initial_state(batch_size)
        x_hat, state = model(x, initial_state)
        predictions = x_hat[:, :-1, :]  #outputs
        predictions = predictions.reshape(b*t, n)
        J = loss(predictions, targets)
        model.zero_grad()
        J.backward()
        optimiser.step()
        losses_train.append(J.item())
    average_loss = torch.tensor(losses_train).mean()
    return average_loss

def sample(model, sample_position, no_of_samples, dataset):
    model.eval()
    x = dataset.get_batch('test', 1)
    context = x[:, :sample_position, :]
    context_batch = context.repeat(no_of_samples, 1, 1)
    initial_state = model.initial_state(no_of_samples)
    context_hat, state = model(context_batch, initial_state)
    hs, cs = state
    particles = [PartialCompletion((hs[0], cs[0]), 0,
                 torch.unsqueeze(context_hat[i, sample_position- 1, :], 0),
                 context_batch[i, :, :]) for i in range(no_of_samples)]
    list_of_particles = particles
    batch_logits = list_of_particles[0].logits
    for particle in list_of_particles[1:]:
        batch_logits = torch.cat((batch_logits, particle.logits), 0)
    for i in range(dataset.message_length- sample_position):
        list_of_states = get_statelist_from_batch(state)
        for index, particle in enumerate(list_of_particles):
            particle.state = list_of_states[index]
            particle.logits = batch_logits[index]
        if i != dataset.message_length - sample_position:
            list_of_particles = beamsearch(list_of_particles, no_of_samples,
                                           dataset.num_of_unique_characters)
            state = get_batchstate_from_list(list_of_particles)
            batch_logits = list_of_particles[0].logits
            for particle in list_of_particles[1:]:
                batch_logits = torch.cat((batch_logits, particle.logits), 0)
            batch_logits = torch.unsqueeze(batch_logits, 1)
            batch_logits, state = model(batch_logits, state)
            batch_logits = torch.squeeze(batch_logits)
    display(x, context, list_of_particles, dataset.list_of_characters)
    return

class Particle():
    def successors(self):
        pass

class PartialCompletion(Particle):
    def __init__(self, state, score, logits, one_hots):
        #super().__init__() not needed because particle doesn't have an init
        self.state = state
        self.score = score
        self.logits = logits
        self.one_hots = one_hots

    def successors(self, num_of_unique_characters):
        score_one = self.score
        score_two = self.score
        Softmax = nn.Softmax(dim= 0)
        Softmax_logits = Softmax(self.logits)
        sample = torch.multinomial(torch.squeeze(Softmax_logits), 2)
        one_hot_one = torch.unsqueeze(one_hot(sample[0],
                                      num_of_unique_characters), 0)
        one_hot_two = torch.unsqueeze(one_hot(sample[1],
                                      num_of_unique_characters), 0)
        score_one += math.log(Softmax_logits[sample[0]])
        score_two += math.log(Softmax_logits[sample[1]])
        one_hots_one = torch.cat((self.one_hots, one_hot_one), 0)
        one_hots_two = torch.cat((self.one_hots, one_hot_two), 0)
        new_one = PartialCompletion(self.state, score_one,
                                    one_hot_one, one_hots_one)
        new_two = PartialCompletion(self.state, score_two,
                                    one_hot_two, one_hots_two)
        return new_one , new_two

def get_statelist_from_batch(state_batch):
    list_of_states = []
    a, b = state_batch
    nl, bs, ls = a.shape
    for i in range(bs):
        list_of_states.append((a[:, i, :], b[:, i, :]))
    return list_of_states

def beamsearch(list_of_particles, max_population, num_of_unique_characters):
    succession_particles = []
    for i in list_of_particles:
        a, b = i.successors(num_of_unique_characters)
        succession_particles.append(a)
        succession_particles.append(b)
    succession_scores = [particle.score for particle in succession_particles]
    succession_probs = softmax(succession_scores)
    selection_samples = np.random.choice(len(succession_particles),
                                         max_population, replace= True,
                                         p= succession_probs)
    selection_particles = []
    for i in selection_samples:
        selection_particles.append(succession_particles[i])
    return selection_particles

def softmax(x):
    temperature = 5
    x=[i/temperature for i in x]
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def get_batchstate_from_list(list_of_particles):
    a, b = list_of_particles[0].state
    a = torch.unsqueeze(a, 1)
    b = torch.unsqueeze(b, 1)
    for i in range(1, len(list_of_particles)):
        c, d = list_of_particles[i].state
        c = torch.unsqueeze(c, 1)
        d = torch.unsqueeze(d, 1)
        a = torch.cat((a, c), 1)
        b = torch.cat((b, d), 1)
    return (a, b)

def display(x, context, list_of_particles, list_of_characters):
    context_index = torch.squeeze(torch.argmax(context, dim=2))
    actual_cutoff = ''
    for j in context_index:
        actual_cutoff = actual_cutoff + list_of_characters[j]
    print('||', actual_cutoff, '||')
    actual_text_indices = torch.squeeze(torch.argmax(x, dim=2))
    for particle in list_of_particles:
        indices = torch.argmax(particle.one_hots, dim=1)
        actual_text = ''
        predicted_version = ''
        for index, value in enumerate(indices):
            actual_text = (actual_text +
                           list_of_characters[
                                              actual_text_indices[index].item()
                                              ]
                          )
            predicted_version = (predicted_version +
                                 list_of_characters[value.item()])
        print('|||', actual_text, '|||', predicted_version)
    return

def main():
    experiment(False, 2, 100, 20)

if __name__ == "__main__":
    main()
