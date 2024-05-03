import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from music21 import note, chord, converter, instrument, stream
import glob

class MIDI():
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.file_notes = []
        self.trainseq = []
        self.transfer_dic = dict()
        self.dic_n = 0

    def parser(self, folderName):
        for file in glob.glob(f"{folderName}/*.mid"):
            midi = converter.parse(file)
            print("Parsing %s" % file)
            notes = []
            for element in midi.flat.elements:
                if isinstance(element, note.Rest) and element.offset != 0:
                    notes.append('R')
                elif isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(pitch) for pitch in element.pitches))
            self.file_notes.append(notes)

        note_set = sorted(set(note for notes in self.file_notes for note in notes))
        self.dic_n = len(note_set)
        self.transfer_dic = dict((note, number) for number, note in enumerate(note_set))

        sequences = self.prepare_sequences()
        # Save processed sequences
        save_path = os.path.join('D:/EC523/project/deal', 'processed_seq' + '_processed.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(sequences, f)
        print(f"Saved processed data at {save_path}")

    def prepare_sequences(self):
        self.trainseq = []
        for notes in self.file_notes:
            for i in range(len(notes) - self.seq_length):
                self.trainseq.append([self.transfer_dic[note] for note in notes[i:i + self.seq_length]])
        self.trainseq = np.array(self.trainseq)
        self.trainseq = (self.trainseq - float(self.dic_n) / 2) / (float(self.dic_n) / 2)
        return self.trainseq

    def create_midi(self, prediction_output, filename):
        offset = 0
        midi_stream = stream.Stream()
        for pattern in prediction_output:
            if pattern == 'R':
                midi_stream.append(note.Rest())
            elif ('.' in pattern) or pattern.isdigit():
                notes = [note.Note(n) for n in pattern.split('.')]
                midi_stream.append(chord.Chord(notes))
            else:
                midi_stream.append(note.Note(pattern))
            offset += 0.5
        midi_stream.write('midi', fp=f'{filename}.mid')

class Discriminator(nn.Module):
    def __init__(self, seq_length):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=512, num_layers=1, batch_first=True)
        self.bidirectional_lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.bidirectional_lstm(x)
        x = x[:, -1]
        x = self.linear_layers(x)
        return x

class Generator(nn.Module):
    def __init__(self, seq_length, latent_dim):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=512, num_layers=1, batch_first=True)
        self.bidirectional_lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, seq_length),
            nn.Tanh()
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.bidirectional_lstm(x)
        x = x[:, -1]
        x = self.linear_layers(x)
        return x.view(-1, seq_length, 1)

class GAN:
    def __init__(self, midi_obj):
        self.midi = midi_obj
        self.seq_length = midi_obj.seq_length
        self.latent_dim = 1000
        self.discriminator = Discriminator(self.seq_length).to(device)
        self.generator = Generator(self.seq_length, self.latent_dim).to(device)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.loss = nn.BCELoss()

    def train(self, epochs, dataFolder, batch_size=128, sample_interval=50):
        self.midi.parser(dataFolder)
        sequences = self.midi.prepare_sequences()
        dataset = TensorDataset(torch.from_numpy(sequences))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        real_label = 1.0
        fake_label = 0.0

        for epoch in range(epochs):
            for i, data in enumerate(dataloader, 0):
                real_data = data[0].to(torch.float32).unsqueeze(-1).to(device)
                batch_size = real_data.size(0)

                real_labels = torch.full((batch_size, 1), real_label, dtype=torch.float32, device=device)
                fake_labels = torch.full((batch_size, 1), fake_label, dtype=torch.float32, device=device)

                self.discriminator.zero_grad()
                real_output = self.discriminator(real_data)
                real_loss = self.loss(real_output, real_labels)
                real_loss.backward()

                noise = torch.randn(batch_size, self.latent_dim, 1, device=device)
                fake_data = self.generator(noise)
                self.discriminator.zero_grad()
                fake_output = self.discriminator(fake_data.detach())
                fake_loss = self.loss(fake_output, fake_labels)
                fake_loss.backward()
                self.d_optimizer.step()

                self.generator.zero_grad()
                output = self.discriminator(fake_data)
                g_loss = self.loss(output, real_labels)
                g_loss.backward()
                self.g_optimizer.step()

                if i % sample_interval == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], D Loss: {(real_loss.item() + fake_loss.item()) * 0.5:.4f}, G Loss: {g_loss.item():.4f}, fake Loss: {fake_loss.item():.4f}, real Loss: {real_loss.item() :.4f}")

    def save(self):
        if not os.path.exists('D:/EC523/project/model'):
            os.makedirs('D:/EC523/project/model')
        torch.save(self.discriminator.state_dict(), 'D:/EC523/project/model/discriminator.pth')
        torch.save(self.generator.state_dict(), 'D:/EC523/project/model/generator.pth')

    def generate(self, num_samples=1):
        noise = torch.randn(num_samples, self.latent_dim, 1, device=device)
        generated_data = self.generator(noise)
        generated_data = generated_data.detach().cpu().numpy()
        generated_notes = [self.midi.transfer_dic.inverse[n] for n in np.argmax(generated_data, axis=1)]
        self.midi.create_midi(generated_notes, 'D:/EC523/project/Result')

    def plot_loss(self):
        plt.plot(self.disc_loss, label='Discriminator Loss', color='red')
        plt.plot(self.gen_loss, label='Generator Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss During Training')
        plt.legend()
        plt.savefig('D:/EC523/project/GAN_Loss_per_Epoch.png')
        plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_length = 100  # Adjust this based on your dataset specifics

midi_processor = MIDI(seq_length=seq_length)

gan_model = GAN(midi_processor)

epochs = 5  # Number of epochs
data_folder = 'D:/EC523/project/midis_piano/test'  # Path to your MIDI data
batch_size = 128  # Batch size
sample_interval = 1  # Interval for logging

gan_model.train(epochs, data_folder, batch_size, sample_interval)

gan_model.save()
gan_model.plot_loss()