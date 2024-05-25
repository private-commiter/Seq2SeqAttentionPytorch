import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from ml.utils import show_plot, time_since
from ml.encoder import EncoderRNN
from ml.decoder_attentive import AttentiveDecoderRNN
from ml.decoder import DecoderRNN
from ml.dataloader import get_dataloader, Config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config(
        data_dir='../storage/data', batch_size=256,#32,
        max_length=10, device=device,
        sos_token=0, eos_token=1
    )
    hidden_size = 128

    input_lang, output_lang, train_dataloader = get_dataloader(config)

    # for row in train_dataloader:
    #     print(row)

    # quit()

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    # decoder = AttentiveDecoderRNN(hidden_size, output_lang.n_words).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, config, encoder, decoder, 80, 0.001, print_every=5, plot_every=5)


def train(
        train_dataloader, config, encoder, decoder, n_epochs, learning_rate,
        print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, config, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (
                time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            )

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)


def train_epoch(
        dataloader, config: Config, encoder, decoder, encoder_optimizer,
        decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(
            # encoder_outputs, encoder_hidden, config.sos_token, config.device, None, config.max_length)
            encoder_outputs, encoder_hidden, config.sos_token, config.device, target_tensor, config.max_length)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == '__main__':
    main()
