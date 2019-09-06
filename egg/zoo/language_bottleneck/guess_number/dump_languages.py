# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import argparse
import pathlib
import torch.utils.data
import egg.core as core


def dump(game, dataset, device, is_gs, is_var_length, output_file):
    sender_inputs, messages, _1, receiver_outputs, _2 = \
        core.dump_sender_receiver(
            game, dataset, gs=is_gs, device=device, variable_length=is_var_length)

    for sender_input, message, receiver_output \
            in zip(sender_inputs, messages, receiver_outputs):
        sender_input = ''.join(map(str, sender_input.tolist()))
        if is_var_length:
            message = ' '.join(map(str, message.tolist()))
        receiver_output = (receiver_output > 0.5).tolist()
        receiver_output = ''.join([str(x) for x in receiver_output])
        f.write(f'{sender_input} -> {message} -> {receiver_output}\n')


def was_success(path):
    with open(path, 'r') as f:
        for line in f:
            try:
                l = json.loads(line)
                if l['mode'] == 'test' and l['acc'] > 0.99:
                    return True
            except:
                pass
    return False

def build_models(opts, job_id):
    from egg.zoo.language_bottleneck.guess_number.archs import Sender, Receiver, ReinforcedReceiver, FactorizedSender, Discriminator
    from egg.zoo.language_bottleneck.guess_number.features import UniformLoader, OneHotLoader

    device = 'cuda'
    train_loader = None
    test_loader = UniformLoader(n_bits=opts.n_bits, bits_s=opts.bits_s, bits_r=opts.bits_r)

    test_loader.batch = [x.to(device) for x in test_loader.batch]
    assert opts.variable_length and opts.mode == 'rf'

    """if opts.sender_cell == 'transformer':
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
                        vocab_size=opts.sender_hidden)  # TODO: not really vocab
        sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_emb, max_len=opts.max_len,
                                                    num_layers=1, num_heads=1, hidden_size=opts.sender_hidden)
    else:
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
                        vocab_size=opts.sender_hidden)  # TODO: not really vocab
        sender = core.RnnSenderReinforce(agent=sender, vocab_size=opts.vocab_size, 
                                    embed_dim=opts.sender_emb, hidden_size=opts.sender_hidden, max_len=opts.max_len, force_eos=True, cell=opts.sender_cell)
    """

    discriminator = Discriminator(opts.vocab_size, n_hidden=64, embed_dim=64)

    if opts.sender_cell == 'transformer':
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
                        vocab_size=opts.sender_hidden)  # TODO: not really vocab
        sender = core.TransformerSenderReinforce(agent=sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_emb, max_len=opts.max_len,
                                                 num_layers=1, num_heads=1, hidden_size=opts.sender_hidden)
    elif opts.sender_cell in ['lstm', 'rnn', 'gru']:
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        sender = Sender(n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
                        vocab_size=opts.sender_hidden)  # TODO: not really vocab
        sender = core.RnnSenderReinforce(agent=sender, vocab_size=opts.vocab_size, 
                                  embed_dim=opts.sender_emb, hidden_size=opts.sender_hidden, max_len=opts.max_len, force_eos=True, cell=opts.sender_cell)
    elif opts.sender_cell == 'factorized':
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        sender = FactorizedSender(max_len=opts.max_len, n_bits=opts.n_bits, n_hidden=opts.sender_hidden,
                        vocab_size=opts.vocab_size)



    if opts.receiver_cell == 'transformer':
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_emb)
        receiver = core.TransformerReceiverDeterministic(receiver, opts.vocab_size, opts.max_len, opts.receiver_emb, num_heads=1, hidden_size=opts.receiver_hidden,
                                                            num_layers=1)
    else:
        receiver = Receiver(n_bits=opts.n_bits, n_hidden=opts.receiver_hidden)
        receiver = core.RnnReceiverDeterministic(
            receiver, opts.vocab_size, opts.receiver_emb, opts.receiver_hidden, cell=opts.receiver_cell)

    game = core.SenderReceiverRnnReinforce(
            sender, receiver, None, sender_entropy_coeff=opts.sender_entropy_coeff, receiver_entropy_coeff=opts.receiver_entropy_coeff, \
                    discriminator=discriminator)

    optimizer = torch.optim.Adam(
        [
            dict(params=sender.parameters(), lr=opts.sender_lr),
            dict(params=receiver.parameters(), lr=opts.receiver_lr)
        ])

    # TODO: super ugly, close eyes
    core.init(params="")
    # TODO: super ugly, open eyes

    trainer = core.Trainer(
        game=game, 
        optimizer=optimizer,
        train_data=None,
        validation_data=None)

    checkpoint_path = pathlib.Path(opts.checkpoint_dir + '/' + job_id)
    trainer.load_from_latest(checkpoint_path)

    return game, test_loader

 
def get_file_params(path):
    with open(path, 'r') as f:
        f.readline(); f.readline()
        h = f.readline()
        h = json.loads(h)

    return h

def dump(game, dataset, device, is_gs, is_var_length, output_file):
    sender_inputs, messages, _1, receiver_outputs, _2 = \
        core.dump_sender_receiver(
            game, dataset, gs=is_gs, device=device, variable_length=is_var_length)

    for sender_input, message, receiver_output \
            in zip(sender_inputs, messages, receiver_outputs):
        sender_input = ''.join(map(str, sender_input.tolist()))
        if is_var_length:
            message = ' '.join(map(str, message.tolist()))
        receiver_output = (receiver_output > 0.5).tolist()
        receiver_output = ''.join([str(x) for x in receiver_output])
        output_file.write(f'{sender_input} -> {message} -> {receiver_output}\n')


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    opts = get_params()

    root = pathlib.Path(opts.input_path).absolute()
    collected = {}
    for file in root.glob('*.out'):
        if not was_success(file):
            continue
        parameters = get_file_params(file)
        if parameters['bits_r'] != 0: continue
        parameters_as_ns = argparse.Namespace(**parameters)
        job_id = file.stem.split('_')[0] + '_0'

        game, test_loader = build_models(parameters_as_ns, job_id)
        output_file_name = pathlib.Path(opts.output_path) / job_id

        print(f'Dumping a lanugage from {job_id}')
        with open(output_file_name, 'w') as f:
            f.write(f'# {json.dumps(parameters)}\n')
            dump(game, test_loader, device='cuda', is_gs=False, is_var_length=True, output_file=f)
