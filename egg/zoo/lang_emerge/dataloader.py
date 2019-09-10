import torch
import itertools, pdb, json, random

def load_from_file(path):
    with open(path, 'r') as f: 
        loaded = json.loads(f.read())
    return loaded


class Dataset:
    def __init__(self, path, a_vocab_size=4, q_vocab_size=3, mode='train'):
        assert mode in ['train', 'test']
        dataset = load_from_file(path)
        self.n_examples = dataset['numInst'][mode]
        self.n_tasks = len(dataset['taskDefn'])

        self.n_values = {attr: len(vals) for attr, vals in dataset['props'].items()}
        self.attr_val_vocab = sum([len(v) for v in dataset['props'].values()])

        # input vocab for answerer
        # inVocab and outVocab same for questioner
        task_vocab = ['<T%d>' % ii for ii in range(self.n_tasks)]
        # A, Q have different vocabs
        q_out_vocab = [chr(ii + 97) for ii in range(q_vocab_size)]
        a_out_vocab = [chr(ii + 65) for ii in range(a_vocab_size)]

        a_in_vocab =  q_out_vocab + a_out_vocab
        q_in_vocab = a_out_vocab + q_out_vocab + task_vocab

        self.n_attrs = len(dataset['attributes'])
        self.task_select = torch.LongTensor(dataset['taskDefn'])

        # number of single and pair wise tasks
        self.n_pair_tasks = 6
        self.n_single_tasks = 3

        # create a vocab map for field values
        attr_vals = []
        for attr in dataset['attributes']:
            attr_vals.extend(dataset['props'][attr])

        self.attr_vocab = {value:ii for ii, value in enumerate(attr_vals)}
        self.inv_attr_vocab = {index:attr for attr, index in self.attr_vocab.items()}

        # get encoding for attribute pairs
        self.attr_pair = itertools.product(attr_vals, repeat=2)
        self.attr_pair_vocab = {value:ii for ii, value in enumerate(self.attr_pair)}
        self.inv_attr_pair_vocab = {index:value for value, index in self.attr_pair_vocab.items()}

        # Separate data loading for test/train
        self.data = torch.LongTensor(self.n_examples, self.n_attrs)
        for ii, attr_set in enumerate(dataset['split'][mode]):
            self.data[ii] = torch.LongTensor([self.attr_vocab[at] for at in attr_set])

        # TODO: should it be until n_examples?
        self.range_inds = torch.range(0, self.n_examples - 1).long()

    def __len__(self): 
        return self.n_examples

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    dataset = Dataset('./data/toy64_split_0.8.json', mode='test')
    train_loader = torch.utils.data.DataLoader(dataset,
                              batch_size=32,
                              shuffle=True, num_workers=1)


    for b in train_loader:
        print(b)