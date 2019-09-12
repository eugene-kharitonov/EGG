import itertools
import json
import torch


def load_from_file(path):
    with open(path, 'r') as f:
        loaded = json.loads(f.read())
    return loaded


class Dataset:
    def __init__(self, path, a_vocab_size=4, q_vocab_size=3, mode='train', inflate=10):
        self.inflate = inflate

        assert mode in ['train', 'test']
        dataset = load_from_file(path)
        self.n_examples = dataset['numInst'][mode]

        self.tasks = dataset['taskDefn']
        self.n_tasks = len(self.tasks)

        self.n_values = {attr: len(vals)
                         for attr, vals in dataset['props'].items()}
        self.attr_val_vocab = sum([len(v) for v in dataset['props'].values()])

        self.n_attrs = len(dataset['attributes'])
        self.n_uniq_attrs = len(dataset['props'])

        self.task_select = torch.LongTensor(dataset['taskDefn'])

        # number of single and pair wise tasks
        self.n_pair_tasks = 6
        self.n_single_tasks = 3

        # create a vocab map for field values
        attr_vals = []
        for attr in dataset['attributes']:
            attr_vals.extend(dataset['props'][attr])

        self.attr_vocab = {value: ii for ii, value in enumerate(attr_vals)}
        self.inv_attr_vocab = {index: attr for attr,
                               index in self.attr_vocab.items()}

        # get encoding for attribute pairs
        self.attr_pair = itertools.product(attr_vals, repeat=2)
        self.attr_pair_vocab = {value: ii for ii,
                                value in enumerate(self.attr_pair)}
        self.inv_attr_pair_vocab = {
            index: value for value, index in self.attr_pair_vocab.items()}

        # Separate data loading for test/train
        self.data = torch.LongTensor(self.n_examples, self.n_attrs)
        for ii, attr_set in enumerate(dataset['split'][mode]):
            self.data[ii] = torch.LongTensor(
                [self.attr_vocab[at] for at in attr_set])


    def __len__(self):
        return self.n_examples * self.n_pair_tasks * self.inflate

    def __getitem__(self, idx):

        task = idx % self.n_pair_tasks
        index = self.tasks[task]
        index = torch.LongTensor(index)


        task = torch.LongTensor([task])
        batch = self.data[idx % self.n_examples]

        label = batch.gather(0, index)
        return batch, task, label


if __name__ == '__main__':
    dataset = Dataset('./data/toy64_split_0.8.json', mode='test')
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=1,
                                               shuffle=True, num_workers=1)

    for b in train_loader:
        print(b)
