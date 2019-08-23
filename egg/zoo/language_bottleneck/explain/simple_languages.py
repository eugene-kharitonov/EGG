import argparse
import numpy as np
import hashlib


def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--language', default='md5')
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--vocab_size', type=int, default=10)
    parser.add_argument('--n_bits', type=int, default=8)

    args = parser.parse_args()

    return args



def get_examples(n_bits):
    batch_size = 2**(n_bits)
    numbers = np.array(range(batch_size))

    examples = np.zeros((batch_size, n_bits), dtype=np.int)

    for i in range(n_bits):
        examples[:, i] = np.bitwise_and(numbers, 2 ** i) > 0

    return examples


if __name__ == '__main__':
    args = get_params()
    examples = get_examples(args.n_bits)

    for i in range(examples.shape[0]):
        meaning = ''.join([str(x) for x in examples[i, :]]) 

        if args.language == 'md5':
            hex = hashlib.md5(meaning.encode('utf8')).hexdigest()
            number = int(hex, 16)

            translation = []
    
            while number and len(translation) < args.max_len - 1:
                translation.append(number % args.vocab_size)
                number = number // args.vocab_size
            translation = [str(x + 1) for x in translation]
        elif args.language.startswith('base-'):
            base = args.language[5:]
            base = int(base)
            assert base in [2, 8, 10, 16]

            number = int(meaning, 2)
            if base == 2:
                translation = '{0:b}'.format(number)
            #elif base == 16:
            #    translation = '{0:x}'.format(number)
            elif base == 8:
                translation = '{0:o}'.format(number)
            elif base == 10:
                translation = f'{number}'
            else:
                assert False

            l = len(translation)
            if l < args.max_len - 1:
                prefix = [0] * (args.max_len - len(translation) - 1)
                prefix.extend(translation)
                translation = prefix
            translation = [str(int(x) + 1) for x in translation]
        translation.append('0')

        translation = ' '.join(translation)


        print(f'{meaning} -> {translation} -> {meaning}')
