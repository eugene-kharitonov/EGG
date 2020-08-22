from collections import defaultdict
import os

def grid():
    folders = ['/private/home/rdessi/checkpoint_local/overfit/values_30/best_gru/', '/private/home/rdessi/checkpoint_local/overfit/values_30/best_lstm/', '/private/home/rdessi/checkpoint_local/overfit/values_30/best_rnn/']


    models_with_seed = {}

    for folder in folders:
        models_with_seed[folder] = defaultdict(list)
        for (dirpath, dirnames, filenames) in os.walk(folder):
            for filename in filenames:
                filename = os.path.join(folder, filename)
                if filename.endswith('err') and os.stat(filename).st_size != 0:
                    continue
                elif filename.endswith('out'):
                    if os.stat(filename).st_size == 0:
                        continue
                    else:
                        with open(filename, 'r') as f:
                            content = f.readlines()
                            params = content[0].split('Namespace(')[1].strip()[:-1]
                            params = params.split(', ')
                            d = {}
                            tr = [tuple(elem.split('=')) for elem in params]
                            for tup in tr:
                                if len(tup) == 3:
                                    continue
                                par, value = tup
                                d[par] = value

                            if 'TOPSIM' in content[-1]:
                                k = f'bs{d["batch_size"]}_lr{d["lr"]}_vocab{d["vocab_size"]}_rec_h{d["receiver_hidden"]}_sen_h_{d["sender_hidden"]}_rec_emb{d["receiver_embedding"]}_send_emb{d["sender_embedding"]}_random_seed{d["random_seed"]}'

                                topsim = round(float(content[-1].split('full: ')[1].split(',')[0]), 4)

                                if 'STOPPING' in content[-2]:
                                    model_acc = round(float(content[-3].split('hard_acc": ')[1].split(',')[0]) * 100, 2)
                                else:
                                    model_acc = round(float(content[-2].split('hard_acc": ')[1].split(',')[0]) * 100, 2)

                                param_list = content[0].strip()[10:][:-1].split(', ')
                                params = []
                                for elem in param_list:
                                    if 'lr' in elem or ('batch_size' in elem and 'val_batch_size' not in elem) or 'vocab_size' in elem or 'receiver_hidden' in elem or 'sender_hidden' in elem:
                                        params.append(elem)
                                    if 'sender_emb' in elem or 'receiver_emb' in elem or 'sender_cell' in elem or 'receiver_cell' in elem:
                                        params.append(elem)
                                model_name_wo_seed = k.split('_random_seed')[0]
                                models_with_seed[folder][model_name_wo_seed].append((model_acc, topsim, params))
            break


    final = []
    for model, arch_dict in models_with_seed.items():
        for (k, v) in arch_dict.items():
            avg = round(sum([el[0] for el in v]) / len(v), 2)
            avg_topsim = round(sum([el[1] for el in v]) / len(v), 2)
            if not avg or not avg_topsim:
                exit(1)
            params = v[-1][-1]
            final.append((avg, params))

    final.sort(key=lambda x: x[0], reverse=True)
    for i, (avg, params) in enumerate(final[:30]):
        #print(f'{i+1}. acc: {avg}, {params}')
        for mtask in [0, 1]:
            s = []
            for elem in params:
                if 'gru' in elem or 'rnn' in elem or 'lstm' in elem:
                    elem = elem.replace('"', '')
                    elem = elem.replace("'", '')
                s.append(f'--{elem}')
            s.extend(['--n_values=30', '--max_len=5', '--n_attributes=3', '--early_stopping_thr=0.9999', '--n_epoch=300', '--evaluate_train_set_limit=10000', f'--mtask={mtask}'])
            for multi in range(1, 11):
                seed = multi * 111
                out = s + [f'--random_seed={seed}']
                yield out
                #print(out)
        break

# '--n_epoch=1', '--evaluate_train_set_limit=100', 'n_validation_samples=300', '--val_batch_size=100', '--samples_per_epoch=600', '--evaluate_train_set_limit=10'
#if __name__ == '__main__':
#    grid()
