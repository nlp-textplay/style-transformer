import torch
import time
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval

import argparse

parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--save_time', type=str, default=False)
parser.add_argument('--data_path', type=str, default='shakespeare')
parser.add_argument('--max_length', type=int, default=30)
parser.add_argument('--l1', type=float, default=1)
parser.add_argument('--l2', type=float, default=1)
parser.add_argument('--l3', type=float, default=1)
parser.add_argument('--total_its', type=int, default=30)
parser.add_argument('--ablate', type=str, default=False)


args = parser.parse_args()

class Config():
    data_path = './data/' + args.data_path + '/'
    log_dir = 'runs/exp'
    save_path = './save/' + args.save_time + '/'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    l1 = args.l1
    l2 = args.l2
    l3 = args.l3
    ablate = args.ablate
    total_its = args.total_its
    min_freq = 3
    max_length = args.max_length
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 250
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0


def main():
    config = Config()

    # train_iters, dev_iters, test_iters, vocab = load_dataset(config, train_pos='train.pos', train_neg='train.neg',
    #              dev_pos='dev.pos', dev_neg='dev.neg',
    #              test_pos='test.pos', test_neg='test.neg')
    
    train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    

    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    print(config.discriminator_method)
    
    train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    

if __name__ == '__main__':
    main()
