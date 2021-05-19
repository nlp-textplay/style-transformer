import torch
import time
from data import load_dataset
from models import StyleTransformer, Discriminator
from train import train, auto_eval

import argparse

parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--save_time', type=str, default=False)
parser.add_argument('--data_path', type=str, default='yelp')
parser.add_argument('--data_size', type=int, default=None)

args = parser.parse_args()

class Config():
    data_path = './data/' + args.data_path + '/'
    data_size = args.data_size
    log_dir = 'runs/exp'
    save_path = './save/' + args.save_time + '/'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 16
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
    F_pretrain_iter = 5 # 500
    log_steps = 5
    eval_steps = 5 # 25
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
    train_iters, dev_iters, test_iters, vocab = load_dataset(config)
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)
    print(config.discriminator_method)
    
    train(config, vocab, model_F, model_D, train_iters, dev_iters, test_iters)
    

if __name__ == '__main__':
    main()
