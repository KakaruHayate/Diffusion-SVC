import os
import argparse
import torch
from torch.optim import lr_scheduler
from logger import utils
from diffusion.data_loaders import get_data_loaders
from diffusion.solver import train
from diffusion.unit2mel import Unit2Mel, Unit2MelNaive, load_svc_model
from diffusion.vocoder import Vocoder
from diffusion.discriminator import Discriminator


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    parser.add_argument(
        "-p",
        "--print",
        type=str,
        required=False,
        default=None,
        help="print model")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    model = load_svc_model(args=args, vocoder_dimension=vocoder.dimension)
    
    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
    
    
    # device
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    if args.model.naive_fn.type is None:
        Discriminator=None
        D_optimizer=None
        D_scheduler=None
    else:
        if args.model.naive_with_discriminator:
            Discriminator = Discriminator(n_mel=128).to(args.device)
            D_optimizer = torch.optim.AdamW(Discriminator.parameters())
            initial_global_step, Discriminator, D_optimizer = utils.load_model(args.env.expdir, Discriminator, D_optimizer, name='D', device=args.device)
            for param_group in D_optimizer.param_groups:
                param_group['initial_lr'] = args.train.d_lr
                param_group['lr'] = args.train.d_lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
                param_group['weight_decay'] = args.train.weight_decay
            D_scheduler = lr_scheduler.StepLR(D_optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2)
            Discriminator.to(args.device)
        else:
            Discriminator=None
            D_optimizer=None
            D_scheduler=None
    # print(Discriminator)
    # 打印模型结构
    if (str(cmd.print) == 'True') or (str(cmd.print) == 'true'):
        print(model)
        if args.model.naive_with_discriminator:
            print(Discriminator)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    # run
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, Discriminator, D_optimizer, D_scheduler)
    
