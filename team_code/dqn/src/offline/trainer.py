import pathlib
import argparse
from dqn.src.agents.map_model import MapModel 

def main(hparams):

    logger = False
    if hparams.log:
        logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='dqn_offline')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=1) 

    model = MapModel(hparams)

    # overwrite args
    if hparams.resume is not None:
        model = MapModel.load_from_checkpoint(RESUME)
        attributes = [a for a in dir(hparams) if not a.startswith('__')]
        attributes = {k:v for k,v in hparams.__dict__.items() if not k.startswith('__')}
        print(attributes)
        for k, v in attributes.items():
            model.hparams.__dict__[k] = v

    # resume and add a couple arguments
    #model.hparams.max_epochs = hparams.max_epochs
    #model.hparams.dataset_dir = hparams.dataset_dir
    #model.hparams.batch_size = hparams.batch_size
    #model.hparams.save_dir = hparams.save_dir
    #model.hparams.n = hparams.n
    #model.hparams.gamma = hparams.gamma
    #model.hparams.num_workers = hparams.num_workers
    #model.hparams.no_margin = hparams.no_margin
    #model.hparams.no_td = hparams.no_td
    #model.hparams.data_mode = 'offline'

    with open(hparams.save_dir / 'config.yml', 'w') as f:
        hparams_copy = copy.copy(vars(model.hparams))
        hparams_copy['dataset_dir'] = str(model.dataset_dir)
        hparams_copy['save_dir'] = str(model.hparams.save_dir)
        del hparams_copy['id']
        yaml.dump(hparams_copy, f, default_flow_style=False, sort_keys=False)

    # offline trainer can use all gpus
    # when resuming, the network starts at epoch 36
    trainer = pl.Trainer(
        gpus=hparams.gpus, max_epochs=hparams.max_epochs,
        resume_from_checkpoint=RESUME,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        distributed_backend='dp',)

    trainer.fit(model)

    if hparams.log:
        wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--debug', action='store_true')

    # Trainer args
    parser.add_argument('-G', '--gpus', type=int, default=-1)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--save_dir', type=pathlib.Path)
    parser.add_argument('--data_root', type=pathlib.Path, default='/data')
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) 
    #parser.add_argument('--offline', action='store_true', default=False)

    # Model args
    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False) # what is this again?
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--sample_by', type=str, 
            choices=['none', 'even', 'speed', 'steer'], default='none')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--no_margin', action='store_true')
    parser.add_argument('--no_td', action='store_true')

    # Program args
    parser.add_argument('--dataset_dir', type=pathlib.Path)
    parser.add_argument('--log', action='store_true')

    
    args = parser.parse_args()
    args.data_mode = 'offline'
    assert not (args.no_margin and args.no_td), 'no loss provided for training'

    if args.dataset_dir is None: # local
        #args.dataset_dir = '/data/leaderboard/data/rl/dspred/debug/20210311_143718'
        args.dataset_dir = '/data/leaderboard/data/rl/dspred/20210311_213726'

    suffix = f'debug/{args.id}' if args.debug else args.id
    save_root = args.data_root / f'leaderboard/training/rl/dspred/{suffix}'

    args.save_dir = save_root
    args.save_dir.mkdir(parents=True, exist_ok=True)

    main(args)
