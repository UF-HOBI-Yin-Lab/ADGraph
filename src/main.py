import argparse
import time
import os
import os.path as osp
import wandb
import pandas as pd
import sys 
import torch
import torch.nn as nn
sys.path.insert(0, '/home/weimin.meng/projects/AD_progression')
os.chdir('/home/weimin.meng/projects/AD_progression')
from src.utils.config import Config
from src.rec.wandb_setup import initialise_wandb
from src.rec.logger import get_root_logger
from src.model.run import Runner
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a graph neural network')
    parser.add_argument('--config', type=str, default='src/utils/config.yml',
                        help='config file path')
    parser.add_argument('--rec', type=str, default="nan",
                        help='method to record. Example: --rec "wandb, logger, nan"')
    parser.add_argument('--log_path', type=str, default=None,
                        help='log path')
    parser.add_argument('--retrain', type=bool, default=None,
                        help='retrain model from a given path')
    parser.add_argument('--task', type=str, default='graph_classification',
                        help='task name. Example: --task "binary_classification, multi_classification, regression, graph_classification, multignn"')
    parser.add_argument('--model', type=str, default='sage',
                        help='model name. Example: --model "lr, rf, xgb, mlp, bilstm, resnet, gcn, gat, sage"')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training.')
    parser.add_argument('--dataset', type=str, default='mci_ad',
                        help='dataset name. Example: --dataset "mci_ad, mci_ad_mst, mci_ad_sep, mci_ad_mst_sep, mci_ad_sep_mst, ad_death, ad_death_mst, ad_death_sep, ad_death_mst_sep, ad_death_sep_mst"')
    parser.add_argument('--graph', type=bool, default=True,
                        help='whether to use graph neural network')
    parser.add_argument('--graph_data', type=bool, default=False,
                        help='whether to use processed graph data')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='whether to shuffle the data')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer name. Example: --optimizer "adam, sgd"')
    parser.add_argument('--momentum', type=float, default=None,
                        help='SGD momentum')
    parser.add_argument('--save_epoch', type=int, default=1,
                        help='save model every n epochs')
    parser.add_argument('--dl', type=bool, default=True,
                        help='whether to use deep learning')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--learning_rate', type=float, default=1.0e-3,
                        help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        help='learning rate scheduler')
    parser.add_argument('--lr_decay_steps', type=int, default=None,
                        help='learning rate decay steps')
    parser.add_argument('--lr_decay_rate', type=float, default=None,
                        help='learning rate decay rate')
    parser.add_argument('--lr_decay_min_lr', type=float, default=None,
                        help='learning rate decay min lr')
    parser.add_argument('--lr_patience', type=int, default=None,
                        help='learning rate patience')
    parser.add_argument('--lr_cooldown', type=int, default=None,
                        help='learning rate cooldown')
    parser.add_argument('--lr_threshold', type=float, default=None,
                        help='learning rate threshold')
    parser.add_argument('--callbacks', type=list, default=None,
                        help='callbacks. Example: --callbacks "early_stopping, model_checkpoint"')
    parser.add_argument('--training', type=bool, default=True,
                        help='whether to train the model')
    parser.add_argument('--cuda', type=int, default=1,
                        help='cuda number')
    parser.add_argument('--dataloader', type=bool, default=False,
                        help='whether to use dataloader')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--pooling', type=str, default='mean',
                        help='pooling method')

    args = parser.parse_args()

    return args

def epoch_results(results, epoch):
    '''
    parameters:
    results: dict, the results of each epoch (list)
    epoch: int, the current epoch

    return:
    epoch_res: dict, the results of the current epoch
    '''
    epoch_res = {}
    for key in results.keys():
        epoch_res[key] = results[key][epoch]
    return epoch_res

def run(cfg, logger):
    # initialize the runner
    runner = Runner(cfg)
    runner_variables_dict = runner.get_self_variables_as_dict()
    if cfg['run_config']['rec'] == 'wandb':
        try:
            wandb.log({"runner variables": runner_variables_dict})
        except:
            wandb.finish()
            sys.exit("wandb failed to log runner variables")
    logger.info(f'runner variables:\n{runner_variables_dict}')
    if cfg['run_config']['rec'] == 'wandb':
        try:
            wandb.log({"train dataset len": torch.sum(runner.dataset.data['train_mask']).item(), "val dataset len": torch.sum(runner.dataset.data['val_mask']).item(), "test dataset len": torch.sum(runner.dataset.data['test_mask']).item()})
        except:
            wandb.finish()
            sys.exit("wandb failed to log dataset length")
    if cfg['run_config']['task'] == 'graph_classification' or cfg['run_config']['task'] == 'multignn':
        logger.info(f'train dataset len: {torch.sum(runner.dataset.data_list[0]["train_mask"]).item()}, val dataset len: {torch.sum(runner.dataset.data_list[0]["val_mask"]).item()}, test dataset len: {torch.sum(runner.dataset.data_list[0]["test_mask"]).item()}')
    else:
        logger.info(f'train dataset len: {torch.sum(runner.dataset.data["train_mask"]).item()}, val dataset len: {torch.sum(runner.dataset.data["val_mask"]).item()}, test dataset len: {torch.sum(runner.dataset.data["test_mask"]).item()}')
    print('Start training...')
    logger.info('Start training...')
    
    duration = []
    if cfg['run_config']['task'] == 'binary_classification' or cfg['run_config']['task'] == 'multi_classification' or cfg['run_config']['task'] == 'graph_classification' or cfg['run_config']['task'] == 'multignn':
        results = {'train_loss':[], 'train_accuracy':[], 'train_precision':[], 'train_recall':[], 'train_f1':[], 'train_auroc':[], 
               'val_loss':[], 'val_accuracy':[], 'val_precision':[], 'val_recall':[], 'val_f1':[], 'val_auroc':[], 
               'test_loss':[], 'test_accuracy':[], 'test_precision':[], 'test_recall':[], 'test_f1':[], 'test_auroc':[]}
    elif cfg['run_config']['task'] == 'regression':
        results = {'train_loss':[], 'train_mse':[], 'train_mae':[], 'train_r2':[], 
               'val_loss':[], 'val_mse':[], 'val_mae':[], 'val_r2':[], 
               'test_loss':[], 'test_mse':[], 'test_mae':[], 'test_r2':[]}
    else:
        raise ValueError(f'Invalid task: {cfg["run_config"]["task"]}')
    # get into the epoch
    for epoch in range(cfg['run_config']['epochs']):
        # if epoch == 95:
        #     print('stop')
        # record the time
        start_time = time.time()
        # train the model
        runner.train()
        if cfg['run_config']['task'] == 'binary_classification' or cfg['run_config']['task'] == 'multi_classification' or cfg['run_config']['task'] == 'graph_classification' or cfg['run_config']['task'] == 'multignn':
            results['train_loss'].append(runner.loss)
            results['train_accuracy'].append(runner.accuracy)
            results['train_precision'].append(runner.precision)
            results['train_recall'].append(runner.recall)
            results['train_f1'].append(runner.f1)
            results['train_auroc'].append(runner.auroc)
        elif cfg['run_config']['task'] == 'regression':
            results['train_loss'].append(runner.loss)
            results['train_mse'].append(runner.mse)
            results['train_mae'].append(runner.mae)
            results['train_r2'].append(runner.r2)
        else:
            raise ValueError(f'Invalid task: {cfg["run_config"]["task"]}')
        # validate the model
        runner.eval()
        if cfg['run_config']['task'] == 'binary_classification' or cfg['run_config']['task'] == 'multi_classification' or cfg['run_config']['task'] == 'graph_classification' or cfg['run_config']['task'] == 'multignn':
            results['val_loss'].append(runner.loss)
            results['val_accuracy'].append(runner.accuracy)
            results['val_precision'].append(runner.precision)
            results['val_recall'].append(runner.recall)
            results['val_f1'].append(runner.f1)
            results['val_auroc'].append(runner.auroc)
        elif cfg['run_config']['task'] == 'regression':
            results['val_loss'].append(runner.loss)
            results['val_mse'].append(runner.mse)
            results['val_mae'].append(runner.mae)
            results['val_r2'].append(runner.r2)
        else:
            raise ValueError(f'Invalid task: {cfg["run_config"]["task"]}')
        # test the model
        runner.test()
        if cfg['run_config']['task'] == 'binary_classification' or cfg['run_config']['task'] == 'multi_classification' or cfg['run_config']['task'] == 'graph_classification' or cfg['run_config']['task'] == 'multignn':
            results['test_loss'].append(runner.loss)
            results['test_accuracy'].append(runner.accuracy)
            results['test_precision'].append(runner.precision)
            results['test_recall'].append(runner.recall)
            results['test_f1'].append(runner.f1)
            results['test_auroc'].append(runner.auroc)
        elif cfg['run_config']['task'] == 'regression':
            results['test_loss'].append(runner.loss)
            results['test_mse'].append(runner.mse)
            results['test_mae'].append(runner.mae)
            results['test_r2'].append(runner.r2)
        else:
            raise ValueError(f'Invalid task: {cfg["run_config"]["task"]}')
        # record the time
        end_time = time.time()
        duration.append(end_time - start_time)
        # print the results
        epoch_res = epoch_results(results, epoch)
        epoch_res['duration'] = duration[-1]
        if cfg['run_config']['rec'] == 'wandb':
            try:
                wandb.log(epoch_res)
            except:
                wandb.finish()
                sys.exit("wandb failed to log epoch results")
        logger.info(f'Epoch {epoch} results:\n{epoch_res}')
        # print(f'Epoch {epoch} results:\n{epoch_res}')
        # save model
        if cfg['run_config']['save_epoch'] and (epoch + 1) % cfg['run_config']['save_epoch'] == 0:
            runner.save_model(cfg['run_config']['model_path'])
    if cfg['run_config']['rec'] == 'wandb':
        wandb.finish()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # save the results to a csv with the timestamp
    results_df = pd.DataFrame(results)
    results_df.to_csv(osp.join(cfg['run_config']['res_path'], f'{cfg["run_config"]["model"]}_{cfg["run_config"]["task"]}_{cfg["run_config"]["dataset"]}_{cfg["run_config"]["learning_rate"]}_{cfg["run_config"]["batch_size"]}_{cfg["run_config"]["epochs"]}_{cfg["run_config"]["optimizer"]}_{timestamp}.csv'))


def main():
    args = parse_args()
    cfg = Config(args.config)

    if args.rec is not None: # can be None
        if args.rec == 'nan':
            cfg.cfg['run_config']['rec'] = None
        else:
            cfg.cfg['run_config']['rec'] = args.rec
    if args.log_path is not None:
        cfg.cfg['run_config']['log_path'] = args.log_path
    if args.retrain is not None:
        cfg.cfg['run_config']['retrain'] = args.retrain
    if args.task is not None:
        cfg.cfg['run_config']['task'] = args.task
    if args.model is not None:
        cfg.cfg['run_config']['model'] = args.model
    if args.batch_size is not None:
        cfg.cfg['run_config']['batch_size'] = args.batch_size
    if args.dataset is not None:
        cfg.cfg['run_config']['dataset'] = args.dataset
    if args.graph is not None:
        cfg.cfg['run_config']['graph'] = args.graph
    if args.graph_data is not None:
        cfg.cfg['run_config']['graph_data'] = args.graph_data
    if args.shuffle is not None:
        cfg.cfg['run_config']['shuffle'] = args.shuffle
    if args.num_workers is not None:
        if args.num_workers == -1:
            cfg.cfg['run_config']['num_workers'] = None
        else:
            cfg.cfg['run_config']['num_workers'] = args.num_workers
    if args.epochs is not None:
        cfg.cfg['run_config']['epochs'] = args.epochs
    if args.optimizer is not None:
        cfg.cfg['run_config']['optimizer'] = args.optimizer
    if args.momentum is not None:
        if args.momentum == -1:
            cfg.cfg['run_config']['momentum'] = None
        else:
            cfg.cfg['run_config']['momentum'] = args.momentum
    if args.save_epoch is not None:
        if args.save_epoch == -1:
            cfg.cfg['run_config']['save_epoch'] = None
        else:
            cfg.cfg['run_config']['save_epoch'] = args.save_epoch
    if args.dl is not None:
        cfg.cfg['run_config']['dl'] = args.dl
    if args.weight_decay is not None:
        if args.weight_decay == -1:
            cfg.cfg['run_config']['weight_decay'] = None
        else:
            cfg.cfg['run_config']['weight_decay'] = args.weight_decay
    if args.learning_rate is not None:
        cfg.cfg['run_config']['learning_rate'] = args.learning_rate
    if args.lr_scheduler is not None:
        if args.lr_scheduler == 'nan':
            cfg.cfg['run_config']['lr_scheduler'] = None
        else:
            cfg.cfg['run_config']['lr_scheduler'] = args.lr_scheduler
    if args.lr_decay_steps is not None:
        if args.lr_decay_steps == -1:
            cfg.cfg['run_config']['lr_decay_steps'] = None
        else:
            cfg.cfg['run_config']['lr_decay_steps'] = args.lr_decay_steps
    if args.lr_decay_rate is not None:
        if args.lr_decay_rate == -1:
            cfg.cfg['run_config']['lr_decay_rate'] = None
        else:
            cfg.cfg['run_config']['lr_decay_rate'] = args.lr_decay_rate
    if args.lr_decay_min_lr is not None:
        if args.lr_decay_min_lr == -1:
            cfg.cfg['run_config']['lr_decay_min_lr'] = None
        else:
            cfg.cfg['run_config']['lr_decay_min_lr'] = args.lr_decay_min_lr
    if args.lr_patience is not None:
        if args.lr_patience == -1:
            cfg.cfg['run_config']['lr_patience'] = None
        else:
            cfg.cfg['run_config']['lr_patience'] = args.lr_patience
    if args.lr_cooldown is not None:
        if args.lr_cooldown == -1:
            cfg.cfg['run_config']['lr_cooldown'] = None
        else:
            cfg.cfg['run_config']['lr_cooldown'] = args.lr_cooldown
    if args.lr_threshold is not None:
        if args.lr_threshold == -1:
            cfg.cfg['run_config']['lr_threshold'] = None
        else:
            cfg.cfg['run_config']['lr_threshold'] = args.lr_threshold
    if args.callbacks is not None:
        if args.callbacks == 'nan':
            cfg.cfg['run_config']['callbacks'] = None
        else:
            cfg.cfg['run_config']['callbacks'] = args.callbacks
    if args.training is not None:
        cfg.cfg['run_config']['training'] = args.training
    if args.cuda is not None:
        if args.cuda == -1:
            cfg.cfg['run_config']['cuda'] = None
        else:
            cfg.cfg['run_config']['cuda'] = args.cuda
    if args.dataloader is not None:
        cfg.cfg['run_config']['dataloader'] = args.dataloader
    if args.num_classes is not None:
        cfg.cfg['run_config']['num_classes'] = args.num_classes
    if args.pooling is not None:
        cfg.cfg['run_config']['pooling'] = args.pooling

    # init wandb
    if cfg.cfg['run_config']['rec'] == 'wandb':
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        cfg.cfg['wandb_config']['wandb_run_name'] = (
            cfg.cfg['run_config']['model'] + '_' +
            cfg.cfg['run_config']['dataset'] + '_' +
            str(cfg.cfg['run_config']['learning_rate']) + '_' +
            str(cfg.cfg['run_config']['batch_size']) + '_' +
            str(cfg.cfg['run_config']['epochs']) + '_' +
            cfg.cfg['run_config']['optimizer'] + '_' +
            str(timestamp)
        )
        wandb_cfg = initialise_wandb(cfg.cfg['wandb_config'])
    else:
        os.environ["WANDB_MODE"] = "disabled"
    # init the logger before other steps
    log_file = osp.join(
        cfg.cfg['run_config']['log_path'],
        '{}_{}_{}_{}_{}_{}_{}_{}.log'.format(cfg.cfg['run_config']['model'], cfg.cfg['run_config']['dataset'], cfg.cfg['run_config']['learning_rate'], cfg.cfg['run_config']['weight_decay'],
                                                      cfg.cfg['run_config']['batch_size'], cfg.cfg['run_config']['epochs'], cfg.cfg['run_config']['optimizer'], cfg.cfg['run_config']['lr_scheduler']))
    # logger = get_root_logger(log_file=log_file, log_level=cfg['log_level'])
    logger = get_root_logger(cfg.cfg['run_config']['model'], log_file=log_file)

    # logger basic setting
    if cfg.cfg['run_config']['rec'] == 'wandb':
        wandb.log({"Config": cfg.cfg})
    logger.info(f'Config:\n{cfg.cfg}')

    # run
    run(cfg.cfg, logger)
    

if __name__ == '__main__':
    main()