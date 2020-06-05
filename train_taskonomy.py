import warnings
warnings.simplefilter("error")
import enum
import os, csv, datetime

import argparse
import os
import shutil
import time
import platform

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets


from taskonomy_losses import *
from taskonomy_loader import TaskonomyLoader


# from apex.parallel import DistributedDataParallel as DDP
# from apex.fp16_utils import *

import copy
import numpy as np
import signal
import sys
import math
from collections import defaultdict
import scipy.stats

#from ptflops import get_model_complexity_info

import model_definitions as models

torch.cuda.empty_cache()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Taskonomy Training')
parser.add_argument('--data_dir', '-d', dest='data_dir',required=True,
                    help='path to training set')
parser.add_argument('-tm', '--train-models', dest="train_models", default=None, required=True,
                    help='Path to training models txt file')
parser.add_argument('-vm', '--val-models', dest="val_models", default=None, required=True,
                    help='Path to validation models txt file')

parser.add_argument('--arch', '-a', metavar='ARCH',required=True,
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (required)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--tasks', '-ts', default='sdnkt', dest='tasks',
                    help='which tasks to train on')
parser.add_argument('--model_dir', default='models', dest='model_dir',
                    help='where to save models')
parser.add_argument('--image-size', default=256, type=int,
                    help='size of image side (images are square)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int,
                    help='maximum number of epochs to run')
parser.add_argument('-mlr', '--minimum_learning_rate', default=3e-5, type=float,
                    metavar='LR', help='End trianing when learning rate falls below this value.')

parser.add_argument('-lr', '--learning-rate',dest='lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-ltw0', '--loss_tracking_window_initial', default=500000, type=int,
                    help='inital loss tracking window (default: 500000)')
parser.add_argument('-mltw', '--maximum_loss_tracking_window', default=2000000, type=int,
                    help='maximum loss tracking window (default: 2000000)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume','--restart', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--start-epoch', default=0, type=int,
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('-n','--experiment_name', default='', type=str,
                    help='name to prepend to experiment saves.')
parser.add_argument('-v', '--validate', dest='validate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='evaluate model on test set')

parser.add_argument('-r', '--no_rotate_loss', dest='no_rotate_loss', action='store_true',
                    help='should loss rotation occur')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('-vb', '--virtual-batch-multiplier', default=1, type=int,
                    metavar='N', help='number of forward/backward passes per parameter update')
# parser.add_argument('--fp16', action='store_true',
#                     help='Run model fp16 mode.')
parser.add_argument('-ml', '--model-limit', default=None, type=int,
                    help='Limit the number of training instances from a single 3d building model.')
parser.add_argument('-par', '--partition', dest='partition', action='store_true',
                    help='N (partition = false) vs C*N (partition = True) where C is number of tasks.')
parser.add_argument('--metrics_directory', dest='metrics_directory', default='metrics_directory', type=str,
                    help='Specify the metrics dirctory.')

cudnn.benchmark = False


class Device(enum.Enum):
    CPU = torch.device("cpu")
    CUDA = torch.device("cuda:0")
    PARALLEL_CUDA = torch.device("cuda:0")

default_device = None
if not torch.cuda.is_available():
    default_device = Device.CPU
elif torch.cuda.device_count() == 1:
    default_device = Device.CUDA
elif torch.cuda.device_count() > 1:
    default_device = Device.PARALLEL_CUDA

if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print('cuda gpus:',os.environ['CUDA_VISIBLE_DEVICES'])


def main(args):
    print(args)
    print('starting on', platform.node())
    
    if default_device != Device.CPU:
        main_stream = torch.cuda.Stream()

    # if args.fp16:
    #     assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    #     print('Got fp16!')
    
    taskonomy_loss, losses, criteria, taskonomy_tasks = get_losses_and_tasks(args)

    print("including the following tasks:", list(losses.keys()))

    criteria2={'Loss':taskonomy_loss}
    for key,value in criteria.items():
        criteria2[key]=value
    criteria = criteria2

    print('data_dir =',args.data_dir, len(args.data_dir))
    train_dataset = TaskonomyLoader(
        args.data_dir,
        label_set=taskonomy_tasks,
        model_whitelist=args.train_models,
        model_limit=args.model_limit,
        output_size = (args.image_size,args.image_size),
        augment=True, 
        partition=args.partition)

    print('Found',len(train_dataset),'training instances.')

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](tasks=losses.keys())

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            #print(p.size())
            nn=1
            for s in list(p.size()):
                
                nn = nn*s
            pp += nn
        return pp

    print("Model has", get_n_params(model), "parameters")
    try:
        print("Encoder has", get_n_params(model.encoder), "parameters")
        #flops, params=get_model_complexity_info(model.encoder,(256,256), as_strings=False, print_per_layer_stat=False)
        #print("Encoder has", flops, "Flops and", params, "parameters,")
    except:
        print("Each encoder has", get_n_params(model.encoders[0]), "parameters")
    for decoder in model.task_to_decoder.values():
        print("Decoder has", get_n_params(decoder), "parameters")

    model = model.to(device=default_device.value)

    # optionally resume from a checkpoint
    checkpoint=None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.to(device=default_device.value))
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    #o_model = model
    # if args.fp16:
    #     print('making network fp16')
    #     model = network_to_half(model)


    if args.pretrained != '':
        print('loading pretrained weights for '+args.arch+' ('+args.pretrained+')')
        model[1].encoder.load_state_dict(torch.load(args.pretrained))
    

    if default_device == Device.PARALLEL_CUDA:
        model = torch.nn.DataParallel(model).cuda()

    print('Virtual batch size =', args.batch_size*args.virtual_batch_multiplier)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    
    #tested with adamW. Poor results observed
    #optimizer = adamW.AdamW(model.parameters(),lr= args.lr,weight_decay=args.weight_decay,eps=1e-3)
    
    # if args.fp16:
    #     sys.stdout = open(os.devnull, "w") # capture junky output
    #     optimizer = FP16_Optimizer(optimizer,
    #                                dynamic_loss_scale=True)
    #     sys.stdout = sys.__stdout__ # stop capturing junky output

    
    if args.resume:
        if os.path.isfile(args.resume):
            optimizer.load_state_dict(checkpoint['optimizer'])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = get_eval_loader(args.data_dir, taskonomy_tasks, args)
    
    trainer=Trainer(train_loader,val_loader,model,optimizer,criteria,args,checkpoint)
    if args.validate:
        trainer.progress_table=[]
        trainer.validate([{}])
        print()
        return

    if args.test:
        trainer.progress_table=[]
        # replace val loader with a loader that loads test data
        trainer.val_loader=get_eval_loader(args.data_dir, taskonomy_tasks, args,model_limit=(1000,2000))
        trainer.validate([{}])
        return

    trainer.train()
   

def get_eval_loader(datadir, label_set, args, model_limit=1000):
    print(datadir)

    val_dataset = TaskonomyLoader(datadir,
                                  label_set=label_set,
                                  model_whitelist=args.val_models,
                                  model_limit=model_limit,
                                  output_size = (args.image_size,args.image_size),
                                  augment=False,
                                  partition=False) # regardless of n or 5n for training, val is always n.
    print('Found',len(val_dataset),'validation instances.')
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=max(args.batch_size//2,1), shuffle=False,
        num_workers=args.workers, pin_memory=True,sampler=None)
    return val_loader

program_start_time = time.time()

def on_keyboared_interrupt(x,y):
    #print()
    sys.exit(1)
signal.signal(signal.SIGINT, on_keyboared_interrupt)

def get_average_learning_rate(optimizer):
    try:
        return optimizer.learning_rate
    except:
        s = 0
        for param_group in optimizer.param_groups:
            s+=param_group['lr']
        return s/len(optimizer.param_groups)

class data_prefetcher():
    def __init__(self, loader):
        self.inital_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream() if default_device != Device.CPU else None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            # self.next_input = None
            # self.next_target = None
            self.loader = iter(self.inital_loader)
            self.preload()
            return

        if self.stream is not None:
          with torch.cuda.stream(self.stream):
              self.next_input = {key: val.cuda(non_blocking=True) for (key,val) in self.next_input.items()} 
              #self.next_target = self.next_target.cuda(async=True)
              self.next_target = {key: val.cuda(non_blocking=True) for (key,val) in self.next_target.items()}

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.std= 0
        self.sum = 0
        self.sumsq = 0
        self.count = 0
        self.lst = []

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        #self.sumsq += float(val)**2
        self.count += n
        self.avg = self.sum / self.count
        self.lst.append(self.val)
        self.std=np.std(self.lst)
        #try:
        #    self.std = ((self.sumsq-self.count*(self.avg**2))/(self.count-1))**.5
        #except:
        #    self.std=-1


class Trainer:
    def __init__(self,train_loader,val_loader,model,optimizer,criteria,args,checkpoint=None):
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.train_prefetcher=data_prefetcher(self.train_loader)
        self.model=model
        self.optimizer=optimizer
        self.criteria=criteria
        self.args = args
        # self.fp16=args.fp16
        self.code_archive=self.get_code_archive()
        if checkpoint:
            self.progress_table = checkpoint['progress_table']
            self.start_epoch = checkpoint['epoch']+1
            self.best_loss = checkpoint['best_loss']
            self.stats = checkpoint['stats']
            self.loss_history = checkpoint['loss_history']
        else:
            self.progress_table=[]
            self.best_loss = 9e9
            self.stats = []
            self.start_epoch = 0
            self.loss_history=[]
        
        self.lr0 = get_average_learning_rate(optimizer)
            
        self.ticks=0
        self.last_tick=0
        self.loss_tracking_window = args.loss_tracking_window_initial

        # Output metrics file
        csv_entries = ['epoch', 'batch_index', 'number_of_batches', 'learning_rate', 
                        'starting_learning_rate', 'eta', 'd%', 'ETA', 'ttest', 'Loss',
                        'val-Loss', 'val-eta']
        csv_entries += list(self.criteria.keys()) + ['val-'+s for s in self.criteria.keys()]

        os.makedirs(args.metrics_directory, exist_ok=True)
        output_metrics_filename = self.prefix() + '_' + datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S") + ".csv"
        output_metrics_filename = os.path.join(args.metrics_directory, output_metrics_filename)
        self.csv_file = open(output_metrics_filename , 'w')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=csv_entries, delimiter=',', 
                                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.csv_writer.writeheader()

        print(f"Metrics written to CSV file: {output_metrics_filename}")

    def __del__(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

    def get_code_archive(self):
        file_contents={}
        for i in os.listdir('.'):
            if i[-3:]=='.py':
                with open(i,'r') as file:
                    file_contents[i]=file.read()
        return file_contents

    def train(self):
        for self.epoch in range(self.start_epoch,self.args.epochs):
            current_learning_rate = get_average_learning_rate(self.optimizer)
            if current_learning_rate < self.args.minimum_learning_rate:
                break
            # train for one epoch
            train_string, train_stats = self.train_epoch()

            # evaluate on validation set
            progress_string=train_string
            loss, progress_string, val_stats = self.validate(progress_string)
            print()

            self.progress_table.append(progress_string)

            self.stats.append((train_stats,val_stats))
            self.checkpoint(loss)

    def prefix(self):
        params = [self.args.experiment_name, self.args.arch, ('p' if self.args.pretrained != '' else 'np'), self.args.tasks]
        return '_'.join(params)

    def checkpoint(self, loss):
        is_best = loss < self.best_loss
        self.best_loss = min(loss, self.best_loss)
        save_filename = self.prefix()+'_checkpoint.pth.tar'

        try:
            to_save = self.model
            if torch.cuda.device_count() >1:
                to_save=to_save.module
            # if self.args.fp16:
            #     to_save=to_save[1]
            #     to_save = to_save.float()
            #     #copy_in_params(model,param_copy)
            gpus='all'
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                gpus=os.environ['CUDA_VISIBLE_DEVICES']
            self.save_checkpoint({
                'epoch': self.epoch,
                'info':{'machine':platform.node(), 'GPUS':gpus},
                'args': self.args,
                'arch': self.args.arch,
                'state_dict': to_save.state_dict(),
                'best_loss': self.best_loss,
                'optimizer' : self.optimizer.state_dict(),
                'progress_table' : self.progress_table,
                'stats': self.stats,
                'loss_history': self.loss_history,
                'code_archive':self.code_archive
            }, False, self.args.model_dir, save_filename)

            if is_best:
                self.save_checkpoint(None, True,self.args.model_dir, save_filename)
            # if self.args.fp16:
            #     to_save = network_to_half(to_save)
        except Exception as e:
            print(e)
            print('save checkpoint failed...')              




    def save_checkpoint(self,state, is_best,directory='', filename='checkpoint.pth.tar'):
        path = os.path.join(directory,filename)
        if is_best:
            best_path = os.path.join(directory,'best_'+filename)
            shutil.copyfile(path, best_path)
        else:
            # print("saving checkpoint to: {}".format(path))
            torch.save(state, path)

    def learning_rate_schedule(self):
        ttest_p=0
        z_diff=0

        #don't reduce learning rate until the second epoch has ended
        if self.epoch < 2:
            return 0,0
        
        wind=self.loss_tracking_window//(self.args.batch_size*args.virtual_batch_multiplier)
        if len(self.loss_history)-self.last_tick > wind:
            a = self.loss_history[-wind:-wind*5//8]
            b = self.loss_history[-wind*3//8:]
            #remove outliers
            a = sorted(a)
            b = sorted(b)
            a = a[int(len(a)*.05):int(len(a)*.95)]
            b = b[int(len(b)*.05):int(len(b)*.95)]
            length_=min(len(a),len(b))
            a=a[:length_]
            b=b[:length_]
            z_diff,ttest_p = scipy.stats.ttest_rel(a,b,nan_policy='omit')

            if z_diff < 0 or ttest_p > .99:
                self.ticks+=1
                self.last_tick=len(self.loss_history)
                self.adjust_learning_rate()
                self.loss_tracking_window = min(self.args.maximum_loss_tracking_window,self.loss_tracking_window*2)
        return ttest_p, z_diff

    def train_epoch(self):
        global program_start_time
        average_meters = defaultdict(AverageMeter)
        display_values = []
        for name,func in self.criteria.items():
            display_values.append(name)

        # switch to train mode
        self.model.train()

        end = time.time()
        epoch_start_time = time.time()
        epoch_start_time2=time.time()

        batch_num = 0
        num_data_points=len(self.train_loader)//self.args.virtual_batch_multiplier
        if num_data_points > 10000:
            num_data_points = num_data_points//5
            
        starting_learning_rate=get_average_learning_rate(self.optimizer)
        csv_entry = {}

        while True:
            if batch_num ==0:
                end=time.time()
                epoch_start_time2=time.time()
            #print("Num data points", num_data_points, "\tBatch num", batch_num)
            if num_data_points==batch_num:
                break

            loss_dict=None
            loss=0

            # accumulate gradients over multiple runs of input
            for _ in range(self.args.virtual_batch_multiplier):
                data_start = time.time()
                input, target = self.train_prefetcher.next()
                average_meters['data_time'].update(time.time() - data_start)
                loss_dict2,loss2 = self.train_batch(input,target)
                loss+=loss2
                if loss_dict is None:
                    loss_dict=loss_dict2
                else:
                    for key,value in loss_dict2.items():
                        loss_dict[key]+=value
            
            # divide by the number of accumulations
            loss/=self.args.virtual_batch_multiplier
            for key,value in loss_dict.items():
                loss_dict[key]=value/self.args.virtual_batch_multiplier
            
            # do the weight updates and set gradients back to zero
            self.update()

            self.loss_history.append(float(loss))
            ttest_p, z_diff = self.learning_rate_schedule()


            for name,value in loss_dict.items():
                try:
                    average_meters[name].update(value.data)
                except:
                    average_meters[name].update(value)

            # time since start of program:
            if batch_num==0 and self.epoch==0:
                print('Time since program start:', time.time() - program_start_time, 'seconds.')
                print()

            elapsed_time_for_epoch = (time.time()-epoch_start_time2)
            eta = (elapsed_time_for_epoch/(batch_num+.2))*(num_data_points-batch_num)
            if eta >= 24*3600:
                eta = 24*3600-1

            
            batch_num+=1
            current_learning_rate= get_average_learning_rate(self.optimizer)

            csv_entry = {}
            csv_entry['epoch'] = ('{0}:').format(self.epoch)
            # csv_entry['batch_index'] = ('{0}').format(batch_num)
            # csv_entry['number_of_batches'] = num_data_points
            csv_entry['learning_rate'] = current_learning_rate
            csv_entry['starting_learning_rate'] = starting_learning_rate
            csv_entry['eta'] = ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
            csv_entry['d%'] = ('{0:0.2g}').format(100*average_meters['data_time'].sum/elapsed_time_for_epoch)

            for task_name in display_values:
                meter = average_meters[task_name]
                csv_entry[task_name]= ('{meter.avg:.4g}').format(meter=meter)

            if batch_num < num_data_points-1:
                csv_entry['ETA']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta+elapsed_time_for_epoch))))
                csv_entry['ttest']= ('{0:0.3g},{1:0.3g}').format(z_diff,ttest_p)


        epoch_time = time.time()-epoch_start_time
        stats={'batches':num_data_points,
            'learning_rate':current_learning_rate,
            'Epoch time':epoch_time,
            }
        for name in display_values:
            meter = average_meters[name]
            stats[name] = meter.avg

        data_time = average_meters['data_time'].sum

        csv_entry['eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
        # self.csv_writer.writerow(csv_entry)
        
        return csv_entry, stats



    def train_batch(self, input, target):

        loss_dict = {}
        
        # if self.args.fp16:
        #     input = input.half()
        output = self.model(input)
        first_loss=None
        for c_name,criterion_fun in self.criteria.items():
            if first_loss is None:first_loss=c_name
            loss_dict[c_name]=criterion_fun(output, target)

        loss = loss_dict[first_loss].clone()
        loss = loss / self.args.virtual_batch_multiplier
            
        # if self.args.fp16:
        #     self.optimizer.backward(loss)
        # else:
        #     loss.backward()
        loss.backward() # Temporary due to fp16

        return loss_dict,loss

    
    def update(self):

        # sys.stdout = open(os.devnull, "w")
        self.optimizer.step()
        sys.stdout = sys.__stdout__
        # if self.args.fp16:
        #     torch.cuda.synchronize()
        # sys.stdout = open(os.devnull, "w")
        self.optimizer.zero_grad()
        sys.stdout = sys.__stdout__
 

    def validate(self, train_table):
        average_meters = defaultdict(AverageMeter)
        self.model.eval()
        epoch_start_time = time.time()
        batch_num=0
        num_data_points=len(self.val_loader)

        csv_entry = train_table
        prefetcher = data_prefetcher(self.val_loader)
        if default_device != Device.CPU:
            torch.cuda.empty_cache()
        with torch.no_grad():
            for i in range(len(self.val_loader)):
                input, target = prefetcher.next()


                if batch_num ==0:
                    epoch_start_time2=time.time()

                output = self.model(input)
                

                loss_dict = {}
                
                for c_name,criterion_fun in self.criteria.items():
                    loss_dict[c_name]=criterion_fun(output, target)
                
                batch_num=i+1

                for name,value in loss_dict.items():    
                    try:
                        average_meters[name].update(value.data)
                    except:
                        average_meters[name].update(value)
                eta = ((time.time()-epoch_start_time2)/(batch_num+.2))*(len(self.val_loader)-batch_num)

                # csv_entry['#/{0}'.format(num_data_points)]= ('{0}').format(batch_num)
                csv_entry['val-eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(eta))))
                for name in self.criteria.keys():
                    meter = average_meters[name]
                    csv_entry['val-'+name]= ('{meter.avg:.4g}').format(meter=meter)

        epoch_time = time.time()-epoch_start_time

        stats={'batches':len(self.val_loader),
            'Epoch time':epoch_time,
            }
        ultimate_loss = None
        for name in self.criteria.keys():
            meter = average_meters[name]
            stats[name]=meter.avg
        ultimate_loss = stats['Loss']
        csv_entry['val-eta']= ('{0}').format(time.strftime("%H:%M:%S", time.gmtime(int(epoch_time))))
        
        self.csv_writer.writerow(csv_entry)
        self.csv_file.flush()

        if default_device != Device.CPU:
            torch.cuda.empty_cache()
        return float(ultimate_loss), csv_entry , stats

    def adjust_learning_rate(self):
        self.lr = self.lr0 * (0.50 ** (self.ticks))
        self.set_learning_rate(self.lr)

    def set_learning_rate(self,lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



if __name__ == '__main__':
    #mp.set_start_method('forkserver')
    args = parser.parse_args()
    main(args)
