import argparse
import os
import torch
from collections import defaultdict
from train_taskonomy import print_table

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model_file', '-m', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--arch', '-a', metavar='ARCH', default='',
                    help='model architecture: ' +
                        ' (default: resnet18)')

parser.add_argument('--save_raw',default='')
parser.add_argument('--show_loss_plot','-s', action='store_true',
                    help='show loss plot')

args = parser.parse_args()




# def print_table(table_list, go_back=True):

#     if go_back:
#         print("\033[F",end='')
#         print("\033[K",end='')
#         for i in range(len(table_list)):
#             print("\033[F",end='')
#             print("\033[K",end='')


#     lens = defaultdict(int)
#     for i in table_list:
#         for ii,to_print in enumerate(i):
#             for title,val in to_print.items():
#                 lens[(title,ii)]=max(lens[(title,ii)],max(len(title),len(val)))
    

#     # printed_table_list_header = []
#     for ii,to_print in enumerate(table_list[0]):
#         for title,val in to_print.items():

#             print('{0:^{1}}'.format(title,lens[(title,ii)]),end=" ")
#     for i in table_list:
#         print()
#         for ii,to_print in enumerate(i):
#             for title,val in to_print.items():
#                 print('{0:^{1}}'.format(val,lens[(title,ii)]),end=" ",flush=True)
#     print()


def create_model():
    import mymodels as models
    try:
        model = models.__dict__[args.arch](num_classification_classes=1000,
                                       num_segmentation_classes=21,
                                       num_segmentation_classes2=90,
                                       normalize=False)
    except:
        model = models.__dict__[args.arch]()

    
    return model


if args.model_file:
    if os.path.isfile(args.model_file):
        print("=> loading checkpoint '{}'".format(args.model_file))
        checkpoint = torch.load(args.model_file)
        
        progress_table = checkpoint['progress_table']
        
        print_table(progress_table,False)

        if args.show_loss_plot:
            loss_history = checkpoint['loss_history']
            print(len(loss_history))
            print()
            import matplotlib.pyplot as plt
            loss_history2 = loss_history[200:]
            loss_history3 = []
            cur = loss_history2[0]
            for i in loss_history2:
                cur = .99*cur+i*.01
                loss_history3.append(cur)
            plt.plot(range(len(loss_history3)),loss_history3)
            plt.show()