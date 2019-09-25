import torch
import collections

sl=0
nl=0
nl2=0
nl3=0
dl=0
el=0
rl=0
kl=0
tl=0
popular_offsets=collections.defaultdict(int)
batch_number=0

def segment_semantic_loss(output,target,mask):
    global sl
    sl = torch.nn.functional.cross_entropy(output.float(),target.long().squeeze(dim=1),ignore_index=0,reduction='mean')
    return sl

def normal_loss(output,target,mask):
    global nl
    nl= rotate_loss(output,target,mask,normal_loss_base)
    return nl
    
def normal_loss_simple(output,target,mask):
    global nl
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask.float()
    nl = out.mean()
    return nl
   
def rotate_loss(output,target,mask,loss_name):
    global popular_offsets
    target=target[:,:,1:-1,1:-1].float()
    mask = mask[:,:,1:-1,1:-1].float()
    output=output.float()
    val1 = loss = loss_name(output[:,:,1:-1,1:-1],target,mask)
    
    val2 = loss_name(output[:,:,0:-2,1:-1],target,mask)
    loss = torch.min(loss,val2)
    val3 = loss_name(output[:,:,1:-1,0:-2],target,mask)
    loss = torch.min(loss,val3)
    val4 = loss_name(output[:,:,2:,1:-1],target,mask)
    loss = torch.min(loss,val4)
    val5 = loss_name(output[:,:,1:-1,2:],target,mask)
    loss = torch.min(loss,val5)
    val6 = loss_name(output[:,:,0:-2,0:-2],target,mask)
    loss = torch.min(loss,val6)
    val7 = loss_name(output[:,:,2:,2:],target,mask)
    loss = torch.min(loss,val7)
    val8 = loss_name(output[:,:,0:-2,2:],target,mask)
    loss = torch.min(loss,val8)
    val9 = loss_name(output[:,:,2:,0:-2],target,mask)
    loss = torch.min(loss,val9)
    
    #lst = [val1,val2,val3,val4,val5,val6,val7,val8,val9]

    #print(loss.size())
    loss=loss.mean()
    #print(loss)
    return loss


def normal_loss_base(output,target,mask):
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask
    out = out.mean(dim=(1,2,3))
    return out

def normal2_loss(output,target,mask):
    global nl3
    diff = output.float() - target.float()
    out = torch.abs(diff)
    out = out*mask.float()
    nl3 = out.mean()
    return nl3

def depth_loss_simple(output,target,mask):
    global dl
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask.float()
    dl = out.mean()
    return dl

def depth_loss(output,target,mask):
    global dl
    dl = rotate_loss(output,target,mask,depth_loss_base)
    return dl

def depth_loss_base(output,target,mask):
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask.float()
    out = out.mean(dim=(1,2,3))
    return out

def edge_loss(output,target,mask):
    global el
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask
    el = out.mean()
    return el

def reshade_loss(output,target,mask):
    global rl
    out = torch.nn.functional.l1_loss(output,target,reduction='none')
    out *=mask
    rl = out.mean()
    return rl

def keypoints2d_loss(output,target,mask):
    global kl
    kl = torch.nn.functional.l1_loss(output,target)
    return kl

def edge2d_loss(output,target,mask):
    global tl
    tl = torch.nn.functional.l1_loss(output,target)
    return tl

def get_taskonomy_loss(losses):
    def taskonomy_loss(output,target):
        if 'mask' in target:
            mask = target['mask']
        else:
            mask=None

        sum_loss=None
        num=0
        for n,t in target.items():
            if n in losses:
                o = output[n].float()
                this_loss = losses[n](o,t,mask)
                num+=1
                if sum_loss:
                    sum_loss = sum_loss+ this_loss
                else:
                    sum_loss = this_loss
        
        return sum_loss#/num # should not take average when using xception_taskonomy_new
    return taskonomy_loss


def get_losses_and_tasks(args):
    task_str = args.tasks
    losses = {}
    criteria = {}
    taskonomy_tasks = []
    if 's' in task_str:
        losses['segment_semantic'] = segment_semantic_loss
        criteria['ss_l']=lambda x,y : sl
        taskonomy_tasks.append('segment_semantic')
    if 'd' in task_str:
        if args.no_rotate_loss:
            losses['depth_zbuffer'] = depth_loss_simple
        else:            
            losses['depth_zbuffer'] = depth_loss
        criteria['depth_l']=lambda x,y : dl
    if 'd' in task_str or 'n' in task_str:
        taskonomy_tasks.append('depth_zbuffer')
    if 'n' in task_str:
        if args.no_rotate_loss:
            losses['normal']=normal_loss_simple
        else:
            losses['normal']=normal_loss
        criteria['norm_l']=lambda x,y : nl
        #criteria['norm_l2']=lambda x,y : nl2
        taskonomy_tasks.append('normal')
    if 'N' in task_str:
        losses['normal2']=normal2_loss
        criteria['norm2']=lambda x,y : nl3
        taskonomy_tasks.append('normal2')
    if 'e' in task_str:
        losses['edge_occlusion']=edge_loss
        criteria['edge_l']=lambda x,y : el
        taskonomy_tasks.append('edge_occlusion')
    if 'r' in task_str:
        losses['reshading']=reshade_loss
        criteria['shade_l']=lambda x,y : rl
        taskonomy_tasks.append('reshading')
    if 'k' in task_str:
        losses['keypoints2d']=keypoints2d_loss
        criteria['key_l']=lambda x,y : kl
        taskonomy_tasks.append('keypoints2d')
    if 't' in task_str:
        losses['edge_texture']=edge2d_loss
        criteria['edge2d_l']=lambda x,y : tl
        taskonomy_tasks.append('edge_texture')

    taskonomy_loss = get_taskonomy_loss(losses)
    return taskonomy_loss,losses, criteria, taskonomy_tasks