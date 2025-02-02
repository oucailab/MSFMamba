import time
import os
import logging
import torch
import torch.nn.functional as F
from setting.dataLoader import get_loader
from setting.utils import compute_accuracy
from setting.utils import create_folder, random_seed_setting
from setting.options import opt
from model.MSFMamba import Net
import yaml
import numpy as np
import utility

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
random_seed_setting(6)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
GPU_NUMS = torch.cuda.device_count()


save_path = create_folder(opt.save_path + opt.dataset)
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

log_dir = save_path + '/log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(save_path+'/weight/'):
    os.makedirs(save_path+'/weight/')

logging.basicConfig(filename=log_dir + opt.dataset + current_time + 'log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,
                    filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')

logging.info(f'********************start train!********************')
logging.info(f'Config--epoch:{opt.epoch}; lr:{opt.lr}; batch_size:{opt.batchsize};')


# load data
train_loader,test_loader,trntst_loader,all_loader,train_num,val_num,trntst_num=get_loader(dataset=opt.dataset, 
        batchsize=opt.batchsize,num_workers=opt.num_work,useval=opt.useval, pin_memory=True)

logging.info(f'Loading data, including {train_num} training images and {val_num} \
        validation images and {trntst_num} train_test images')

model = Net(opt.dataset).cuda()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), opt.lr)

criterion = torch.nn.CrossEntropyLoss().cuda()


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    model.train()
    loss_all = 0
    iteration = len(train_loader)
    acc=0
    num=0
    for i, (hsi, Xdata, hsi_pca, gt,h,w) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        hsi = hsi.cuda()
        Xdata = Xdata.cuda()
        hsi_pca = hsi_pca.cuda()
        gt = gt.cuda()
        _ , outputs = model(hsi_pca.unsqueeze(1),Xdata)
        gt_loss = criterion(outputs,gt) 
        loss = gt_loss
        loss.backward()
        optimizer.step()
        loss_all += loss.detach()
        acc += compute_accuracy(outputs , gt) * len(gt)
        num += len(gt)
    loss_avg = loss_all / iteration
    acc_avg = acc / num
    logging.info(f'Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_train_avg: {loss_avg:.4f},acc_avg:{acc_avg:.4f}')

    if(epoch==opt.epoch or epoch==opt.epoch//2):
        torch.save(optimizer.state_dict(),
                save_path+'/weight/' + current_time + opt.dataset + "_optimizer" + "Epoch" + str(epoch) + '.pth')
        torch.save(model.state_dict(), save_path+'/weight/' + current_time + opt.dataset + '_Net_epoch_{}.pth'.format(epoch))

best_acc = opt.best_acc
best_epoch = opt.best_epoch

def test(val_loader, model, epoch, save_path):
    global best_acc, best_epoch
    if(opt.dataset=='Berlin'):
        oa,aa,kappa,acc=utility.createBerlinReport(net=model, data=val_loader,device='cuda:0')
    if oa > best_acc:   
        best_acc, best_epoch = oa, epoch
        if(epoch>=1):
            torch.save(optimizer.state_dict(),
                save_path+'/weight/' + current_time + '_' + str(best_acc) +'_' + opt.dataset + "_optimizer" + "Epoch" + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_path+'/weight/' + current_time + '_' + str(best_acc) +'_' + opt.dataset + '_Net_epoch_{}.pth'.format(epoch))
    print(f'Epoch [{epoch:03d}/{opt.epoch:03d}]'
            f' best_acc={best_acc:.4f}, Best_epoch:{best_epoch:03d}')
    logging.info(f'Best_acc:{best_acc:.4f},Best_epoch:{best_epoch:03d}')


if __name__ == '__main__':
    print("Start train...")
    time_begin = time.time()

    for epoch in range(opt.start_epoch, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
        time_epoch = time.time()
        print(f"Time out:{time_epoch - time_begin:.2f}s\n")
        logging.info(f"Time out:{time_epoch - time_begin:.2f}s\n")