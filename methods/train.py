import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from methods.ARPL.arpl_utils import AverageMeter, torch_accuracy

from collections import OrderedDict
from tqdm import tqdm


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, x, y, y_a, y_b, lam):
    return lam * criterion(x, y, y_a)[1] + (1 - lam) * criterion(x, y, y_b)[1]


def train(net, criterion, optimizer, warmup_scheduler, trainloader, epoch=None, optimizer_h=None, **options):
    net.train()
    losses = AverageMeter()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    loss_all = 0
    pbar = tqdm(trainloader)

    for batch_idx, tuples in enumerate(pbar):
        if len(tuples) == 2:
            data, labels = tuples
        elif len(tuples) == 3:
            data, labels, idx = tuples
        pbar_dic = OrderedDict()
        data, labels = data.to(options['device']), labels.to(options['device'])

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            if optimizer_h is not None:
                optimizer_h.zero_grad()

            if options['mixup']:
                inputs, targets_a, targets_b, lam = mixup_data(data, labels)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                x, y = net(inputs, return_feat=True)
                loss = mixup_criterion(criterion, x, y, targets_a, targets_b, lam)
                logits = y
            else:
                x, y = net(data, return_feat=True)
                logits, loss = criterion(x, y, labels)
            loss.backward()

            optimizer.step()
            if optimizer_h is not None:
                optimizer_h.step()

        if options['in_dataset'] == 'cifar-100' and options['ablation'] == 'conv-default' and epoch <= 0:
            warmup_scheduler.step()

        losses.update(loss.item(), data.size(0))
        loss_all += losses.mean

        total += labels.size(0)
        correct += (logits.data.max(1)[1] == labels.data).sum()

        pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
        pbar_dic['Acc'] = '{:.2f}'.format(float(correct) * 100. / float(total))
        pbar.set_postfix(pbar_dic)



def train_godin(net, criterion, optimizer, optimizer_h, trainloader, **options):
    net.train()
    losses = AverageMeter()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    loss_all = 0
    pbar = tqdm(trainloader)

    for batch_idx, tuples in enumerate(pbar):
        if len(tuples) == 2:
            data, labels = tuples
        elif len(tuples) == 3:
            data, labels, idx = tuples
        pbar_dic = OrderedDict()
        data, labels = data.to(options['device']), labels.to(options['device'])

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            optimizer_h.zero_grad()

            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)
            
            loss.backward()
            optimizer.step()
            optimizer_h.step()
        
        losses.update(loss.item(), data.size(0))
        loss_all += losses.mean

        total += labels.size(0)
        correct += (logits.data.max(1)[1] == labels.data).sum()              

        pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
        pbar_dic['Acc'] = '{:.2f}'.format(float(correct) * 100. / float(total))
        pbar.set_postfix(pbar_dic)


def train_oe(net, optimizer, schedule, warmup_scheduler, trainloader, oeloader, epoch=None, optimizer_h=None, **options):
    net.train()  # enter train mode
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all, loss_avg = 0, 0
    correct, total = 0, 0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    oeloader.dataset.offset = np.random.randint(len(oeloader.dataset))
    pbar = tqdm(zip(trainloader, oeloader), total=len(trainloader))
    
    for in_set, out_set in pbar:
        data = torch.cat((in_set[0], out_set[0]), 0)
        labels = in_set[1]

        pbar_dic = OrderedDict()

        data, labels = data.to(options['device']), labels.to(options['device'])
        # forward
        x = net(data)

        # backward
        if options['scheduler'] == 'oe_opt':
            schedule.step()
        else:
            schedule.step(epoch=epoch)

        optimizer.zero_grad()
        if optimizer_h is not None:
            optimizer_h.zero_grad()

        loss = F.cross_entropy(x[:len(in_set[0])], labels)
        loss += options['lamb'] * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
        loss.backward()

        optimizer.step()
        warmup_scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        losses.update(loss_avg, data.size(0))
        loss_all += losses.mean

        total += labels.size(0)
        correct += (x[:len(in_set[0])].data.max(1)[1] == labels.data).sum()              

        pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
        pbar_dic['Acc'] = '{:.2f}'.format(float(correct) * 100. / float(total))
        pbar.set_postfix(pbar_dic)


def train_osr_oe(net, optimizer, schedule, warmup_scheduler, trainloader, oeloader, epoch=None, optimizer_h=None, **options):
    net.train()  # enter train mode
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all, loss_avg = 0, 0
    correct, total = 0, 0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    oeloader.dataset.offset = np.random.randint(len(oeloader.dataset))
    pbar = tqdm(zip(trainloader, oeloader), total=len(trainloader))
    
    for in_set, out_set in pbar:
        data = torch.cat((in_set[0], out_set[0]), 0)
        labels = in_set[1]

        pbar_dic = OrderedDict()

        data, labels = data.to(options['device']), labels.to(options['device'])
        # forward
        x, _ = net(data, True)

        # backward
        if options['scheduler'] == 'oe_opt':
            schedule.step()
        else:
            schedule.step(epoch=epoch)

        optimizer.zero_grad()
        if optimizer_h is not None:
            optimizer_h.zero_grad()

        loss = F.cross_entropy(x[:len(in_set[0])], labels)
        loss += options['lamb'] * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
        loss.backward()

        optimizer.step()
        if optimizer_h is not None:
            optimizer_h.step()

        if options['in_dataset'] == 'cifar-100' and options['ablation'] == 'conv-default' and epoch <= 0:
            warmup_scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        losses.update(loss_avg, data.size(0))
        loss_all += losses.mean

        total += labels.size(0)
        correct += (x[:len(in_set[0])].data.max(1)[1] == labels.data).sum()              

        pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
        pbar_dic['Acc'] = '{:.2f}'.format(float(correct) * 100. / float(total))
        pbar.set_postfix(pbar_dic)

        

def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, trainloader, optimizer_h=None, **options):
    print('train with confusing samples')
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    netD.train()
    netG.train()

    torch.cuda.empty_cache()
    
    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, tuples in enumerate(tqdm(trainloader)):
        if len(tuples) == 2:
            data, labels = tuples
        elif len(tuples) == 3:
            data, labels, idx = tuples
            
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        gan_target = gan_target.to(options['device'])
        
        data, labels = Variable(data), Variable(labels)
        
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).to(options['device'])
        noise = noise.to(options['device'])
        noise = Variable(noise)
        fake = netG(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()

        # train with fake
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterionD(output, targetv)

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        errG_F = criterion.fake_loss(x).mean()
        generator_loss = errG + options['beta'] * errG_F
        generator_loss.backward()
        optimizerG.step()

        lossesG.update(generator_loss.item(), labels.size(0))
        lossesD.update(errD.item(), labels.size(0))


        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        if optimizer_h is not None:
            optimizer_h.zero_grad()

        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        _, loss = criterion(x, y, labels)

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).to(options['device'])
        if options['device']:
            noise = noise.to(options['device'])
        noise = Variable(noise)
        fake = netG(noise)
        
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).to(options['device']))
        F_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + options['beta'] * F_loss_fake
        total_loss.backward()
        optimizer.step()
        if optimizer_h is not None:
            optimizer_h.step()

        losses.update(total_loss.item(), labels.size(0))
        loss_all += losses.mean

    print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
    .format(batch_idx+1, len(trainloader), losses.now, losses.mean, lossesG.now, lossesG.mean, lossesD.now, lossesD.mean))



def train_large(net, criterion, optimizer, trainloader, oeloader=None, **options):
    net.train()
    losses = AverageMeter()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    loss_all = 0
    pbar = tqdm(trainloader)

    if options['loss_strategy'] == 'OE':
        # oeloader.dataset.offset = np.random.randint(len(oeloader.dataset))
        for in_tuples, out_tuples in zip(trainloader, oeloader):
            if len(in_tuples) == 2:
                in_data, in_labels = in_tuples
            elif len(in_tuples) == 3:
                in_data, in_labels, idx = in_tuples            

            if len(out_tuples) == 2:
                out_data, out_labels = out_tuples
            elif len(out_tuples) == 3:
                out_data, out_labels, idx = out_tuples

            data = torch.cat((in_data, out_data), 0)            
            pbar_dic = OrderedDict()
            data, in_labels = data.to(options['device']), in_labels.to(options['device'])

            with torch.set_grad_enabled(True):
                x = net(data)

                # options['scheduler'].step()
                optimizer.zero_grad()

                logits, loss = criterion(x, x, in_labels)
                
                loss.backward()
                optimizer.step()
            
            losses.update(loss.item(), data.size(0))

            loss_all += losses.mean
            total += in_labels.size(0)
            correct += (logits[:len(in_labels)].data.max(1)[1] == in_labels.data).sum()            

            pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
            pbar_dic['Acc'] = '{:.2f}'.format(float(correct) * 100. / float(total))
            pbar.set_postfix(pbar_dic)

    else:
        for batch_idx, tuples in enumerate(pbar):
            if len(tuples) == 2:
                data, labels = tuples
            elif len(tuples) == 3:
                data, labels, idx = tuples
            pbar_dic = OrderedDict()
            data, labels = data.to(options['device']), labels.to(options['device'])

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                x, y = net(data, return_feat=True)
                logits, loss = criterion(x, y, labels)
                
                loss.backward()
                optimizer.step()
            
            losses.update(loss.item(), data.size(0))
            loss_all += losses.mean

            total += labels.size(0)
            correct += (logits.data.max(1)[1] == labels.data).sum()              

            pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
            pbar_dic['Acc'] = '{:.2f}'.format(float(correct) * 100. / float(total))
            pbar.set_postfix(pbar_dic)

    return loss_all
