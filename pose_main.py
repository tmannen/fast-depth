import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
cudnn.benchmark = True

import models
import models_pose
from metrics import AverageMeter, Result
import utils

args = utils.parse_command()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # Data loading code
    print("=> creating data loaders...")
    # TODO: Test - change later
    valdir = os.path.join('..', 'data', args.data, 'val')
    testdir = args.test_path
    traindir = args.train_path

    if args.data == 'sun3d':
        from dataloaders.sun3d import Sun3DDataset
        val_dataset = Sun3DDataset(testdir, split='val', modality=args.modality)
        train_dataset = Sun3DDataset(traindir, split='train', modality=args.modality)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=4, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=1   , shuffle=False, num_workers=args.workers, pin_memory=True)
    print("=> data loaders created.")

    # evaluation mode
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate, map_location='cpu')
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
        output_directory = os.path.dirname(args.evaluate)
        validate(val_loader, model, args.start_epoch, write_to_file=False)
        return

    # Train from start
    if args.train:
        model = models_pose.MobileNetSkipAddAlt(10)
        args.start_epoch = 0
        output_directory = os.path.dirname(args.train)
        train(train_loader, model, args.start_epoch)
        return

def save_prediction(img, path, img_name):
    pred_dir = os.path.join(path, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    image_path = os.path.join(pred_dir, img_name)
    utils.save_image(img, image_path)

def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target, pose) in enumerate(val_loader):
        #input, target, pose = input.cuda(), target.cuda(), pose.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input, pose)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 10

        if args.modality == 'rgb':
            rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8*skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8*skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        rgb_img, target_img, pred_img = utils.merge_into_row(rgb, target, pred, separate=True)
        row = utils.merge_into_row(rgb, target, pred)
        save_prediction(row, output_directory, str(i) + ".png")

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

def train(train_loader, model, epoch):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    # TODO: continue from here, test if training works
    for i, (input, target, pose) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(input, pose)
        loss = torch.sqrt(criterion(output, target))
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
