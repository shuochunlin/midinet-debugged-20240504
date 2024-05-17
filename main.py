import numpy as np 
import random 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import *
from ops import *

class get_dataloader(object):
    def __init__(self, data, prev_data, y):
        self.size = data.shape[0]
        self.data = torch.from_numpy(data).float()
        self.prev_data = torch.from_numpy(prev_data).float()
        self.y   = torch.from_numpy(y).float()

         # self.label = np.array(label)
    def __getitem__(self, index):
        return self.data[index],self.prev_data[index], self.y[index]

    def __len__(self):
        return self.size

def load_data(batch_size=72, has_val_test=0):
    #######load the data########
    check_range_st = 0
    check_range_ed = 129 # reduced from 129
    pitch_range = check_range_ed - check_range_st-1
    # print('pitch range: {}'.format(pitch_range))

    if has_val_test:
        X_tr = np.load('data/train-val-test/data_X_tr.npy')
        prev_X_tr = np.load('data/train-val-test/prev_X_tr.npy')
        y_tr    = np.load('data/train-val-test/data_y_tr.npy')  
        X_val = np.load('data/train-val-test/data_X_val.npy')
        prev_X_val = np.load('data/train-val-test/prev_X_val.npy')
        y_val    = np.load('data/train-val-test/data_y_val.npy')  
        # transpose val data
        X_val = np.transpose(X_val, (0,1,3,2))
        prev_X_val = np.transpose(prev_X_val, (0,1,3,2))

        # Reshape y_tr to have lower dimension (batch_size, 13) (temporary - needs a better writing during earlier stage)
        y_val = y_val.reshape(y_val.shape[0], -1)

        X_val = X_val[:,:,:,check_range_st:check_range_ed]
        prev_X_val = prev_X_val[:,:,:,check_range_st:check_range_ed]
    else:
        X_tr = np.load('data/data_X_tr.npy')
        prev_X_tr = np.load('data/prev_X_tr.npy')
        y_tr    = np.load('data/data_y_tr.npy')  
    
    # transpose training data
    X_tr = np.transpose(X_tr, (0,1,3,2))
    prev_X_tr = np.transpose(prev_X_tr, (0,1,3,2))

    # Reshape y_tr to have lower dimension (batch_size, 13) (temporary - needs a better writing during earlier stage)
    y_tr = y_tr.reshape(y_tr.shape[0], -1)

    print(np.shape(X_tr))
    print(np.shape(prev_X_tr))

    X_tr = X_tr[:,:,:,check_range_st:check_range_ed]
    prev_X_tr = prev_X_tr[:,:,:,check_range_st:check_range_ed]

    #test data shape(5048, 1, 16, 128)
    #train data shape(45448, 1, 16, 128)

    train_iter = get_dataloader(X_tr,prev_X_tr,y_tr)
    kwargs = {'num_workers': 4, 'pin_memory': True}# if args.cuda else {}
    train_loader = DataLoader(
                   train_iter, batch_size=batch_size, shuffle=True, **kwargs)
    
    val_loader = None
    if has_val_test:
        val_iter = get_dataloader(X_val,prev_X_val,y_val)
        kwargs = {'num_workers': 4, 'pin_memory': True}# if args.cuda else {}
        val_loader = DataLoader(
                    val_iter, batch_size=batch_size, shuffle=True, **kwargs)
        
        print('data preparation is completed, with validation set')
        return train_loader, val_loader
    else:
        print('data preparation is completed, without validation set')
        return train_loader
    

def main():
    is_train   = 1 #1
    is_draw    = 1 #1
    is_sample  = 1 #0

    model_id   = 105 # has augmentation

    has_val_test = 1  # 0 if just train/test split, 1 with validation

    epochs = 40
    lrD = 0.00015 # 0.0002
    lrG = 0.00025 # 0.0002

    check_range_st = 0
    check_range_ed = 129  # modified from 129
    pitch_range = check_range_ed - check_range_st-1

    label_smoothing_weight = 0.9  # default is 0.9 by author
    feature_matching_weight = 0.15  # default if 0.1 by author
    mean_image_weight = 0.01  # default if 0.01 by author
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_train == 1 :

        netG = generator(pitch_range).to(device)
        netD = discriminator(pitch_range).to(device)  

        netD.train()
        netG.train()
        optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999)) 
             
        batch_size = 64 # 72
        nz = 100
        fixed_noise = torch.randn(batch_size, nz, device=device)
        real_label = 1
        fake_label = 0
        average_lossD = 0
        average_lossG = 0
        average_D_x   = 0
        average_D_G_z = 0

        if has_val_test:
            train_loader, val_loader = load_data(batch_size=batch_size, has_val_test=1)
        else:
            train_loader = load_data(batch_size=batch_size, has_val_test=0)

        lossD_list =  []
        lossD_list_all = []
        lossG_list =  []
        lossG_list_all = []
        D_x_list = []
        D_G_z_list = []

        best_val_lossD = float('inf')
        best_val_lossG = float('inf')
        lr_adjust_cooldown = 0

        for epoch in range(epochs):
            sum_lossD = 0
            sum_lossG = 0
            sum_D_x   = 0
            sum_D_G_z = 0
            
            netD.train()
            netG.train()
            for i, (data,prev_data,chord) in enumerate(train_loader, 0):
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                netD.zero_grad()
                real_cpu = data.to(device)
                prev_data_cpu = prev_data.to(device)
                chord_cpu = chord.to(device)

                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label, device=device, dtype=int)
                D, D_logits, fm = netD(real_cpu,chord_cpu,batch_size,pitch_range)

                #####loss

                # label smoothing is used here, default label_smoothing_weight is 0.9
                d_loss_real = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, label_smoothing_weight*torch.ones_like(D)))
                d_loss_real.backward()
                D_x = D.mean().item()
                sum_D_x += D_x 

                # train with fake
                noise = torch.randn(batch_size, nz, device=device)
                fake = netG(noise,prev_data_cpu,chord_cpu,batch_size,pitch_range)
                label.fill_(fake_label)
                D_, D_logits_, fm_ = netD(fake.detach(),chord_cpu,batch_size,pitch_range)
                d_loss_fake = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.zeros_like(D_)))

                d_loss_fake.backward()
                D_G_z1 = D_.mean().item()
                errD = d_loss_real + d_loss_fake
                errD = errD.item()
                lossD_list_all.append(errD)
                sum_lossD += errD
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                # Commented out to train G network only once
                # netG.zero_grad()
                # label.fill_(real_label)  # fake labels are real for generator cost
                # D_, D_logits_, fm_= netD(fake,chord_cpu,batch_size,pitch_range)

                # ###loss
                # g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                # #Feature Matching
                # features_from_g = reduce_mean_0(fm_)
                # features_from_i = reduce_mean_0(fm)
                # fm_g_loss1 =torch.mul(l2_loss(features_from_g, features_from_i), feature_matching_weight)

                # mean_image_from_g = reduce_mean_0(fake)
                # smean_image_from_i = reduce_mean_0(real_cpu)
                # fm_g_loss2 = torch.mul(l2_loss(mean_image_from_g, smean_image_from_i), mean_image_weight)

                # errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                # errG.backward(retain_graph=True)
                # D_G_z2 = D_.mean().item()
                # optimizerG.step()
              
                ############################
                # (3) Update G network again: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                D, D_logits, fm = netD(real_cpu,chord_cpu,batch_size,pitch_range)
                D_, D_logits_, fm_ = netD(fake,chord_cpu,batch_size,pitch_range)

                ###loss
                g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                #Feature Matching
                features_from_g = reduce_mean_0(fm_)
                features_from_i = reduce_mean_0(fm)
                loss_ = nn.MSELoss(reduction='sum')
                feature_l2_loss = loss_(features_from_g, features_from_i)/2
                fm_g_loss1 =torch.mul(feature_l2_loss, feature_matching_weight) # feature_matching_weight, default 0.1

                mean_image_from_g = reduce_mean_0(fake)
                smean_image_from_i = reduce_mean_0(real_cpu)
                mean_l2_loss = loss_(mean_image_from_g, smean_image_from_i)/2   
                fm_g_loss2 = torch.mul(mean_l2_loss, mean_image_weight)     # mean_image_weight, default 0.01
                errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                sum_lossG +=errG
                errG.backward()
                lossG_list_all.append(errG.item())

                D_G_z2 = D_.mean().item()
                sum_D_G_z += D_G_z2
                optimizerG.step()
            
                if epoch % 5 == 0:
                    if i % 100 == 0:
                        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                            % (epoch, epochs, i, len(train_loader),
                                errD, errG, D_x, D_G_z1, D_G_z2))

                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                            '%s/real_samples.png' % 'file',
                            normalize=True)
                    fake = netG(fixed_noise,prev_data_cpu,chord_cpu,batch_size,pitch_range)
                    vutils.save_image(fake.detach(),
                            '%s/fake_samples_epoch_%03d.png' % ('file', epoch),
                            normalize=True)
                    
            if has_val_test:
                # use validation set for metrics
                netD.eval()
                netG.eval()

                # Initialize evaluation metrics
                total_lossD = 0
                total_D_x = 0
                total_D_G_z1 = 0
                num_batches = 0
                total_lossG = 0
                total_D_G_z2 = 0

                for i, (data,prev_data,chord) in enumerate(val_loader, 0):
                    # Move data to device
                    real_cpu = data.to(device)
                    prev_data_cpu = prev_data.to(device)
                    chord_cpu = chord.to(device)

                    batch_size = real_cpu.size(0)
                    label = torch.full((batch_size,), real_label, device=device, dtype=int)

                    # Forward pass through D with real data
                    D, D_logits, _ = netD(real_cpu, chord_cpu, batch_size, pitch_range)

                    # Compute loss for real data
                    d_loss_real = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, label_smoothing_weight * torch.ones_like(D)))
                    total_lossD += d_loss_real.item()
                    total_D_x += D.mean().item()

                    # !!!
                    #  Generate fake data
                    noise = torch.randn(batch_size, nz, device=device)
                    fake = netG(noise, prev_data_cpu, chord_cpu, batch_size, pitch_range)

                    # Forward pass through D with fake data
                    D_, D_logits_, fm_ = netD(fake, chord_cpu, batch_size, pitch_range)

                    # Compute loss for fake data
                    g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                    total_D_G_z1 += D_.mean().item()

                    # Feature matching loss
                    features_from_g = reduce_mean_0(fm_)
                    features_from_i = reduce_mean_0(fm)
                    loss_ = nn.MSELoss(reduction='sum')
                    feature_l2_loss = loss_(features_from_g, features_from_i) / 2
                    fm_g_loss1 = torch.mul(feature_l2_loss, feature_matching_weight) # feature_matching_weight, default 0.1
                    mean_image_from_g = reduce_mean_0(fake)
                    smean_image_from_i = reduce_mean_0(real_cpu)
                    mean_l2_loss = loss_(mean_image_from_g, smean_image_from_i) / 2
                    fm_g_loss2 = torch.mul(mean_l2_loss, mean_image_weight)    # mean_image_weight, default 0.01
                    errG = g_loss0 + fm_g_loss1 + fm_g_loss2

                    total_lossG += errG.item()
                    total_D_G_z2 += D_.mean().item()

                    num_batches += 1
                
                # Compute average losses and metrics
                average_lossD = total_lossD  / num_batches
                average_D_x = total_D_x / num_batches
                average_D_G_z1 = total_D_G_z1 / num_batches

                average_lossG = total_lossG  / num_batches
                average_D_G_z = total_D_G_z2 / num_batches  # z2 is the z in author's code so i'll use this

                # Print validation metrics
                print('==> Epoch: {}'.format(epoch))
                print('  (Val set) Avg lossD: {:.8f}, Avg lossG: {:.8f}, Avg D(x): {:.8f}, Avg D(G(z)): {:.8f}'.format(
                    average_lossD, average_lossG, average_D_x, average_D_G_z))
                
                lossD_list.append(average_lossD)
                lossG_list.append(average_lossG)            
                D_x_list.append(average_D_x)
                D_G_z_list.append(average_D_G_z)

                # assess validation loss to determine if LR should get changed, using G loss
                # if average_lossG < best_val_lossG:
                #     best_val_lossG = average_lossG
                #     lr_adjust_cooldown = 0
                #     # haven't implemented onto the optimizer yet
                # else:
                #     lr_adjust_cooldown += 1
                
                # if 5 epochs without improvement, lower learning rate by 0.5
                # if lr_adjust_cooldown >= 5 and epoch > 15:
                #     lr = max(0.000001, lr * 0.5)  # learning rate has to be above 0.000001
                #     lr_adjust_cooldown = 0
                #     print("Adjusting Learning Rate to:", lr)
                #     # haven't implemented onto the optimizer yet
                    
            else:
                # use training data metrics 
                # TODO: save both training loss and val loss

                average_lossD = (sum_lossD / len(train_loader.dataset))
                average_lossG = (sum_lossG / len(train_loader.dataset))
                average_D_x = (sum_D_x / len(train_loader.dataset))
                average_D_G_z = (sum_D_G_z / len(train_loader.dataset))

                lossD_list.append(average_lossD)
                lossG_list.append(average_lossG)            
                D_x_list.append(average_D_x)
                D_G_z_list.append(average_D_G_z)
                print('==> Epoch: {} Average lossD: {:.8f} average_lossG: {:.8f},average D(x): {:.8f},average D(G(z)): {:.8f} '.format(
                    epoch, average_lossD,average_lossG,average_D_x, average_D_G_z)) 

            # 1 epoch done, do some checkpointing
            torch.save(netG.state_dict(), '%s/%d_netG_epoch_%d.pth' % ('models', model_id, epoch))
            torch.save(netD.state_dict(), '%s/%d_netD_epoch_%d.pth' % ('models', model_id, epoch))

        # training complete - save list
        with torch.no_grad():
            np.save(f'lossD_list_{model_id}.npy',lossD_list)
            np.save(f'lossG_list_{model_id}.npy',lossG_list)
            np.save(f'lossD_list_all_{model_id}.npy',lossD_list_all)
            np.save(f'lossG_list_all_{model_id}.npy',lossG_list_all)
            np.save(f'D_x_list.npy_{model_id}',D_x_list)
            np.save(f'D_G_z_list.npy_{model_id}',D_G_z_list)
        

    if is_draw == 1:
        lossD_print = np.load(f'lossD_list_{model_id}.npy')
        lossG_print = np.load(f'lossG_list_{model_id}.npy')
        length = lossG_print.shape[0]

        x = np.linspace(0, length-1, length)
        x = np.asarray(x)
        plt.figure()
        plt.plot(x, lossD_print,label=' lossD',linewidth=1.5)
        plt.plot(x, lossG_print,label=' lossG',linewidth=1.5)

        plt.legend(loc='upper right')
        plt.xlabel('data')
        plt.ylabel('loss')
        plt.savefig(f'draw_figs/{model_id}_lrD'+ str(lrD) + '_lrG' + str(lrG) +'_epoch'+str(epochs)+'.png')

    if is_sample == 1:

        ###
        generate_samples_multiplier = 1 #  
        ###

        batch_size = 8
        nz = 100
        n_bars = 7
        X_te = np.load('data/data_X_te.npy')
        prev_X_te = np.load('data/prev_X_te.npy')

        # transpose test
        X_te = np.transpose(X_te, (0,1,3,2))
        prev_X_te = np.transpose(prev_X_te, (0,1,3,2))

        prev_X_te = prev_X_te[:,:,check_range_st:check_range_ed,:]

        y_te    = np.load('data/data_y_te.npy') 
       
        test_iter = get_dataloader(X_te,prev_X_te,y_te)
        kwargs = {'num_workers': 4, 'pin_memory': True}# if args.cuda else {}
        test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, **kwargs)

        netG = sample_generator(pitch_range)
        netG.load_state_dict(torch.load(f'models/{model_id}_netG_epoch_{epochs-1}.pth'))

        output_songs = []
        output_chords = []
        for i, (data,prev_data,chord) in enumerate(test_loader, 0):
            list_song = []
            # print(np.shape(data))

            # vary the first measure sample point
            for _ in range(generate_samples_multiplier):
                chosen_offset = 0  # always 0
            
                first_bar = data[chosen_offset].view(1,1,16,128)
                list_song.append(first_bar)

                list_chord = []
                first_chord = chord[chosen_offset].view(1,13).cpu().numpy()  # originally 13
                list_chord.append(first_chord)
                noise = torch.randn(batch_size, nz)

                for bar in range(n_bars):
                    z = noise[bar].view(1,nz)
                    y = chord[bar].view(1,13)
                    # print(y)
                    if bar == 0:
                        prev = data[chosen_offset].view(1,1,16,128)
                        # print("PREV", prev.shape)
                    else:
                        prev = list_song[bar-1].view(1,1,16,128)
                    sample = netG(z, prev, y, 1,pitch_range)
                    # print(sample.shape)
                    list_song.append(sample)
                    list_chord.append(y.numpy())

            if len(output_songs) % 10 == 0:
                print('num of output_songs: {}'.format(len(output_songs)))
            output_songs.append(list_song)
            output_chords.append(list_chord)
        
        with torch.no_grad():
            # Convert each tensor in the nested lists to a NumPy array
            # hardcoded for now
            output_songs_np = [[tensor.numpy() for tensor in sublist] for sublist in output_songs]#.view(5,8,-1)
            output_chords_np = [tensor for tensor in output_chords]#.view(5,8,-1)

            # Save the nested list of NumPy arrays to a file
            np.save(f'{model_id}_output_songs.npy', output_songs_np)
            # Convert each tensor in the list to a NumPy array
            np.save(f'{model_id}_output_chords.npy', output_chords_np) #np.asarray(output_chords))

        print('creation completed, check out what I make!')


if __name__ == "__main__" :
    torch.autograd.set_detect_anomaly(True)
    main()

