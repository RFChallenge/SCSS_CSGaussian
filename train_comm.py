import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from tqdm import tqdm
import pickle

import rfcutils.srrcgaussian_helper_fn as srrcfn
import rfcutils.ofdmgaussian_helper_fn as ofdmfn
soi_type = 'RRCGaussian'
interference_sig_type = 'OFDMGaussian'

import random
from src import unet_model as unet
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

sig_len = 1280

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


all_sig_mixture, all_sig1, all_sig2 = [], [] , []
all_val_sig_mixture, all_val_sig1, all_val_sig2 = [], [] , []

for idx in tqdm(range(10000)):
    for target_sinr in np.arange(-30,4,1.5):
        sig1, _, _, _ = srrcfn.generate_srrc_signal(sig_len//16 + 80)
        start_idx0 = np.random.randint(len(sig1)-sig_len)
        sig1 = sig1[start_idx0:start_idx0+sig_len]
        data2, _, _ = ofdmfn.generate_ofdm_signal(sig_len//80*56+4*56)
        start_idx = np.random.randint(len(data2)-sig_len)
        sig2 = data2[start_idx:start_idx+sig_len]

        coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr/10))))
        
        noise = 0.1 * 1./np.sqrt(2) * (np.random.randn(sig_len) + 1j*np.random.randn(sig_len))
        sig_mixture = sig1 + sig2 * coeff + noise
        all_sig_mixture.append(sig_mixture)
        all_sig1.append(sig1)
        all_sig2.append(sig2*coeff)
        
for idx in tqdm(range(500)):
    for target_sinr in np.arange(-30,4,1.5):
        sig1, _, _, _ = srrcfn.generate_srrc_signal(sig_len//16 + 80)
        start_idx0 = np.random.randint(len(sig1)-sig_len)
        sig1 = sig1[start_idx0:start_idx0+sig_len]

        data2, _, _ = ofdmfn.generate_ofdm_signal(sig_len//80*56+4*56)
        start_idx = np.random.randint(len(data2)-sig_len)
        sig2 = data2[start_idx:start_idx+sig_len]

        coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr/10))))
        
        noise = 0.1 * 1./np.sqrt(2) * (np.random.randn(sig_len) + 1j*np.random.randn(sig_len))
        sig_mixture = sig1 + sig2 * coeff + noise
        all_val_sig_mixture.append(sig_mixture)
        all_val_sig1.append(sig1)
        all_val_sig2.append(sig2*coeff)
        
all_sig_mixture = np.array(all_sig_mixture)
all_sig1 = np.array(all_sig1)
all_sig2 = np.array(all_sig2)

all_val_sig_mixture = np.array(all_val_sig_mixture)
all_val_sig1 = np.array(all_val_sig1)
all_val_sig2 = np.array(all_val_sig2)

window_len = sig_len

sig1_out = all_sig1.reshape(-1,window_len)
out1_comp = np.dstack((sig1_out.real, sig1_out.imag))

sig_mix_out = all_sig_mixture.reshape(-1,window_len)
mixture_bands_comp = np.dstack((sig_mix_out.real, sig_mix_out.imag))

sig1_val_out = all_val_sig1.reshape(-1,window_len)
out1_val_comp = np.dstack((sig1_val_out.real, sig1_val_out.imag))

sig_mix_val_out = all_val_sig_mixture.reshape(-1,window_len)
mixture_bands_val_comp = np.dstack((sig_mix_val_out.real, sig_mix_val_out.imag))

for long_k_sz in [101, 11, 3]:
    def scheduler(epoch, lr=0.0003):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.04)
    lr_callback = LearningRateScheduler(scheduler)
    
    model_name = f'{soi_type}_{interference_sig_type}_{window_len}_K{long_k_sz}'
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    checkpoint = ModelCheckpoint(filepath=f'unet_comm_models/tmp/{model_name}/checkpoint', monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)    
    print(f'Training {model_name}')

    nn_model = unet.get_unet_model0((window_len, 2), k_sz=3, long_k_sz=long_k_sz, k_neurons=32, lr=0.0003)
    nn_model.fit(mixture_bands_comp, out1_comp, epochs=2000, batch_size=32, shuffle=True, verbose=1, validation_data=(mixture_bands_val_comp, out1_val_comp), callbacks=[checkpoint, earlystopping, lr_callback])    