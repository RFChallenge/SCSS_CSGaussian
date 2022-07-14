import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from tqdm import tqdm
import pickle

soi_type = 'CycloA'
interference_sig_type = 'CycloB'


import random
from src import unet_model as unet
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

import random
random.seed(123)
np.random.seed(123)

ns, nb = 11, 5
sig_len0 = (ns*nb)*10

block_w1 = np.random.randn(ns,ns)
block_w2 = np.random.randn(nb,nb)

w1, w2 = np.zeros((sig_len0, sig_len0)), np.zeros((sig_len0, sig_len0))
for i in range(sig_len0//ns):
    w1[i*ns:(i+1)*ns, i*ns:(i+1)*ns] = block_w1
for j in range(sig_len0//nb):
    w2[j*nb:(j+1)*nb, j*nb:(j+1)*nb] = block_w2

w1 = w1 / np.sqrt(np.mean(np.diag(np.matmul(w1,w1.conj().T))))
w2 = w2 / np.sqrt(np.mean(np.diag(np.matmul(w2,w2.conj().T))))

pickle.dump((w1, w2), open(f'cyclo_example_filter_{ns}_{nb}.pickle','wb'))

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


all_sig_mixture, all_sig1, all_sig2 = [], [] , []
all_val_sig_mixture, all_val_sig1, all_val_sig2 = [], [] , []

sig_len = 256
for idx in tqdm(range(10000)):
    for target_sinr in np.arange(-6,7,3):
        c1 = 1./np.sqrt(2) * (np.random.randn(sig_len0) + 1j*np.random.randn(sig_len0))
        c2 = 1./np.sqrt(2) * (np.random.randn(sig_len0) + 1j*np.random.randn(sig_len0))

        sig1 = np.matmul(w1, c1)
        sig2 = np.matmul(w2, c2)
        
        roll_idx1 = np.random.randint(len(sig1))
        roll_idx2 = np.random.randint(len(sig2))
        sig1 = np.roll(sig1, roll_idx1)[:sig_len]
        sig2 = np.roll(sig2, roll_idx2)[:sig_len]

        coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr/10))))
        
        noise = 0.1 * 1./np.sqrt(2) * (np.random.randn(sig_len) + 1j*np.random.randn(sig_len))
        sig_mixture = sig1 + sig2 * coeff + noise
        all_sig_mixture.append(sig_mixture)
        all_sig1.append(sig1)
        all_sig2.append(sig2*coeff)
        
for idx in tqdm(range(500)):
    for target_sinr in np.arange(-6,7,3):
        c1 = 1./np.sqrt(2) * (np.random.randn(sig_len0) + 1j*np.random.randn(sig_len0))
        c2 = 1./np.sqrt(2) * (np.random.randn(sig_len0) + 1j*np.random.randn(sig_len0))

        sig1 = np.matmul(w1, c1)
        sig2 = np.matmul(w2, c2)
        
        roll_idx1 = np.random.randint(len(sig1))
        roll_idx2 = np.random.randint(len(sig2))
        sig1 = np.roll(sig1, roll_idx1)[:sig_len]
        sig2 = np.roll(sig2, roll_idx2)[:sig_len]

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
    checkpoint = ModelCheckpoint(filepath=f'unet_cyclo_models/tmp/{model_name}/checkpoint', monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)    
    print(f'Training {model_name}')
    nn_model = unet.get_unet_model0((window_len, 2), k_sz=3, long_k_sz=long_k_sz, k_neurons=32, lr=0.0003)
    nn_model.fit(mixture_bands_comp, out1_comp, epochs=2000, batch_size=32, shuffle=True, verbose=1, validation_data=(mixture_bands_val_comp, out1_val_comp), callbacks=[checkpoint, earlystopping, lr_callback])