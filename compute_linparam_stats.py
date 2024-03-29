import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from tqdm import tqdm
import pickle


##################################################
# Communication Example
##################################################
window_len = 1280
cov_true_srrc, cov_ofdm_sync = pickle.load(open('stats/comm_oracle_covariance.pickle','rb'))
for sinr_db in tqdm(np.arange(-30, 4, 1.5)):
    for tau_s in range(16):
        for tau_b in range(80):
            Css = cov_true_srrc[tau_s:tau_s+window_len, tau_s:tau_s+window_len]
            Cbb = cov_ofdm_sync[tau_b:tau_b+window_len, tau_b:tau_b+window_len]

            sinr = 10**(sinr_db/10)
            scaled_Cbb = Cbb * 1/sinr
            Cyy = Css + scaled_Cbb + 0.01*np.eye(window_len, dtype=complex)
            Csy = Css.copy()

            U,S,Vh = np.linalg.svd(Cyy,hermitian=True)
            sthr_idx = np.linalg.matrix_rank(Cyy) + 1
            Cyy_inv = np.matmul(U[:,:sthr_idx], np.matmul(np.diag(1.0/(S[:sthr_idx])), U[:,:sthr_idx].conj().T))

            log_det_Cyy = np.sum(np.log(S[:sthr_idx]))
            
            W = np.matmul(Csy, Cyy_inv)
            Ce = Css - np.matmul(W, Css.conj().T)
            pickle.dump((Cyy_inv, log_det_Cyy, W, Ce), open(f'stats/comm_param/filters_param_taus{tau_s}_taub{tau_b}_sinr{sinr_db:.01f}.pickle','wb'))


            
            
##################################################
# Cyclostationary Example
##################################################
ns, nb = 11, 5
w1, w2 = pickle.load(open(f'stats/cyclo_example_filter_{ns}_{nb}.pickle','rb'))
sig_len0, sig_len = (ns*nb)*10, 256
cov1 = np.matmul(w1, w1.conj().T)
cov2 = np.matmul(w2, w2.conj().T)

window_len = sig_len
for sinr_db in tqdm(np.arange(-6,7,1.5)):
    for tau_s in range(ns):
        for tau_b in range(nb):
            Css = cov1[tau_s:tau_s+window_len, tau_s:tau_s+window_len]
            Cbb = cov2[tau_b:tau_b+window_len, tau_b:tau_b+window_len]

            sinr = 10**(sinr_db/10)
            scaled_Cbb = Cbb * 1/sinr
            Cyy = Css + scaled_Cbb + 0.01*np.eye(window_len, dtype=complex)
            Csy = Css.copy()

            U,S,Vh = np.linalg.svd(Cyy,hermitian=True)
            sthr_idx = np.linalg.matrix_rank(Cyy) + 1
            Cyy_inv = np.matmul(U[:,:sthr_idx], np.matmul(np.diag(1.0/(S[:sthr_idx])), U[:,:sthr_idx].conj().T))

            log_det_Cyy = np.sum(np.log(S[:sthr_idx]))
            
            W = np.matmul(Csy, Cyy_inv)
            Ce = Css - np.matmul(W, Css.conj().T)
            pickle.dump((Cyy_inv, log_det_Cyy, W, Ce), open(f'stats/cyclo_param/filters_param_taus{tau_s}_taub{tau_b}_sinr{sinr_db:.01f}.pickle','wb'))
