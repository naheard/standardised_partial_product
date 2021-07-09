#! /usr/bin/env python
import sys, re
import numpy as np
from scipy.special import polygamma

# Calculate standardised partial sum, PS
def PS(ps,M=100000,nmax=None,min_alpha=0,seed=0,try_lookup=True):
    ps = np.sort(ps)
    if ps[0] == 0:
        return(0)
    n=len(ps)
    if nmax is None or nmax > n:
        nmax = n
    nmin = min(np.searchsorted(ps,min_alpha/float(n),side='right'),nmax-1)
    k = np.arange(1,nmax+1)
    mean_sumunif = k/2.0
    mean_beta = (k+1)/(n+1.0)
    var_sumunif = k/12.0
    var_beta = (k+1)/(n+1.0)*(n-k)/((n+1.0)*(n+2.0))
    mean =mean_sumunif*mean_beta
    sd = np.sqrt(var_beta*mean_sumunif**2+var_sumunif*mean_beta**2+var_sumunif*var_beta)
    z = min((np.cumsum(ps[:nmax])[nmin:]-mean[nmin:])/sd[nmin:])
    if try_lookup and min_alpha in [0,1]:
        try:
            return(look_up_distribution(z,n,'standardised_sum'+('2' if min_alpha==1 else ''),nrows=M))
        except:
            None
    ctr = 0
    rng = np.random.default_rng(seed)
    for _ in range(M):
        x = rng.exponential(size=nmax+1)
        scaler = sum(x) + 0 if n==nmax else rng.gamma(shape=n-nmax)
        ps_ = np.cumsum(x[:nmax])/scaler
        nmin_ = min(np.searchsorted(ps_,min_alpha/float(n),side='right'),nmax-1)
        ctr += min((np.cumsum(ps_[:nmax])[nmin_:]-mean[nmin_:])/sd[nmin_:]) < z
    return(ctr/float(M))

# Calculate PS+
def PS2(ps,M=100000,nmax=None,seed=0,try_lookup=False):#look up for PS2 not supported
    return(PS(ps,M,nmax,1,seed,try_lookup))

# Attempt obtaining null distribution from the web
def look_up_distribution(z,n,method_name,nrows=10000):
    import urllib.request
    import pandas as pd
    scores=pd.read_csv(urllib.request.urlopen("http://null-distributions.ma.ic.ac.uk/"+str(len(p_values))+".txt"), sep='\t',nrows=nrows)[method_name]
    return(sum(scores<z)/float(len(scores)))

p_values = [0.1,.15,.2]
if not sys.stdin.isatty(): # Obtain p-values from command line
    p_values = list(map(float,re.split(', | |,',sys.stdin.readline().strip())))

print(PS(p_values,50000,min_alpha=0 if len(sys.argv)<2 else float(sys.argv[1])))
