#! /usr/bin/env python
import sys
import numpy as np
from scipy.special import polygamma

# Calculate standardised partial complementary product, PCP
def PCP(ps,M=100000,nmax=None,min_alpha=0,seed=0,try_lookup=True):
    ps = np.sort(ps)
    if ps[0] == 0:
        return(0)
    n=len(ps)
    if nmax is None or nmax > n:
        nmax = n
    if ps[-1]==1:
        try:
            nmax=ps.index(1)
        except:
            nmax=np.where(ps==1)[0][0]
        if nmax==0:
            return(1)
    nmin = min(np.searchsorted(ps,min_alpha/float(n),side='right'),nmax-1)
    k = np.arange(1,nmax+1)
    mean = k-(n-k)*(polygamma(0,n+1)-polygamma(0,n-k+1))
    sd = np.sqrt(np.abs(k-(n-k)**2*(polygamma(1,n+1)-polygamma(1,n-k+1))-2*(n-k)*(polygamma(0,n+1)-polygamma(0,n-k+1))))
    z = min((-np.cumsum(np.log(1-ps[:nmax]))[nmin:]-mean[nmin:])/sd[nmin:])
    if try_lookup and min_alpha in [0,1]:
        try:
            return(look_up_distribution(z,n,'standardised_complementary_product'+('2' if min_alpha==1 else ''),nrows=M))
        except:
            None
    ctr = 0
    rng = np.random.default_rng(seed)
    for _ in range(M):
        x = rng.exponential(size=nmax+1)
        scaler = sum(x) + 0 if n==nmax else rng.gamma(shape=n-nmax)
        ps_ = np.cumsum(x[:nmax])/scaler
        nmin_ = min(np.searchsorted(ps_,min_alpha/float(n),side='right'),nmax-1)
        ctr += min((-np.cumsum(np.log(1-ps_))[nmin_:]-mean[nmin_:])/sd[nmin_:]) < z
    return(ctr/float(M))

# Calculate PCP+
def PCP2(ps,M=100000,nmax=None,seed=0,try_lookup=False):#look up for PCP2 not supported
    return(PCP(ps,M,nmax,1,seed,try_lookup))

# Attempt obtaining null distribution from the web
def look_up_distribution(z,n,method_name,nrows=10000):
    import urllib.request
    import pandas as pd
    scores=pd.read_csv(urllib.request.urlopen("http://null-distributions.ma.ic.ac.uk/"+str(len(p_values))+".txt"), sep='\t',nrows=nrows)[method_name]
    return(sum(scores<z)/float(len(scores)))

p_values = [0.1,.15,.2]
if not sys.stdin.isatty(): # Obtain p-values from command line
    p_values = list(map(float,sys.stdin.readline().split()))

print(PCP(p_values,50000,min_alpha=0 if len(sys.argv)<2 else float(sys.argv[1])))
