#! /usr/bin/env python
import sys
import numpy as np
import operator
from scipy.special import polygamma,betainc,digamma,binom
from scipy.stats import norm,chi2,t,gamma,f,cauchy,poisson
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from math import factorial
from heapq import merge
from collections import Counter
from bisect import bisect
import pandas as pd

null_distributions_url="http://null-distributions.ma.ic.ac.uk/"

def draw_ordered_uniforms(n,k=0):
    if n==0:
        return(np.empty(shape=0))
    if k==0 or k>n:
        k=n
#    x= sorted(np.random.uniform(size=k))
#    if n>k:
#        x=x*np.random.beta(nmax+1,n-k)
#    return(x)
    x=np.random.exponential(size=k+1)
    scaler=sum(x)
    if n>k:
        scaler+=np.random.gamma(shape=n-k)
    return(np.cumsum(x[:k])/scaler)

def monte_carlo_standardised_product(ps,n=0,nmax=-1,complementary=False):
    global raw_stats,building_null
    if not complementary and ps[0]==0:
        return(-sys.float_info.max,1)
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    ps=ps[0:nmax]
    if not complementary:
        raw_stats=np.cumsum(np.log(ps))
    else:
        raw_stats=np.cumsum(np.log(1-ps))

    if not building_null:
        ranks=[bisect(mcsp_stores[:,j],raw_stats[j])/float(mcsp_stores.shape[0]) for j in range(nmax)]
        z_argmin,z_min=min(enumerate(ranks), key=operator.itemgetter(1))
    else:
        z_min,z_argmin=-1,0
    return(z_min,z_argmin+1)

def partial_product(ps,n=0,nmax=-1,complementary=False):
    if not comlementary and ps[0]==0:
        return(-sys.float_info.max,1)
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    ps=ps[0:nmax]
    if not complementary:
        lps=np.cumsum(np.log(ps))
        z=[partial_product_SF(-lps[k-1],n,k) for k in range(1,nmax+1)]
    else:
        lps=np.cumsum(np.log(1-ps))
        z=[partial_complementary_product_CDF(-lps[k-1],n,k) for k in range(1,nmax+1)]
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def partial_sum(ps,n=0,nmax=-1):
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    ps=ps[0:nmax]
    sps=np.cumsum(ps)
    z=[partial_sum_CDF(sps[k-1],n,k) for k in range(1,nmax+1)]
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def normalised_product(ps,n=0,nmax=-1):
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    ps=ps[0:nmax]
    one_above=np.concatenate((ps[1:nmax],[1 if nmax==n else ps[nmax]]))
    normalisers=np.log(one_above)*(np.arange(nmax)+1)
    z=chi2.sf(-2*(np.cumsum(np.log(ps))-normalisers),2*(np.arange(nmax)+1))
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def normalised_complementary_product(ps,n=0,nmax=-1):
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    ps=ps[0:nmax]
    one_above=np.concatenate((ps[1:nmax],[1 if nmax==n else ps[nmax]]))
    normalisers=np.log(one_above)*(np.arange(nmax)+1)
    z=[chi2.cdf(-2*(np.sum(np.log(one_above[i]-ps[0:(i+1)]))-normalisers[i]),2*(i+1)) for i in np.arange(nmax)]
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def standardised_product(ps,n=0,nmax=-1,min_alpha=0,first=True):
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    nmin=min(np.searchsorted(ps,min_alpha/float(n),side='right'),nmax-1)
    if first:
        sps=np.cumsum(np.log(ps[:nmax]))[nmin:]
        mean,sd=mu_prod[nmin:nmax],sigma_prod[nmin:nmax]
    else:
        sps=np.cumsum(np.log(ps[n-nmax:][::-1]))
        mean,sd=-mu_comp_prod[:nmax],sigma_comp_prod[:nmax]
    z=(sps-mean)/sd
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def standardised_product2(ps,n=0,nmax=-1):
    return(standardised_product(ps,n,nmax,1))

def asymptotic_standardised_product(ps,n=0,nmax=-1,min_alpha=0):
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>=n:
        nmax=n-1
    nmin=min(np.searchsorted(ps,min_alpha/float(n),side='right'),nmax-1)
    sps=np.exp((np.cumsum(np.log(ps[:nmax]))/np.arange(1,nmax+1))[nmin:])
    mean,sd=mu_asymp_prod[nmin:nmax],sigma_asymp_prod[nmin:nmax]
    z=(sps-mean)/sd
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def standardised_complementary_product(ps,n=0,nmax=-1):
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    if ps[-1]==1:
        try:
            nmax=ps.index(1)
        except:
            nmax=np.where(ps==1)[0][0]
        if nmax==0:
            return(sys.float_info.max,1)
    ps=np.array(ps[:nmax])
    z=(-np.cumsum(np.log(1-ps))-mu_comp_prod[:nmax])/sigma_comp_prod[:nmax]
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def standardised_product_subset_k(ps,n=0,nmax=-1,k=0):
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    ps=ps[0:nmax]
    cumps=np.insert(np.cumsum(np.log(ps)),0,0)
    if k==0:
        k=true_k
    try:
        z=((cumps[k:nmax+1]-cumps[0:nmax-k+1])-mu_subset_prod[0:(nmax-k+1),k])/sigma_subset_prod[0:(nmax-k),k]
    except:
        z=((cumps[k:nmax+1]-cumps[0:nmax-k+1])-mu_subset_prod[0:(nmax-k+1)])/sigma_subset_prod[0:(nmax-k+1)]
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def standardised_product_subset(ps,n=0,nmax=-1):
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    ps=ps[0:nmax]
    cumps=np.insert(np.cumsum(np.log(ps)),0,0)
    z_min=float("inf")
    for k in np.arange(1,nmax+1):
        z=((cumps[k:nmax+1]-cumps[0:nmax-k+1])-mu_subset_prod[0:(nmax-k+1),k])/sigma_subset_prod[0:(nmax-k+1),k]
        z_argmin_k,z_min_k=min(enumerate(z), key=operator.itemgetter(1))
        if z_min_k<z_min:
            z_min=z_min_k
            z_argmin=np.array([k,z_argmin_k])[0]
    return(z_min,z_argmin+1)

def standardised_logit(ps,n=0,nmax=-1):
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    if ps[-1]==1:
        try:
            nmax=ps.index(1)
        except:
            nmax=np.where(ps==1)[0][0]
        if nmax==0:
            return(sys.float_info.max,1)
    ps=np.array(ps[:nmax])
    z=(np.cumsum(np.log(ps)-np.log(1-ps))-mu_logit[:nmax])/sigma_logit[:nmax]
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def standardised_sum(ps,n=0,nmax=-1):
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    ps=ps[0:min(n,nmax)]
    z=(np.cumsum(ps)-mu_sum)/sigma_sum
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def monte_carlo_logit_expected_square(n,nmax=0,M=100000):
    if nmax<0 or nmax>n:
        nmax=n
    ssq=np.zeros(nmax)
    for _ in range(M):
        ps=draw_ordered_uniforms(n,nmax)
        x=np.cumsum(np.log(ps)-np.log(1-ps))
        ssq+=x**2
    return(ssq/M)

def higher_criticism(ps,n=0,nmax=-1,min_alpha=0):
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    nmin=min(np.searchsorted(ps,min_alpha/float(n),side='right'),nmax-1)
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if ps[nmax-1]==1:
        try:
            nmax=ps.index(1)
        except:
            nmax=np.where(ps==1)[0][0]
        if nmax<=nmin:
            return(1,1)
    ps=np.array(ps[nmin:nmax])
    sqrt_n=np.sqrt(n)
    z=(sqrt_n*ps-np.arange(nmin+1,nmax+1)/sqrt_n)/np.sqrt(ps*(1-ps))
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def higher_criticism2(ps,n=0,nmax=-1):
    return(higher_criticism(ps,n,nmax,1))

def berk_jones(ps,n=0,nmax=-1,min_alpha=0):
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n:
        nmax=n
    nmin=min(np.searchsorted(ps,min_alpha/float(n),side='right'),nmax-1)
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if ps[nmax-1]==1:
        try:
            nmax=ps.index(1)
        except:
            nmax=np.where(ps==1)[0][0]
        if nmax<=nmin:
            return(1,1)
    ps=np.array(ps[nmin:nmax])
    sqrt_n=np.sqrt(n)
    k_=np.arange(nmin+1,nmax+1)
#    with np.errstate(divide='ignore', invalid='ignore'):
#        z=-np.sqrt(-2*k_*np.log(n*ps/k_)-(n-k_)*np.log(n*(1-ps)/(n-k_)))
    z=-np.sqrt(np.maximum(-2*k_*np.log(n*ps/k_)+_xlogy(n-k_,(n-k_)/(n*(1-ps))),0))
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def _xlogy_(x,y):
    if x==0:
        return(0)
    return(x*np.log(y))

def _xlogy(xx,yy):
    return([_xlogy_(x,y) for x,y in zip(xx,yy)])

def modified_berk_jones(ps,n=0,nmax=-1,min_alpha=0):
    if n==0:
        n=len(ps)
    if nmax<0 or nmax>n/2:
        nmax=int(n/2)
    nmin=min(np.searchsorted(ps,min_alpha/float(n),side='right'),nmax-1)
    if ps[0]==0:
        return(-sys.float_info.max,1)
    if ps[nmax-1]==1:
        try:
            nmax=ps.index(1)
        except:
            nmax=np.where(ps==1)[0][0]
        if nmax<=nmin:
            return(1,1)
    k_=np.array([i for i in np.arange(nmin+1,nmax+1) if n*ps[i-1]<i])
    ps=np.array([ps[i-1] for i in k_])
    sqrt_n=np.sqrt(n)
    z=-np.sqrt(-2*(k_*np.log(n*ps/k_)+k_-n*ps))
    try:
        z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    except:
        z_argmin,z_min=0,0

    return(z_min,z_argmin+1)

def standardised_order_statistics(ps,n=0,nmax=-1):
    if n==0:
        n=len(ps)
    if nmax<=0 or nmax>n:
        nmax=n
    ps=ps[0:min(n,nmax)]
    z=(ps-mu_os)/sigma_os
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def standardised_stouffer(ps,n=0,nmax=-1):
    if n==0:
        n=len(ps)
    if ps[0]==0:
        return(0,n)
    if ps[-1]==1:
        return(1,n)
    if nmax<=0 or nmax>n:
        nmax=n
    if nmax<n:
        ps=np.array(ps[:nmax])
    z=(np.cumsum(norm.ppf(ps))-mu_stouffer)/sd_stouffer
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def simes(ps,n=0,nmax=-1):
    if n==0:
        n=len(ps)
    if nmax<=0 or nmax>n:
        nmax=n
    ps=ps[0:min(n,nmax)]
    z=ps/np.arange(1,nmax+1)*float(nmax)
    z_argmin,z_min=min(enumerate(z), key=operator.itemgetter(1))
    if nmax<n:
        if nmax==1:
            z_min=betainc(1,n+1,z_min)
        else:
            z_min=betainc(nmax,n+1-nmax,z_min)+n/float(nmax)*z_min*(1-betainc(nmax-1,n+1-nmax,z_min))
    return(z_min,z_argmin+1)

def fisher(ps,n=0,nmax=""):
    if n==0:
        n=len(ps)
    if ps[0]==0:
        return(0,n)
    return(chi2.sf(-2*sum(np.log(ps)),2*n),n)

def pearson(ps,n=0,nmax=""):
    if n==0:
        n=len(ps)
    if ps[-1]==1:
        return(1,n)
    return(chi2.cdf(-2*sum(np.log([1-p for p in ps])),2*n),n)

def sum_method(ps,n=0,nmax=""):
    if n==0:
        n=len(ps)
    return(norm.cdf(np.sqrt(12.0/n)*(sum(ps)-.5*n)),n)

def beta(ps,n=0,nmax=""):
    return(1.0-(1.0-ps[0])**n,1)

def stouffer(ps,n=0,nmax=""):
    if n==0:
        n=len(ps)
    if ps[0]==0:
        return(0,n)
    if ps[-1]==1:
        return(1,n)
    return(norm.cdf(sum(norm.ppf(ps))/np.sqrt(n)),n)

def logistic(ps,n=0,nmax=""):
    if n==0:
        n=len(ps)
    if ps[0]==0:
        return(0,n)
    if ps[-1]==1:
        return(1,n)
    return norm.cdf(sum(np.log(ps)-np.log(1.0-ps))/(np.pi*np.sqrt(n/3.0))),n

def inverse_cauchy(ps,n=0,nmax=""):#Yaowu Liu, Harvard Sch of Pub Health
    if n==0:
        n=len(ps)
    if ps[0]==0:
        return(0,n)
    if ps[-1]==1:
        return(1,n)
    return(cauchy.sf(sum(cauchy.isf(ps))/float(n)),n)

def truncated_product(ps,n=0,nmax=-1,tau=0.05):
    if n==0:
        n=len(ps)
    if ps[0]==0:
        return(0,n)
    if ps[0]>tau:
        return(1,n)
    if nmax<=0 or nmax>n:
        nmax=n
    if nmax<n:
        ps=np.array(ps[:nmax])
    z_argmin=np.searchsorted(ps,tau,side='right')
    z=-2*sum(np.log(ps[:z_argmin]))
    ltau=-2*np.log(tau)
    Fz=stats.binom.pmf(0,nmax,tau)+sum([tmp_cdf_contribution(z,tau,ltau,j+1,nmax) for j in range(nmax)])
    return(1-Fz,z_argmin)

def tmp_cdf_contribution(z,tau,ltau,j,nmax):
    return(0 if z<ltau*j else stats.binom.pmf(j,nmax,tau)*chi2.cdf(z-ltau*j,2*j))

def ensemble(pvals):
    return sum(pvals)
    z_argmin,z_min=min(enumerate(pvals), key=operator.itemgetter(1))
    return(z_min,z_argmin+1)

def partial_product_CDF(y,n,k=1):#cdf for sum of -log of k smallest p-values from n
    if k==1:
        cdf=(1-np.exp(-y))**n
    elif k==n:
        cdf=chi2.cdf(2*y,2*n)
    else:
        nck=binom(n,k)
        terms=[nck*(k/float(i))**(k-1)*binom(n-k,i)*(-1)**(k+i)*gamma_exp_terms(i,y,k) for i in range(1,n-k+1)]
        cdf=sum(terms)
        if k==14:
            i=1
            print([(-i/float(k))**j*poisson.sf(j,y) for j in range(k)], k/float(k+i)*(np.exp(-(1+i/float(k))*y)-1), gamma_exp_terms(i,y,k))
            a=k/float(k+i)*(np.exp(-(k+i)*y/float(k))-1)
            for j in range(k):
                a+=(-i/float(k))**j*poisson.sf(j,y)
            print(a)
    return(cdf)

def partial_complementary_product_CDF(y,n,k=1):#cdf for sum of -log of k largest p-values from n
    if k==1:
        cdf=1-np.exp(-y*n)
    elif k==n:
        cdf=chi2.cdf(2*y,2*n)
    else:
        cdf=factorial(n)/float(factorial(n-k))*sum([(i/float((n-k)))**(k-1)*(-1)**(k-i)/float(factorial(i-1)*factorial(k-i)*(n-k+i))*(1-np.exp(-(n-k+i)*y/float(i))) for i in range(1,k+1)])
    return(cdf)

def partial_sum_CDF(y,n,k=1):#cdf for sum of of k smallest p-values from n
    if y>=n:
        return(1)
    if k==1:
        cdf=1-(1-y)**n
    elif k==n:
        cdf=sum([(-1)**k*(y-k)**n/float(factorial(n-k)*factorial(k)) for k in range(0,int(y)+1)])
    else:
        cdf=factorial(n)/factorial(n-k-1)*sum([(-1)**i/float(factorial(i)*factorial(k-i))*sum([(y-i)**(k-j)*i**j*(1 if y>=i else (1-(1-y/float(i))**(j+n-k)))/float(factorial(j)*factorial(k-j)*(j+n-k)) for j in range(0,k+1)]) for i in range(0,k+1)])
    return(cdf)

def partial_product_SF(y,n,k=1):#cdf for sum of -log of k smallest p-values from n
    if k==1:
        sf=1-(1-np.exp(-y))**n
    elif k==n:
        sf=chi2.sf(2*y,2*n)
    else:
        y/=float(k)
        const=np.exp(-y*k)*factorial(n)/float(factorial(k))
        sf=0
        for j in range(k):
            term_j=const*k**j
            for i in range(n-k):
                l_sum=sum([(-1)**(l+i)*y**(j-l)/float(factorial(j-l)*(i+1)**(l+1)) for l in range(j+1)])
                sf+=term_j*(l_sum-np.exp(-(i+1)*y)*(-1)**(j+i)/float((i+1)**(j+1)))/(factorial(i)*factorial(n-k-1-i))
###        sf=sum([k**j*sum([(sum([(-1)**(l+i)*y**(j-l)/float(factorial(j-l)*(i+1)**(l+1)) for l in range(j+1)])-np.exp(-(i+1)*y)*(-1)**(j+i)/float((i+1)**(j+1)))/(factorial(i)*factorial(n-k-1-i)) for i in range(n-k)]) for j in range(k)])
        sf+=betainc(k+1,n-k,np.exp(-y))
    return(sf)

def gamma_exp_terms(i,y,k):
  return(sum([(-i/float(k))**j*poisson.sf(j,y) for j in range(k)]) + k/float(k+i)*(np.exp(-(k+i)*y/float(k))-1))

def make_parameter_vectors(n,nmax,methods=[standardised_product]):
    global mu_prod,sigma_prod,mu_sum,sigma_sum,mu_comp_prod,sigma_comp_prod,mu_subset_prod,sigma_subset_prod,mu_logit,sigma_logit,mu_os,sigma_os,true_k,mu_stouffer,sd_stouffer,mu_asymp_prod,sigma_asymp_prod
    k=np.arange(1,nmax+1)
    if standardised_product in methods or standardised_complementary_product in methods:
        mu_prod=-k*(1+polygamma(0,n+1)-polygamma(0,k+1))#-k*(1+np.log(n+1)-np.log(k+1))
        sigma_prod=k*np.sqrt(1.0/k+polygamma(1,k+1)-polygamma(1,n+1))
#        mu_comp_prod=n-(n-k)*(1-polygamma(0,n-k+1)+polygamma(0,n+1))
        mu_comp_prod=k-(n-k)*(polygamma(0,n+1)-polygamma(0,n-k+1))
        sigma_comp_prod=np.sqrt(np.abs(k-(n-k)**2*(polygamma(1,n+1)-polygamma(1,n-k+1))-2*(n-k)*(polygamma(0,n+1)-polygamma(0,n-k+1))))
        sigma_comp_prod[0]=1.0/n
#    mu_asymp_prod=k*(1-np.log(k/float(n)))  #(k+1)/float(n+1)
#    sigma_asymp_prod=np.sqrt(2*k)  #np.sqrt((k+1)*(n-k)/float((n+1)**2*(n+2)))
    if standardised_logit in methods:
        mu_logit=-n*polygamma(0,n+1)+(n-k)*polygamma(0,n-k+1)+k*polygamma(0,k+1)
        if nmax==1:
            sigma_logit=np.array([2*polygamma(1,1)])
        else:
            sigma_logit=np.sqrt(monte_carlo_logit_expected_square(n,nmax)-mu_logit**2)
    if standardised_complementary_product in methods:
        failures=np.where(np.isnan(sigma_comp_prod))[0]
        if len(failures):
            last_failure=failures[-1]+1
            for kval in np.arange(1,last_failure+1):
                sigma_comp_prod[kval-1]=np.sqrt(sum([(float(i)/(n-kval+i))**2 for i in np.arange(1,kval+1)]))
    if standardised_sum in methods:
        mean_sumunif=k/2.0
        mean_beta=(k+1)/(n+1.0)
        var_sumunif=k/12.0
        var_beta=(k+1)/(n+1.0)*(n-k)/((n+1.0)*(n+2.0)) #(k+1)*(n-k)/((n+1.0)**2*(n+2.0))
        mu_sum=mean_sumunif*mean_beta
        sigma_sum=np.sqrt(var_beta*mean_sumunif**2+var_sumunif*mean_beta**2+var_sumunif*var_beta)
    if standardised_order_statistics in methods:
        mu_os=k/(n+1.0)
        sigma_os=np.sqrt(k/(n+1.0)*(n-k+1)/((n+1.0)*(n+2.0)))
    if standardised_product_subset in methods:
        mu_subset_prod,sigma_subset_prod=np.zeros((nmax+1,nmax+1)),np.zeros((nmax+1,nmax+1))
        get_subset_moments(n,nmax)
    elif standardised_product_subset_k in methods:
        mu_subset_prod,sigma_subset_prod=np.zeros((nmax+1)),np.zeros((nmax+1))
        get_subset_moments(n,nmax,true_k)

    if standardised_stouffer in methods:
        mu_stouffer,sd_stouffer=np.zeros(n),np.zeros(n)
        mu_stouffer[-1]=0
        sd_stouffer[-1]=np.sqrt(n)
        if n>2:
            try:
                f=get_file("normal_os_moments_"+str(n)+".txt")
                mu_stouffer[:(n-1)]=map(float,f.readline().strip().split()[:(n-1)])
                sd_stouffer[:(n-1)]=map(float,f.readline().strip().split()[:(n-1)])
                f.close()
                if n==3:
                    mu_stouffer[0]=mu_stouffer[1]=-1.5/np.sqrt(np.pi)
            except:
                methods.remove(standardised_stouffer)
        elif n==2:
            mu_stouffer[0]=-1/np.sqrt(np.pi)
            sd_stouffer[0]=np.sqrt(1-1/np.pi)

def get_subset_moments_k(n,nmax,k=1):
    for l in np.arange(0,nmax+1-k):
        mu_subset_prod[l],sigma_subset_prod[l]=get_subset_moments_lk(l,k,n,nmax)

def get_subset_moments(n,nmax):
    for l in np.arange(0,nmax+1):
        for k in np.arange(1,nmax+1-l):
            mu_subset_prod[l,k],sigma_subset_prod[l,k]=get_subset_moments_lk(l,k,n,nmax)

def get_subset_moments_lk(l,k,n,nmax):#for sum_{i=l+1}^{l+k} log p_(i)
    mean=mu_prod[l+k-1]- (0 if l==0 else mu_prod[l-1])
#    sd=np.sqrt(k-l**2*(polygamma(1,l+k+1)-polygamma(1,l+1))-2*l*(polygamma(0,l+k+1)-polygamma(0,l+1))+k**2*(polygamma(1,l+k+1)-polygamma(1,n+1)))
    sd=np.sqrt(k-(l**2-k**2)*polygamma(1,l+k+1)+l**2*polygamma(1,l+1)-k**2*polygamma(1,n+1)-2*l*(polygamma(0,l+k+1)-polygamma(0,l+1)))
    return(mean,sd)

def get_null_distns(n,nmax=0,M=10000,methods=[standardised_product],calculate_cdfs=True,seed=None,output_filename=None,write_mcsp_to_file=False):
    global mcsp_stores,raw_stats,building_null
    if nmax>n or nmax==0:
        nmax=n
    scores={}
    indices={}
    outputstream=sys.stdout if output_filename==None else open(output_filename,"w")
    if calculate_cdfs:
        for m in methods:
            scores[m]=[0]*M
    if seed is not None:
        np.random.seed(seed)
    if not calculate_cdfs:
        outputstream.write("\t".join([m.__name__ for m in methods])+"\n")
    if monte_carlo_standardised_product in methods:
        mcsp_filename="mcsp_"+str(n)+"_.npy"
        try:
            mcsp_stores=np.load(mcsp_filename)
            mcsp_min_ranks=create_mcsp_ranks()
            building_null=False
        except:
            mcsp_stores=np.empty([M,nmax])
            building_null=True
    for i in range(M):
        ps=draw_ordered_uniforms(n,nmax)
        for m in methods:
            score_m_i,index_m_i=m(ps,n,nmax)
            if calculate_cdfs:
                scores[m][i]=score_m_i
            else:
                scores[m]=score_m_i
                indices[m]=index_m_i
            if m==monte_carlo_standardised_product and building_null:
                mcsp_stores[i,:]=raw_stats
        if not calculate_cdfs:
#            outputstream.write("\t".join(str(indices[m]) for m in methods)+"\t")
            outputstream.write("\t".join(str(scores[m]) for m in methods)+"\n")
            outputstream.flush()

    if monte_carlo_standardised_product in methods and building_null:
        building_null=False
        if write_mcsp_to_file:
            np.save(mcsp_filename,mcsp_stores)
#            np.savetxt("mcsp_"+str(n)+"_.txt",mcsp_stores,delimiter=',',comments="")
        mcsp_min_ranks=create_mcsp_ranks()
        if calculate_cdfs:
            scores[monte_carlo_standardised_product]=mcsp_min_ranks[:]

    if not calculate_cdfs:
        return(0)
    for m in methods:
        scores[m]=ECDF(scores[m])
    return(scores)

def create_mcsp_ranks():
    global mcsp_stores
    from scipy.stats import rankdata
    M=mcsp_stores.shape[0]
    mcsp_ranks,mcsp_min_ranks=np.empty([M,nmax]),np.empty(M)
    for j in range(nmax):
        mcsp_ranks[:,j]=rankdata(mcsp_stores[:,j])
        mcsp_stores[:,j]=sorted(mcsp_stores[:,j])
    for i in range(M):
        mcsp_min_ranks[i]=min(mcsp_ranks[i,])/float(M)
    return(mcsp_min_ranks)

def make_3d_plot_points(num_points=20,methods=[standardised_product],sep=" ",maxval=1):
    sys.stdout.write(sep.join(["p_i","p_j"]+[m.__name__ for m in methods])+"\n")
    for i in range(num_points):
        pi=float(i)/(num_points-1)*maxval
        for j in range(num_points):
            pj=float(j)/(num_points-1)*maxval
            sys.stdout.write(sep.join(map(str,[pj,pi]+[null_distns[method](method(sorted([pi,pj]))[0]) if method in null_distns else method(sorted([pi,pj]))[0] for method in methods]))+"\n")
        sys.stdout.write("\n")

def make_2d_plot_points(num_points=20,ps=[],methods=[standardised_product],sep=" ",maxval=1):
    sys.stdout.write(sep.join(["p_i"]+[m.__name__ for m in methods])+"\n")
    for i in range(num_points):
        pi=float(i)/(num_points-1)*maxval
        psi=np.sort([pi]+list(ps))
        sys.stdout.write(sep.join(map(str,[pi]+[null_distns[method](method(sorted(psi))[0]) if method in null_distns else method(sorted(psi))[0] for method in methods]))+"\n")

def get_null_distns_from_file(filename,methods=[standardised_product]):
    global mcsp_stores,building_null
    scores={}
    for m in methods:
        scores[m]=[]
    if monte_carlo_standardised_product in methods:
        try:
            mcsp_filename="mcsp_"+str(n)+"_.npy"
            mcsp_stores=np.load(mcsp_filename)
            mcsp_min_ranks=create_mcsp_ranks()
            scores[monte_carlo_standardised_product]=mcsp_min_ranks[:]
            building_null=False
#            if len(methods)==1:
#                scores[monte_carlo_standardised_product]=ECDF(scores[monte_carlo_standardised_product])
#                return(scores)
        except:
            return(None)
    f=get_file(filename)
    df=pd.read_csv(f, sep='\t')
    first_line=list(df.columns)
#    f=get_file(filename)
#    try:
#        first_line = f.readline()
#        try:
#            first_line = first_line.decode('ascii').split()
#        except:
#            first_line = first_line.split()
#    except:
#        return(None)
    method_columns={m: (first_line.index(m.__name__) if m.__name__ in first_line else -1) for m in methods}# if m is not monte_carlo_standardised_product}
    methods_present=[m for m in methods if method_columns[m]!=-1]
    if monte_carlo_standardised_product in methods_present:
        scores[monte_carlo_standardised_product]=[]
#    for line in f:
#        try:
#            s=list(map(float,line.strip().split()))
#            for m in methods_present:
#                scores[m]+=[s[method_columns[m]]]
#        except:
#            None
#    f.close()

    for m in methods:
        try:
            scores[m]=ECDF(df[m.__name__])
        except:
            scores[m]=ECDF(scores[m])

    include_ensemble=False
    if include_ensemble:
        scores[ensemble]=get_ensemble_distn(filename,scores,methods)
    return(scores)

def get_ensemble_distn(filename,scores,methods):
    f=get_file(filename)
    ensemble_scores=[]
    for line in f:
        s=[float(x) for x in line.strip().split()]
        pvals_i=[]
        for i in range(3):
            pvals_i+=[scores[methods[i]](s[i])]
        ensemble_scores+=[ensemble(pvals_i)]
    f.close()
    return(ECDF(ensemble_scores))

def get_file(filename):
    try:
        return(open(filename, "r"))
    except:
        try:
            import urllib.request
            if filename.startswith("null_distns_"):
                filename=filename.split("_",2)[-1]
            url_file=urllib.request.urlopen(null_distributions_url+filename)
            return(url_file)
        except:
#            sys.stderr.write("Null distribution file " + filename + " not found.\n")
            return(None)

def get_k_epsilon_mu(n,beta,dr):
    epsilon=n**(-beta)
    k=int(n*epsilon)
    r=0
    if dr>0:
        if beta<.5:
            r=dr
        elif beta<.75:
            r=dr+beta-.5
        else:
            r=dr+(1-np.sqrt(1-beta))**2
    mu=np.sqrt(2*r*np.log(n))
    return(k,epsilon,mu)

def h1_draw(n,k,mu):
    null_p_values=draw_ordered_uniforms(n-k)
    if alternative_distn=='normal':
        p1s=norm.sf(norm.isf(draw_ordered_uniforms(k),loc=mu,scale=1))#,scale=.05))#use smaller scale to favour HC
#        p1s=norm.sf(norm.isf(draw_ordered_uniforms(k),loc=mu,scale=.0001))#.0001))#,scale=1/(1.0+mu)**2))
#        null_p_values=norm.sf(norm.isf(draw_ordered_uniforms(n-k),loc=-mu*.1,scale=1))
#        p1s=norm.sf(norm.isf(draw_ordered_uniforms(1).repeat(k),loc=mu))
#        p1s=sorted(norm.cdf(np.random.normal(loc=-mu,size=k)))
    elif alternative_distn=='normal_hc':
        p1s=norm.sf(norm.isf(draw_ordered_uniforms(k),loc=mu,scale=.05))#use smaller scale to favour HC
    elif alternative_distn=='cor_normal':
        rh=.995
        w=(rh-np.sqrt(rh*(1-rh)))/(2*rh-1.0)
        p1s=sorted(norm.sf((w*np.random.normal(loc=mu/w)+(1-w)*np.random.normal(size=k))/(w**2+(1-w)**2)))
    elif alternative_distn=='normal_var':
        mu*=.5
        p1s=2*norm.sf(norm.isf(.5*draw_ordered_uniforms(k),scale=1.0+mu))
    elif alternative_distn=='gamma':
        p1s=gamma.sf(gamma.isf(draw_ordered_uniforms(k),10,scale=1.0+.5*mu),10)
    elif alternative_distn=='exp':
        mu*=20
        p1s=-np.log(1-(1-np.exp(-mu))*draw_ordered_uniforms(k))/mu
    elif alternative_distn=='power':
        inv_rate=round(1.0+mu)
        p1s=draw_ordered_uniforms(k)**inv_rate
    elif alternative_distn=='comp_power':
        mu*=.5#50
        inv_rate=1.0/(1.0+mu)
        inv_rate=2/3.0
        p1s=1-(1-draw_ordered_uniforms(k))**inv_rate
    elif alternative_distn=='pw_uniform':
        mu*=.1*.1#*k
        inv_rate=k/(n*(1.0+mu))
#        inv_rate=1.0/(1.0+mu)
        p1s=draw_ordered_uniforms(k)*inv_rate
#        p1s=draw_ordered_uniforms(1).repeat(k)*inv_rate
#        null_p_values=inv_rate+(1-inv_rate)*null_p_values
    elif alternative_distn=='pw_uniform2':
        mu*=.1*.1*k
        inv_rate=k/(n*(1.0+mu))
        p1s=draw_ordered_uniforms(k)
        p1s*=inv_rate/p1s[k-1]
    elif alternative_distn=='point_mass':
        mu*=.1*3
        inv_rate=k/(n*(1.0+mu*5))
        p1s=np.full(k,inv_rate)
#        null_p_values=np.full(n-k,1)
    elif alternative_distn=='boundary':
#        inv_rate=k/(n*(1.0+.75*mu))
        inv_rate=np.random.beta(k,(n-k+1)*(1.0+.5*mu))
##        inv_rate=k/(n+1.0+.5*mu)*np.random.uniform()
        p1s=inv_rate*np.array([1]*k)#draw_ordered_uniforms(k)
##        p1s=np.append(draw_ordered_uniforms(k-1)*inv_rate,inv_rate)
        null_p_values=inv_rate+(1-inv_rate)*(null_p_values)
    elif alternative_distn=='linear': #f(p)=c+1-2cp, c\in(0,1], F(p)=p(1+c(1-p))
        mu*=1000
        c=mu/(1.0+mu)
        p1s=((c+1)-np.sqrt((c+1)**2-4*c*draw_ordered_uniforms(k)))/(2.0*c)
    elif alternative_distn=='t_norm':
        mu*=.1
        p1s=2*norm.sf(t.isf(.5*draw_ordered_uniforms(k),df=1.0/mu))
    elif alternative_distn=='beta':
#        p1s=draw_ordered_uniforms(k)
#        p1s*=np.random.beta(1/(1.0+mu*2),1)/p1s[k-1]
        mean,ss=(.95*k)/(n+1),10000
        p1s=np.sort(np.random.beta(mean*ss,(1-mean)*ss,size=k))
    elif alternative_distn=='logistic':
        inv_rate=np.exp(mu)
        p1s=draw_ordered_uniforms(k)
        p1s=1/(1+inv_rate*(1-p1s)/p1s)
    elif alternative_distn=='exp2':#H_0:x_i~Exp(1),H_1:x_i-mu~Exp(1)
        p1s=np.exp(-3*mu)*draw_ordered_uniforms(k)
    elif alternative_distn=='pareto':#H_0:x_i~Pareto(1),H_1:x_i~Pareto_mu(1)
        p1s=1-(1-draw_ordered_uniforms(k))**np.exp(-3*mu)
    elif alternative_distn=='F':
        df2_lo,df2_hi,df1_lo,df1_hi=1+mu,1,1,1
        sided=['left','right','two'][1]
        if sided=='right' or (sided=='two' and np.random.uniform()<0.5*2):
            p1s=f.sf(f.isf(draw_ordered_uniforms(k),df1_hi,df2_hi),df1_lo,df2_lo)
        else:
            p1s=f.cdf(f.isf(draw_ordered_uniforms(k)[::-1],df1_lo,df2_lo),df1_hi,df2_hi)
        if sided=='two':
            p1s=np.sort([2*min(p,1-p) for p in p1s])
    elif alternative_distn=='tippett_beta':
        p1s=[np.random.beta(k/(1.0+.5*mu),n-k+1)]*k
#        p1k=np.random.beta(k/(1.0+.5*mu),n-k+1)
#        p1s=np.append(draw_ordered_uniforms(k-1)*p1k,p1k)
        null_p_values=p1s[k-1]+(1-p1s[k-1])*null_p_values
    else:
        sys.stderr.write("unrecognised alternative distribution\n")
        exit(1)
    return(np.array(list(merge(p1s,null_p_values))))
#    return(sorted(np.concatenate((p1s,np.random.uniform(size=n-k)))))

def get_alternative_pvalue_distributions(n,nmax,epsilon,k,mu,random_k=True,M=1000,methods=[standardised_product],sort_output=False,alt_distn_filename="",index_pmf_filename="",auc_filename="",seed=None,append_zeros_and_ones=True):
    if nmax>n or nmax==0:
        nmax=n
    num_methods=len(methods)
    method_pvals={}
    calculate_index_pmfs=not index_pmf_filename==""
    method_index_pmfs={}
    for m in methods:
        if sort_output:
            method_pvals[m]=[0]*M
        if calculate_index_pmfs:
            method_index_pmfs[m]=Counter()
    if seed is not None:
        np.random.seed(seed)
    outputstream=sys.stdout if alt_distn_filename=="" else open(alt_distn_filename,"w",buffering=1)
    outputstream.write("\t".join([m.__name__ for m in methods])+"\n")
    for i in range(M):
        ps=h1_draw(n,k if not random_k else np.random.binomial(n,epsilon),mu)
        pvals_i=[]
        for j in range(num_methods):
            m=methods[j]
            stat,stat_index=m(ps,n,nmax)
            if calculate_index_pmfs:
                method_index_pmfs[m][stat_index]+=1
            if m in null_distns:
                pval_j=null_distns[m](stat)
            else:
                pval_j=stat
            pvals_i+=[pval_j]
            if sort_output:
                method_pvals[m][i]=pval_j
            else:
                if j<num_methods-1:
                    outputstream.write(str(pval_j)+"\t")
                else:
                    outputstream.write(str(pval_j)+"\n")
                if j==2 and ensemble in null_distns:
                    outputstream.write(str(null_distns[ensemble](ensemble(pvals_i)))+"\t")
#        if not sort_output:
#            outputstream.flush()

    if sort_output:
        try:
            from sklearn.metrics import auc
            auc_outputstream=sys.stderr if auc_filename=="" else open(auc_filename,"w")
            if auc_filename=="":
                auc_outputstream.write('AUC\n')
            for m in methods:
                method_pvals[m].sort()
                auc_outputstream.write(m.__name__+"\t"+str(auc(np.array(method_pvals[m]+[1]),np.arange(0,M+1)/float(M)))+"\n")
            if auc_filename!="":
                auc_outputstream.close()
        except:
            None
        if append_zeros_and_ones:
            outputstream.write("\t".join(map(str,[0]*(len(methods))))+"\n")
        for i in range(M):
#            outputstream.write(str((i+1.0)/M)+"\t"+"\t".join([str(method_pvals[m][i]) for m in methods])+"\n")
            outputstream.write("\t".join([str(method_pvals[m][i]) for m in methods])+"\n")

        if append_zeros_and_ones:
            outputstream.write("\t".join(["1"]*(len(methods))))
    if alt_distn_filename != "":
        outputstream.flush()
    if calculate_index_pmfs:
        f=open(index_pmf_filename,"w")
        f.write("k\t"+"\t".join([m.__name__ for m in methods])+"\n")
        for i in range(n):
            f.write(str(i+1)+"\t"+"\t".join([str(method_index_pmfs[m][i+1]/float(M)) for m in methods])+"\n")
        f.close()

def random_p_value(x):
    n=len(x)
    y=[0]*n
    for i in range(n):
        if np.isscalar(x[i]):
            y[i]=x[i]
        else:
            y[i]=np.random.uniform(low=x[i][0],high=x[i][1])
    return(y)

def get_p_values(x,methods,nmax=0,samples=10000,alpha=0.5):
    num_methods=len(methods)
    n=len(x)
    if nmax>n or nmax==0:
        nmax=n
    if any(not np.isscalar(xi) for xi in x):
        mc_pvals=np.zeros((samples,num_methods))
        for s in range(samples):
            mc_pvals[s,:]=calculate_p_values(random_p_value(x),methods,n,nmax)
        pvals=[np.percentile(mc_pvals[:,j],alpha*100) for j in range(num_methods)]
    else:
        pvals=calculate_p_values(x,methods,n,nmax)
    if ensemble in null_distns:
        pvals=np.insert(pvals,2,null_distns[ensemble](ensemble(pvals[0:2])))
    sys.stdout.write("\t".join([str(p) for p in pvals])+"\n")
    sys.stdout.flush()

def calculate_p_values(x,methods,n,nmax):
    num_methods=len(methods)
    pvals=np.zeros(num_methods)
    xs=np.array(sorted(x))
    for j in range(num_methods):
        m=methods[j]
        stat,stat_index=m(xs,n,nmax)
        if m in null_distns:
            pvals[j]=null_distns[m](stat)
        else:
            pvals[j]=stat

    return(pvals)

def get_limits(s,scale=1):
    d=sorted([float(si)/scale for si in s.split(":")])
    if len(d)==1:
        return(d[0])
    return([2*d[0]-d[1],d[1]])

def plot_cdfs(filename,methods=None,grid=None,outputfile="",indices=None):
    import matplotlib.pyplot as plt
    x=np.loadtxt(filename,skiprows=1 if methods is None else 0,ndmin=2)
    if methods is None:
        with open(filename,'r') as f:
            method_names=f.readline().split()
    else:
        method_names=[m.__name__ for m in methods]
    if indices is not None:
        x=x[:,indices]
        method_names=np.array(method_names)[indices]
    grid_size=x.shape[0] if grid is None else grid
    grid_indices=np.round(np.arange(grid_size)/float(grid_size-1)*(x.shape[0]-1)).astype(int)
    quantiles=np.arange(grid_size)/float(grid_size-1)
    pval_quantiles=np.empty([grid_size,x.shape[1]])
    for j in range(x.shape[1]):
        values=np.sort(x[:,j]) if grid is None else np.sort(x[:,j])[grid_indices]
        if outputfile=="":
            plt.plot(values,quantiles,label=method_names[j])
        else:
            pval_quantiles[:,j]=values

    if outputfile=="":
        plt.plot([0,1], [0,1], ls="--", c=".3")
        plt.legend(loc=4,prop={'size':8})
        plt.show()
    else:
        np.savetxt(outputfile,pval_quantiles,delimiter=',',header=",".join(method_names),comments="")

def get_command_line_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--seed",type=int,dest="seed", metavar="INT",default=0)
    parser.add_argument("-n","--number of p-values",type=int,dest="n", metavar="INT",default=2)
    parser.add_argument("-n0","--number of H0 Monte Carlo p-values",type=int,dest="n0", metavar="INT",default=1000000)
    parser.add_argument("-n1","--number of H1 Monte Carlo p-values",type=int,dest="n1", metavar="INT",default=10000)
    parser.add_argument("-h1","--alternative",type=str,dest="alternative_distn", metavar="STR",default="normal")
    parser.add_argument("-b","--beta",type=float,dest="beta", metavar="FLOAT",default=2.0/3.0)
    parser.add_argument("-r","--dr",type=float,dest="dr", metavar="FLOAT",default=.1)
    parser.add_argument("-a0","--alpha_0",type=float,dest="alpha_0", metavar="FLOAT",default=1)
    parser.add_argument("-nf","--null_filename",type=str,dest="null_filename", metavar="STR",default=None)
    parser.add_argument("-af","--alt_filename",type=str,dest="alt_filename", metavar="STR",default=None)
    parser.add_argument("-m","--methods",type=str,nargs='+',dest="methods",metavar="STR",default=None)
    parser.add_argument("-g","--grid",type=int,dest="grid", metavar="INT",default=0)
    parser.add_argument('--tty', dest='checkstdin', action='store_false',default=True)
    parser.add_argument('--noplot', dest='cdfplot', action='store_false',default=True)
    parser.add_argument('--noones', dest='append_zeros_and_ones', action='store_false',default=True)

    return(parser.parse_args())

closed_form_methods=[simes,fisher,pearson,sum_method,beta,stouffer,logistic,inverse_cauchy,truncated_product]
monte_carlo_methods=[standardised_product,standardised_complementary_product,standardised_sum,higher_criticism,standardised_product2,higher_criticism2,berk_jones,modified_berk_jones]#,standardised_stouffer]
doubly_monte_carlo_methods=[monte_carlo_standardised_product]
##monte_carlo_methods=[standardised_product,standardised_complementary_product,standardised_logit,standardised_sum,standardised_order_statistics,higher_criticism]
alternative_distns=['normal','exp','power','comp_power','pw_uniform','point_mass','linear','F']

args=get_command_line_arguments()
seed,n,M0,M1,alternative_distn,beta,dr,alpha_0,null_filename,checkstdin,alt_distn_filename,methods,grid=args.seed,args.n,args.n0,args.n1,args.alternative_distn,args.beta,args.dr,args.alpha_0,args.null_filename,args.checkstdin,args.alt_filename,args.methods,args.grid

try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    awfisher = importr("AWFisher")

    def aw_fisher(ps,n=0,nmax=-1):
        if ps[0]==0:
            return(-sys.float_info.max,1)
        if n==0:
            n=len(ps)
        if nmax<0 or nmax>n:
            nmax=n
        ps=ps[0:nmax]
        aw=awfisher.AWFisher_pvalue(robjects.FloatVector(ps))
        return(float(aw[0][-1]),int(sum(list(aw[1]))))

    if n<=100:
        closed_form_methods+=[aw_fisher]
except:
    None

if methods is not None:
    methods=[globals()[m] for m in methods]
else:
    methods=monte_carlo_methods+closed_form_methods#,standardised_product_subset,asymptotic_standardised_product
    if n<=10000:
        methods+=doubly_monte_carlo_methods
#methods=[partial_sum]
#methods=[monte_carlo_standardised_product]

auc_filename="auc_"+str(n)+".txt"
alt_index_pmf_filename="alt_distns_index_pmfs_"+str(n)+".txt"
if alt_distn_filename is None:
    alt_distn_filename="h1_cdfs_"+alternative_distn+"_"+str(n)+".txt"    #""+"out_with_ensemble.txt"
make_a_plot=M1==0 and grid>0

#alpha_0=1
if checkstdin and not sys.stdin.isatty():
    try:
        pvalue_maxs=np.loadtxt('max_pvalues.txt')
    except:
        pvalue_maxs=[]
        scale=1.0
#    M0=n
    n_old=None
    counter=0
    for line in sys.stdin:
        if pvalue_maxs!=[]:
            scale=pvalue_maxs[counter]
        if ":" not in line:
            x=np.array([float(xi)/scale for xi in line.strip().split()])
        else:
            x=np.array([get_limits(xi,scale) for xi in line.strip().split()])
        n=len(x)
        nmax=int(round(n*alpha_0))
        if n!=n_old:
            make_parameter_vectors(n,nmax,methods)
            if M0>0:
                null_distns=get_null_distns(n,M=M0,nmax=nmax,methods=[m for m in methods if m not in closed_form_methods],calculate_cdfs=True,seed=0)
            elif M0==0:
                null_filename="null_distns_"+str(n)+".txt"
                null_distns=get_null_distns_from_file(null_filename,methods=[m for m in methods if m not in closed_form_methods])
            else: #print the raw combined statistics
                null_distns=[]
            if n_old is None:
                sys.stdout.write("\t".join([m.__name__ for m in methods])+"\n")
            n_old=n
        get_p_values(x,methods,nmax,samples=1000)
        counter+=1

    exit(1)

nmax=int(round(n*alpha_0))
#beta=2.0/3##*0+.55#
if alternative_distn=='comp_power':
    beta=.25
#dr=.1#*2.5
k,epsilon,mu=get_k_epsilon_mu(n,beta,dr)
true_k=k
random_k=not True

make_parameter_vectors(n,nmax,methods)
#print(mu_comp_prod,sigma_comp_prod**2)

if M0>0:
    null_distns=get_null_distns(n,M=M0,nmax=nmax,methods=[m for m in methods if m not in closed_form_methods],calculate_cdfs=(M1>0 or make_a_plot),seed=seed,output_filename=null_filename,write_mcsp_to_file=True)
elif M1>0 or make_a_plot:
    if null_filename is None:
        null_filename="null_distns_"+str(n)+".txt"
    null_distns=get_null_distns_from_file(null_filename,methods=[m for m in methods if m not in closed_form_methods])
if M1>0:
#    k=1
#    k,mu=n,1
#    methods=closed_form_methods
    sys.stderr.write("k="+str(k)+", mu="+str(mu)+"\n")
    get_alternative_pvalue_distributions(n,nmax,epsilon,k,mu,random_k,M1,methods,M0>=0,alt_distn_filename,alt_index_pmf_filename,auc_filename=auc_filename,seed=seed,append_zeros_and_ones=args.append_zeros_and_ones)
    if args.cdfplot and alt_distn_filename != "":
        if ensemble in null_distns:
            methods=np.array(methods)
            methods=np.insert(methods,2,ensemble)
        plot_cdfs(alt_distn_filename)#,methods)#,200,alternative_distn+"_"+str(n)+"_pvalues.txt")

if make_a_plot:
    if n==2:
        make_3d_plot_points(num_points=grid,methods=methods,maxval=1)
    else:
#        make_2d_plot_points(num_points=grid,ps=np.arange(1,5)/float(5),methods=methods,maxval=1)
#        make_2d_plot_points(num_points=grid,ps=[0.05,0.1,0.3,0.7],methods=methods,maxval=1)
        make_2d_plot_points(num_points=grid,ps=[0.05,0.2,0.4,0.8],methods=methods,maxval=1)
