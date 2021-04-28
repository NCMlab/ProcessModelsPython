#==================================
import nibabel as nib
import numpy as np 
from sklearn import linear_model
import pandas as pd
import time
from sklearn.utils import resample
from dataclasses import dataclass
import multiprocessing as mp
#==================================

@dataclass
class ModelInfo:
    names : dict 
    basedir : str
    nboot : int
    nsub : int
    nvar : int
    nvoxels : int
    direct : list
    inter : list
    paths : list
    model_data : list

def load_img_data(path):
    """ Returns NIfTI brain image as a 4-D numpy array.

    Keyword arguments:
    path -- the path of the NIfTI file    
    """
    nii_img = nib.load(path)
    return nii_img.get_fdata()

def create_mask(img_data):
    """Returns image data where the voxels values are 0 or 1 based on if they are nan or not.

    Keyword arguments:
    img_data -- Loaded brain imaging data
    """
    return np.where(np.isnan(img_data), 0, 1)

def create_brain_data(img_data, mask):
    """Returns n by m sized brain imaging data.

    Keyword arguments:
    img_data -- Loaded brain imaging data
    mask -- Loaded brain imaging data that represents a mask
    """
    # Check if the mask and img_data are the same size later
    brain_data = []
    for i in range(img_data.shape[3]):
        p = np.multiply(img_data[:,:,:,i], mask)
        pf = p.flatten()
        pfn = pf[np.logical_not(np.isnan(pf))]
        brain_data.append(pfn)
    return np.array(brain_data)

def calculate_beta(x,y):
    """Returns estimated coefficients and intercept for a linear regression problem.
    
    Keyword arguments:
    x -- Training data
    y -- Target values
    """
    reg = linear_model.LinearRegression().fit(x, y)
    return reg.coef_,reg.intercept_

def flatten_data(brain_data, *other_data):
    """Returns flatten brain imaging data and a list of other data that maintains the same shape as the brain imaging data.
    Keyword arguments:
    brain_data -- Loaded brain imaging data
    *other_data -- Any number of data that is the same shape of brain_data and must maintain the same shape
    """
    n,m = brain_data.shape
    all_other_data = []
    for i in range(len(other_data)):
        new_data = []
        for j in np.nditer(other_data[i]):
            new_data += [j] * m
        all_other_data.append(np.array(new_data)) 
    return brain_data.flatten(), all_other_data

def save_results(path, results):
    """ Saves the estimated coefficients and intercept into a pickle file.
    Keyword arguments:
    path -- name of the path to store the results
    results -- job list of estimated betas
    """
    new_df = pd.DataFrame(columns=['coef','intercept'])
    for f in results:
        coef,intercept = f.get(timeout=60)
        new_df = new_df.append({'coef' : coef, 'intercept' : intercept},ignore_index=True)
    #(f'results-{os.environ["SLURM_JOBID"]}.pkl')
    new_df.to_pickle(path)

def calculate_regression_model(model_info, i):
    pass

def combine_array(x,y):
    """ Returns a 2-D np array of shape (len(x), 2) from two 1-D arrays of size (len(x),).
    Keyword arguments:
    x -- np array 
    y -- np array
    """
    return np.hstack([x.reshape(-1,1),y.reshape(-1,1)])

def boot_sample(x,n,random_):
    """ Returns a resample of data x.
    Keywords Arguments:
    x -- data the needs to be resampled
    n -- number of samples
    random_ the random state that effects the seed
    """
    return resample(x, n_samples = n, replace = True, random_state = random_)

def CalculateMediationPEEffect(PointEstimate2, PointEstimate3):
    """Calculate derived effects from simple mediation model.
    Given parameter estimates from a simple mediation model,
    calculate the indirect effect, the total effect and the indirect effects
    
    Parameters
    ----------
    PointEstimate2 : array
        This is an array of parameter estimates for the regression equation
        of A on B. With no covariates, this will be an array of length 1
    PointEstimate3 : array
        This is an array of parameter estimates for the regression equation
        of A and B on C. With no covariates, this will be an array of length 2

    Returns
    -------
    IE
        The indirect effect, parameter a times b
    TE
        The total effect, which is IE plus DE
    DE
        The direct effect, the effect of A on C, when B is in the model
    a
        The effect of A on B
    b
        The effect of B on C, when A is in the model

    """
    # Indirect Effect
    a = PointEstimate2[0] # The model of B with A has one beta which is index 0
    b = PointEstimate3[1] # The model of C with A and B has two betas, b has index = 1
    IE = a*b
    # Direct Effect
    DE = PointEstimate3[0] # This is c'
    # Total Effect
    TE = DE + IE
    return IE, TE, DE, a, b

def CalculateMediationResampleEffect(BSbetalist2, BSbetalist3):
    # Indirect effect
    a  = np.array([x[0] for x in BSbetalist2])
    b  = np.array([x[1] for x in BSbetalist3])
    IE = a*b
    # Direct effect
    DE = np.array([x[0] for x in BSbetalist3])
    # Total effect
    TE = DE + IE
    return IE, TE, DE, a, b

def CalculateCI(BS, PE, alpha):
    """Calculate confidence intervals from the bootstrap resamples
    
    Confidence intervals are calculated using two difference methods:
        Percentile
            This method finds the alpha/2 percentiles at both ends of 
            the distribution of bootstratp resamples. First, the 
            index for these limits is found as: NBoot*(alpha/2).
            If this is a non interger value, it is rounded in the more
            conservative direction. Using these indices, the bootstrap 
            values are the confidence intervals.

        Bias-corrected
            It is possible that the distribution of bootstrap resmaples
            are biased with respect to the point estimate. Ideally,
            there whould be an equal number of bootstrap resample values
            above and below the point estimate. And difference is considered
            a bias. This approach adjusts for this bias. If there is no bias
            in the bootstrap resamples, the adjustment factor is zero and no
            adjustment is made so the result is the same as from the percentile
            method.
     
    Parameters
    ----------
    BS : array of length number of bootstrap resamples
        bootstrap resamples.
    PE : float
        the point estimate value.
    alpha : float
        statstical alpha used to calculate the confidence intervals.

    Returns
    -------
    PercCI : array of two floats
        Confidence intervals calculated using the percentile method.
    BCCI : array of two floats
        Confidence intervals calculated using the bias-corrected method.
    Bias : float
        The size of the bias calculated from the distribution of bootstrap
        resamples.
    """
    # If there were no bias, the zh0 would be zero
    # The percentile CI assume bias and skew are zero
    # The bias-corrected CI assume skew is zero
    NBoot = BS.shape[0]
    zA = norm.ppf(alpha/2)
    z1mA = norm.ppf(1 - alpha/2)
    
    # Percentile
    Alpha1 = norm.cdf(zA)
    Alpha2 = norm.cdf(z1mA)
    PCTlower = np.percentile(BS,Alpha1*100)
    PCTupper = np.percentile(BS,Alpha2*100)
    PercCI = [PCTlower, PCTupper]

    # Find resamples less than point estimate
    F = np.sum(BS < PE)
    if F > 0:
        pass
    else:
        F = 1 
    # Estimate the bias in the BS
    zh0 = norm.ppf(F/NBoot)
    # Calculate CI using just the bias correction
    Alpha1 = norm.cdf(zh0 + (zh0 + zA))
    Alpha2 = norm.cdf(zh0 + (zh0 + z1mA))
    PCTlower = np.percentile(BS,Alpha1*100)
    PCTupper = np.percentile(BS,Alpha2*100) 
    BCCI = [PCTlower, PCTupper]
    # Record the amount of bias and the amount of skewness in the resamples
    Bias = F/NBoot
    return PercCI, BCCI, Bias

def calculateIndPower(aEstimates, bEstimates, alpha):
    dfa = pd.read_pickle(aEstimates) # Dataframe used to read in a parameter estimates
    dfb = pd.read_pickle(bEstimates) # Dataframe used to read in b parameter estimates

    NBOOT = dfa.shape[0]
    PercList = np.zeros((NBOOT,5))
    BCList = np.zeros((NBOOT,5))
    IEBiasList = np.zeros((NBOOT,1))
    
    aCoef = dfa.values[:,0] # parameter a coefficients
    bCoef = dfb.values[:,0] # parameter b coefficients

    for i in range(NBOOT):
        IE, TE, DE, Act_a, Act_b = CalculateMediationPEEffect(aCoef[i], bCoef[i])
        BSIE, BSTE, BSDE, BSa, BSb = CalculateMediationResampleEffect(aCoef, bCoef)
        IEPercCI, IEBCCI, IEBiasList[i] = CalculateCI(BSIE, IE, alpha)
        TEPercCI, TEBCCI, TEBias = CalculateCI(BSTE, TE, alpha)
        DEPercCI, DEBCCI, DEBias = CalculateCI(BSDE, DE, alpha)
        aPercCI, aBCCI, aBias = CalculateCI(BSa, Act_a, alpha)
        bPercCI, bBCCI, bBias = CalculateCI(BSb, Act_b, alpha)

        # Make boolean list of whether the confidence intervals include zero        
        PercList[i,:] = ProcessTools.DoCIIncludeZero(IEPercCI,TEPercCI, DEPercCI, aPercCI, bPercCI)
        BCList[i,:] = ProcessTools.DoCIIncludeZero(IEBCCI,TEBCCI, DEBCCI, aBCCI, bBCCI)
    
    # Now that simulations have been done, how many simulations resulted in significant results
    PercPower = PercList.sum(0)/NBOOT
    BCPower = BCList.sum(0)/NBOOT
    BiasList = [IEBiasList.mean(), IEBiasList.std()]
