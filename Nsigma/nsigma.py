import ROOT
import os
import numpy as np
from ROOT import TMath
from functools import wraps
from ctypes import c_double
from scipy import special 


def read_data(finput, ptMinToConsider=4.0, ptMaxToConsider=12.0, firstBin=7, nBinsToConsider=6,
              method='linear-difference', gname=['gvn_prompt_stat', 'tot_syst']):
    """Read data and extract v2 values and various error components
    
    Args:
        finput: Input file path
        ptMinToConsider: Minimum pT to consider
        ptMaxToConsider: Maximum pT to consider
        firstBin: Index of the first bin to process
        nBinsToConsider: Number of bins to process
        method: Data processing method
        gname: List of graph names for data and total systematics
    
    Returns:
        Tuple of numpy arrays: (v2 values, total errors, statistical errors, 
                              resolution systematic errors, fit systematic errors, 
                              fraction systematic errors)
    """
    x, exlow, exhigh = [], [], []
    y, v2_err_tot, v2_err_stat = [], [], []  # Total error, statistical error
    eySyst_reso, eySyst_fit, eySyst_frac = [], [], []  # Systematic error components

    finput = ROOT.TFile.Open(finput)
    grPoints = finput.Get(gname[0])
    grSystTot = finput.Get(gname[1])
    grSystReso = finput.Get("reso_syst") if finput.Get("reso_syst") else None
    grSystFit = finput.Get("fit_syst") if finput.Get("fit_syst") else None
    grSystFrac = finput.Get("fd_syst") if finput.Get("fd_syst") else None

    xPoint, yPoint = c_double(0.0), c_double(0.0)
    for iBin in range(nBinsToConsider):
        idx = iBin + firstBin
        grPoints.GetPoint(idx, xPoint, yPoint)
        x.append(xPoint.value)
        exlow.append(grPoints.GetErrorXlow(idx))
        exhigh.append(grPoints.GetErrorXhigh(idx))
        y.append(yPoint.value)

        # Statistical error
        yErrStat = grPoints.GetErrorYhigh(idx)
        v2_err_stat.append(yErrStat)

        # Systematic error components
        reso_err = grSystReso.GetErrorYhigh(idx) if grSystReso else 0
        fit_err = grSystFit.GetErrorYhigh(idx) if grSystFit else 0
        frac_err = grSystFrac.GetErrorYhigh(idx) if grSystFrac else 0
        
        # Verify resolution error calculation
        factor = 0.002
        if reso_err != TMath.Sqrt((yPoint.value * factor)** 2):
            raise ValueError(f"Abnormal resolution systematic error: {reso_err} vs {TMath.Sqrt((yPoint.value * factor)**2)}")
        
        eySyst_reso.append(reso_err)
        eySyst_fit.append(fit_err)
        eySyst_frac.append(frac_err)

        # Total error calculation
        yErrSyst = TMath.Sqrt(reso_err**2 + fit_err**2 + frac_err**2)
        v2_err_tot.append(TMath.Sqrt(yErrStat**2 + yErrSyst**2))
    
    # Convert to numpy arrays
    return (np.asarray(arr, 'd') for arr in 
            [y, v2_err_tot, v2_err_stat, eySyst_reso, eySyst_fit, eySyst_frac])


def rebin_with_weighted_average(original_bins, target_bins, values, v2_err_tot, 
                               v2_err_stat=None, reso_errs=None, fit_errs=None, frac_errs=None, eps=1e-3, ignore_reso=False):
    """Rebin data to target bins using weighted average, supporting rebinning of statistical and systematic error components
    
    Args:
        original_bins: Original bin boundaries
        target_bins: Target bin boundaries
        values: Data values to rebin
        v2_err_tot: Total errors corresponding to values
        v2_err_stat: Statistical errors (optional)
        reso_errs: Resolution systematic errors (optional)
        fit_errs: Fit systematic errors (optional)
        frac_errs: Fraction systematic errors (optional)
        eps: Epsilon for floating point comparisons
        ignore_reso: Whether to ignore resolution errors in weighting
    
    Returns:
        Tuple of rebinned data and errors
    """
    if len(original_bins) - 1 != len(values) or len(values) != len(v2_err_tot):
        raise ValueError(f"Data length mismatch: original bins={len(original_bins)-1}, values={len(values)}, total errors={len(v2_err_tot)}")
    
    # Check if bins are identical to avoid unnecessary rebinning
    if np.array_equal(original_bins, target_bins):
        print('Original bins match target bins, no rebinning needed')
        return (values, v2_err_tot, v2_err_stat, reso_errs, fit_errs, frac_errs) if v2_err_stat is not None else (values, v2_err_tot)
    
    original_intervals = [(original_bins[i], original_bins[i+1]) for i in range(len(original_bins)-1)]
    target_intervals = [(target_bins[i], target_bins[i+1]) for i in range(len(target_bins)-1)]
    
    rebinned_values = []
    rebinned_v2_err_tot = []  # Rebinned total error
    rebinned_v2_err_stat = [] if v2_err_stat is not None else None  # Rebinned statistical error
    rebinned_reso = [] if reso_errs is not None else None
    rebinned_fit = [] if fit_errs is not None else None
    rebinned_frac = [] if frac_errs is not None else None
    
    for tgt_left, tgt_right in target_intervals:
        included_indices = []
        for idx, (orig_left, orig_right) in enumerate(original_intervals):
            if (orig_left >= tgt_left - eps and orig_right <= tgt_right + eps):
                included_indices.append(idx)
        
        if not included_indices:
            rebinned_values.append(np.nan)
            rebinned_v2_err_tot.append(np.nan)
            if v2_err_stat is not None:
                rebinned_v2_err_stat.append(np.nan)
                rebinned_reso.append(np.nan)
                rebinned_fit.append(np.nan)
                rebinned_frac.append(np.nan)
            continue
        
        # Extract included data
        included_values = [values[idx] for idx in included_indices]
        # Determine weighting based on ignore_reso flag
        if ignore_reso and reso_errs is not None:
            # When ignoring reso, calculate weights using errors excluding reso
            included_tot_err = []
            for idx in included_indices:
                # Recalculate total error without reso component
                stat_err = v2_err_stat[idx] if v2_err_stat is not None else 0
                fit_err = fit_errs[idx] if fit_errs is not None else 0
                frac_err = frac_errs[idx] if frac_errs is not None else 0
                total_err_no_reso = np.sqrt(stat_err**2 + fit_err**2 + frac_err**2)
                included_tot_err.append(total_err_no_reso)
        else:
            included_tot_err = [v2_err_tot[idx] for idx in included_indices]
        
        weights = [1.0 / (err **2) for err in included_tot_err if abs(err) > eps]
        
        if not weights:
            raise ZeroDivisionError(f"Zero error found in original bins {included_indices}")
        
        # Weighted average calculation
        sum_weighted = np.sum(np.array(included_values) * np.array(weights))
        sum_weights = np.sum(weights)
        weighted_mean = sum_weighted / sum_weights
        weighted_tot_err = 1.0 / np.sqrt(sum_weights)  # Rebinned total error
        
        rebinned_values.append(weighted_mean)
        rebinned_v2_err_tot.append(weighted_tot_err)
        
        # Statistical and systematic error components use weighted average
        if v2_err_stat is not None:
            included_stat = [v2_err_stat[idx] for idx in included_indices]
            included_reso = [reso_errs[idx] for idx in included_indices]
            included_fit = [fit_errs[idx] for idx in included_indices]
            included_frac = [frac_errs[idx] for idx in included_indices]
            
            rebinned_v2_err_stat.append(np.average(included_stat, weights=weights))
            rebinned_reso.append(np.average(included_reso, weights=weights) if not ignore_reso else 0)
            rebinned_fit.append(np.average(included_fit, weights=weights))
            rebinned_frac.append(np.average(included_frac, weights=weights))
    
    # Convert to numpy arrays
    result = (np.asarray(rebinned_values), np.asarray(rebinned_v2_err_tot))
    if v2_err_stat is not None:
        result += (np.asarray(rebinned_v2_err_stat), np.asarray(rebinned_reso), 
                  np.asarray(rebinned_fit), np.asarray(rebinned_frac))
    print(f"Rebinning completed: original bins {original_bins} -> target bins {target_bins}")
    return result


def filter_bins(original_bins, target_bins):
    """Filter original bins to the target range
    
    Args:
        original_bins: Original bin boundaries
        target_bins: Target range [min, max]
    
    Returns:
        Filtered bin boundaries within target range
    """
    min_t, max_t = target_bins
    filtered = [b for b in original_bins if min_t <= b <= max_t]
    
    if filtered and filtered[0] > min_t:
        filtered.insert(0, min_t)
    if filtered and filtered[-1] < max_t:
        filtered.append(max_t)
    
    return sorted(list(set(filtered)))


def weighted_merge_data(data1, err1, data2, err2, per_element=False, ignore_reso=False, 
                       stat1=None, fit1=None, frac1=None, stat2=None, fit2=None, frac2=None):
    """Weighted merge of two datasets
    
    Args:
        data1: First dataset values
        err1: First dataset errors
        data2: Second dataset values
        err2: Second dataset errors
        per_element: Whether to merge element-wise (True) or globally (False)
        ignore_reso: Whether to ignore resolution errors in weighting
        stat1, fit1, frac1: Error components for first dataset
        stat2, fit2, frac2: Error components for second dataset
    
    Returns:
        Tuple of merged data and merged errors
    """
    data1, err1 = np.asarray(data1, float), np.asarray(err1, float)
    data2, err2 = np.asarray(data2, float), np.asarray(err2, float)
    
    if len(data1) != len(err1) or len(data2) != len(err2):
        raise ValueError("Data and error length mismatch")
    if np.any(err1 < 0) or np.any(err2 < 0):
        raise ValueError("Errors cannot be negative")
    if per_element and len(data1) != len(data2):
        raise ValueError("Element-wise merging requires same length")
    
    # Calculate weights (considering whether to ignore reso errors)
    if ignore_reso and all(arr is not None for arr in [stat1, fit1, frac1, stat2, fit2, frac2]):
        err1_no_reso = np.sqrt(np.array(stat1)**2 + np.array(fit1)** 2 + np.array(frac1)**2)
        err2_no_reso = np.sqrt(np.array(stat2)** 2 + np.array(fit2)**2 + np.array(frac2)** 2)
        weight1, weight2 = 1.0/(err1_no_reso**2 + 1e-10), 1.0/(err2_no_reso**2 + 1e-10)
    else:
        weight1, weight2 = 1.0/(err1**2 + 1e-10), 1.0/(err2**2 + 1e-10)
    
    if per_element:
        sum_w = weight1 + weight2
        combined_data = (data1 * weight1 + data2 * weight2) / sum_w
        combined_err = 1.0 / np.sqrt(sum_w)
        return combined_data.tolist(), combined_err.tolist()
    else:
        sum_weighted = np.sum(data1 * weight1) + np.sum(data2 * weight2)
        sum_weights = np.sum(weight1) + np.sum(weight2)
        return sum_weighted / sum_weights, 1.0 / np.sqrt(sum_weights)


class SystCorrelationParams:
    """Configuration class for systematic error correlation parameters"""
    _case_config = {
        "reso CORR, fit & frac UNC": dict(fit_corr=0.0, frac_corr=0.0, reso_corr=1.0, ignore_reso=False),
        "ignoring reso, fit & frac UNC": dict(fit_corr=0.0, frac_corr=0.0, reso_corr=0.0, ignore_reso=True),
        "Everything UNC.": dict(fit_corr=0.0, frac_corr=0.0, reso_corr=0.0, ignore_reso=False),
        "Everything CORR.": dict(fit_corr=1.0, frac_corr=1.0, reso_corr=1.0, ignore_reso=False),
        "only fit and reso CORR.": dict(fit_corr=1.0, frac_corr=0.0, reso_corr=1.0, ignore_reso=False),
        "only fraction and reso as CORR.": dict(fit_corr=0.0, frac_corr=1.0, reso_corr=1.0, ignore_reso=False)
    }

    def __init__(self, case):
        if case not in self._case_config:
            raise ValueError(f"Unsupported test case: {case}")
        config = self._case_config[case]
        self.fit_corr = config['fit_corr']
        self.frac_corr = config['frac_corr']
        self.reso_corr = config['reso_corr']
        self.ignore_reso = config['ignore_reso']


def get_nsigma(lc_data, d0_data, description='', print_per_bin=False, method='', 
               target_bins=[4, 12], allow_negative_nsigma=False, corr_params=None):
    """Calculate Nsigma with support for systematic error correlation configuration
    
    Args:
        lc_data: Dataset 1 (Lambda_c or similar)
        d0_data: Dataset 2 (reference, e.g., D0)
        description: Description for output
        print_per_bin: Whether to print per-bin results
        method: Calculation method ('ratio' or difference-based)
        target_bins: Target bin boundaries
        allow_negative_nsigma: Whether to allow negative Nsigma values
        corr_params: SystCorrelationParams instance controlling error correlations
    
    Returns:
        Tuple of total Nsigma and per-bin Nsigma array
    """
    # Use uncorrelated configuration by default
    if corr_params is None:
        corr_params = SystCorrelationParams("Everything UNC.")
    
    # Unpack data (explicit variable meaning)
    lc_v2, lc_v2_err_tot, lc_v2_err_stat, lc_reso, lc_fit, lc_frac, lc_pt_bins = lc_data
    d0_v2, d0_v2_err_tot, d0_v2_err_stat, d0_reso, d0_fit, d0_frac, d0_pt_bins = d0_data
    
    if 'binBybin' in method:
        target_bins = lc_pt_bins if len(lc_pt_bins) < len(d0_pt_bins) else d0_pt_bins
    
    # Rebin data (including statistical and systematic error components)
    lc_rebinned = rebin_with_weighted_average(
        lc_pt_bins, target_bins, lc_v2, lc_v2_err_tot, 
        lc_v2_err_stat, lc_reso, lc_fit, lc_frac, ignore_reso=corr_params.ignore_reso)
    d0_rebinned = rebin_with_weighted_average(
        d0_pt_bins, target_bins, d0_v2, d0_v2_err_tot, 
        d0_v2_err_stat, d0_reso, d0_fit, d0_frac, ignore_reso=corr_params.ignore_reso)
    
    lc_v2, lc_v2_err_tot, lc_v2_err_stat, lc_reso, lc_fit, lc_frac = lc_rebinned
    d0_v2, d0_v2_err_tot, d0_v2_err_stat, d0_reso, d0_fit, d0_frac = d0_rebinned

    # Convert to numpy arrays
    lc_v2_arr = np.asarray(lc_v2)
    d0_v2_arr = np.asarray(d0_v2)
    lc_stat = np.asarray(lc_v2_err_stat)  # Explicitly statistical error
    d0_stat = np.asarray(d0_v2_err_stat)
    lc_reso_arr = np.asarray(lc_reso)
    d0_reso_arr = np.asarray(d0_reso)
    lc_fit_arr = np.asarray(lc_fit)
    d0_fit_arr = np.asarray(d0_fit)
    lc_frac_arr = np.asarray(lc_frac)
    d0_frac_arr = np.asarray(d0_frac)

    # Verify input shapes
    input_shapes = [lc_v2_arr.shape, d0_v2_arr.shape, lc_stat.shape, d0_stat.shape]
    if len(set(input_shapes)) != 1:
        raise ValueError(f"Input shape mismatch: {input_shapes}")
    
    # Calculate combined errors considering correlations
    # Statistical errors are always uncorrelated
    combined_stat = np.sqrt(lc_stat**2 + d0_stat**2)
    
    # Core modification: decide whether to ignore reso contribution based on config
    if corr_params.ignore_reso:
        # Ignore resolution error contribution, set to 0
        combined_reso = np.zeros_like(combined_stat)
    else:
        # Normal calculation of reso error (considering correlation)
        combined_reso = np.sqrt(
            lc_reso_arr**2 + d0_reso_arr**2 + 
            2 * corr_params.reso_corr * lc_reso_arr * d0_reso_arr
        )
    
    # Normal calculation of fit and fraction errors
    combined_fit = np.sqrt(
        lc_fit_arr**2 + d0_fit_arr**2 + 
        2 * corr_params.fit_corr * lc_fit_arr * d0_fit_arr
    )
    combined_frac = np.sqrt(
        lc_frac_arr**2 + d0_frac_arr**2 + 
        2 * corr_params.frac_corr * lc_frac_arr * d0_frac_arr
    )
    
    # Total combined error (combined_reso=0 when ignoring reso)
    combine_err_arr = np.sqrt(combined_stat**2 + combined_reso**2 + combined_fit**2 + combined_frac**2)
    
    # Calculate Nsigma
    if 'ratio' in method:
        if np.any(d0_v2_arr < 1e-10) or np.any(lc_v2_arr < 1e-10):
            zero_idx = np.where((d0_v2_arr < 1e-10) | (lc_v2_arr < 1e-10))[0]
            raise ZeroDivisionError(f"Data values near zero at indices {zero_idx}")
        
        # Calculate ratios
        ratio_lc_over_d0 = lc_v2_arr / d0_v2_arr
        ratio_d0_over_lc = d0_v2_arr / lc_v2_arr
        
        # Determine which ratio to use (ensure each term > 1)
        use_lc_over_d0 = np.all(ratio_lc_over_d0 >= 1.0 - 1e-10)
        use_d0_over_lc = np.all(ratio_d0_over_lc >= 1.0 - 1e-10)
        
        # Try automatic order swapping
        if not use_lc_over_d0 and not use_d0_over_lc:
            # Check if each bin can satisfy condition through swapping
            valid = np.logical_or(ratio_lc_over_d0 >= 1.0 - 1e-10, ratio_d0_over_lc >= 1.0 - 1e-10)
            if not np.all(valid):
                bad_indices = np.where(~valid)[0]
                raise ValueError(f"Invalid values in ratio calculation (all < 1) at indices: {bad_indices}")
            
            # Handle ratio selection per bin
            ratio = np.where(ratio_lc_over_d0 >= 1.0 - 1e-10, ratio_lc_over_d0, ratio_d0_over_lc)
            # Record numerator/denominator selection for error calculation
            is_lc_numerator = ratio_lc_over_d0 >= 1.0 - 1e-10
            
            # Select numerator/denominator data and errors (vectorized)
            numerator_data = np.where(is_lc_numerator, lc_v2_arr, d0_v2_arr)
            denominator_data = np.where(is_lc_numerator, d0_v2_arr, lc_v2_arr)
            numerator_reso = np.where(is_lc_numerator, lc_reso_arr, d0_reso_arr)
            denominator_reso = np.where(is_lc_numerator, d0_reso_arr, lc_reso_arr)
            numerator_fit = np.where(is_lc_numerator, lc_fit_arr, d0_fit_arr)
            denominator_fit = np.where(is_lc_numerator, d0_fit_arr, lc_fit_arr)
            numerator_frac = np.where(is_lc_numerator, lc_frac_arr, d0_frac_arr)
            denominator_frac = np.where(is_lc_numerator, d0_frac_arr, lc_frac_arr)
            numerator_stat = np.where(is_lc_numerator, lc_stat, d0_stat)
            denominator_stat = np.where(is_lc_numerator, d0_stat, lc_stat)
        else:
            # Select ratio globally
            if use_lc_over_d0:
                ratio = ratio_lc_over_d0
                numerator_data, denominator_data = lc_v2_arr, d0_v2_arr
                numerator_reso, denominator_reso = lc_reso_arr, d0_reso_arr
                numerator_fit, denominator_fit = lc_fit_arr, d0_fit_arr
                numerator_frac, denominator_frac = lc_frac_arr, d0_frac_arr
                numerator_stat, denominator_stat = lc_stat, d0_stat
            else:
                ratio = ratio_d0_over_lc
                numerator_data, denominator_data = d0_v2_arr, lc_v2_arr
                numerator_reso, denominator_reso = d0_reso_arr, lc_reso_arr
                numerator_fit, denominator_fit = d0_fit_arr, lc_fit_arr
                numerator_frac, denominator_frac = d0_frac_arr, lc_frac_arr
                numerator_stat, denominator_stat = d0_stat, lc_stat
        
        # Calculate relative errors (numerator_reso/denominator_reso=0 when ignoring reso)
        if corr_params.ignore_reso:
            rel_err_num = np.sqrt(
                (numerator_stat/numerator_data)**2 + 
                (numerator_fit/numerator_data)**2 + 
                (numerator_frac/numerator_data)** 2
            )
            rel_err_den = np.sqrt(
                (denominator_stat/denominator_data)**2 + 
                (denominator_fit/denominator_data)**2 + 
                (denominator_frac/denominator_data)** 2
            )
            # Relative error considering correlations (remove reso term)
            ratio_err = ratio * np.sqrt(
                rel_err_num**2 + rel_err_den**2 - 
                2*(corr_params.fit_corr*numerator_fit*denominator_fit + 
                   corr_params.frac_corr*numerator_frac*denominator_frac)/(numerator_data*denominator_data)
            )
        else:
            rel_err_num = np.sqrt(
                (numerator_stat/numerator_data)**2 + 
                (numerator_reso/numerator_data)** 2 + 
                (numerator_fit/numerator_data)**2 + 
                (numerator_frac/numerator_data)** 2
            )
            rel_err_den = np.sqrt(
                (denominator_stat/denominator_data)**2 + 
                (denominator_reso/denominator_data)** 2 + 
                (denominator_fit/denominator_data)**2 + 
                (denominator_frac/denominator_data)** 2
            )
            # Relative error considering correlations
            ratio_err = ratio * np.sqrt(
                rel_err_num**2 + rel_err_den**2 - 
                2*(corr_params.reso_corr*numerator_reso*denominator_reso + 
                   corr_params.fit_corr*numerator_fit*denominator_fit + 
                   corr_params.frac_corr*numerator_frac*denominator_frac)/(numerator_data*denominator_data)
            )
        
        if np.any(ratio_err < 1e-10):
            raise ZeroDivisionError(f"Ratio error near zero")
        
        nsigma_arr = (ratio - 1.0) / ratio_err
    else:
        if np.any(combine_err_arr == 0):
            raise ZeroDivisionError(f"Combined error is zero")
        
        # Calculate differences
        diff_lc_minus_d0 = lc_v2_arr - d0_v2_arr
        diff_d0_minus_lc = d0_v2_arr - lc_v2_arr
        
        # Determine which difference to use (ensure each term is positive)
        use_lc_minus_d0 = np.all(diff_lc_minus_d0 >= -1e-10)
        use_d0_minus_lc = np.all(diff_d0_minus_lc >= -1e-10)
        
        # Try automatic order swapping
        if not use_lc_minus_d0 and not use_d0_minus_lc:
            # Check if each bin can satisfy condition through swapping
            valid = np.logical_or(diff_lc_minus_d0 >= -1e-10, diff_d0_minus_lc >= -1e-10)
            if not np.all(valid):
                bad_indices = np.where(~valid)[0]
                raise ValueError(f"Invalid values in difference calculation (all negative) at indices: {bad_indices}")
            
            # Handle difference selection per bin
            diff = np.where(diff_lc_minus_d0 >= -1e-10, diff_lc_minus_d0, diff_d0_minus_lc)
        elif use_lc_minus_d0:
            diff = diff_lc_minus_d0
        else:
            diff = diff_d0_minus_lc
        
        nsigma_arr = diff / combine_err_arr
    
    # Handle negative nsigma (if not allowed)
    if not allow_negative_nsigma and np.any(nsigma_arr < 0):
        neg_indices = np.where(nsigma_arr < 0)[0]
        nsigma_arr[neg_indices] = np.abs(nsigma_arr[neg_indices])
        if print_per_bin:
            print(f"Warning: Negative Nsigma found at indices {neg_indices}, absolute values taken")
    
    # Print per-bin results
    if print_per_bin:
        for i, nsigma in enumerate(nsigma_arr):
            bin_range = f"{target_bins[i]}-{target_bins[i+1]}"
            print(f"Bin {bin_range}: {description} Nsigma = {nsigma:.2f}")
    
    prob_pt = (1 - special.erf(nsigma_arr / np.sqrt(2))) / 2.0
    tot_prob = np.prod(prob_pt)
    tot_nsigma = special.erfinv(2.*(1-tot_prob)-1)*np.sqrt(2)
    return tot_nsigma, nsigma_arr


def read_particle_data(finput_path, pt_bins, all_pt_bins, first_bin, n_bins_to_consider, method):
    """Helper function to read particle data
    
    Args:
        finput_path: Input file path
        pt_bins: pT bins to consider
        all_pt_bins: All available pT bins
        first_bin: Index of first bin to process
        n_bins_to_consider: Number of bins to process
        method: Data processing method
    
    Returns:
        Output from read_data function
    """
    return read_data(
        finput_path,
        ptMinToConsider=pt_bins[0],
        ptMaxToConsider=pt_bins[-1],
        firstBin=first_bin,
        nBinsToConsider=n_bins_to_consider,
        method=method
    )


def _read_and_filter_data(file_path, particle_name, all_bins, target_bins, method):
    """Read and filter data for specified particle
    
    Args:
        file_path: Path to data file
        particle_name: Name of particle
        all_bins: All available bins
        target_bins: Target bin range
        method: Data processing method
    
    Returns:
        Tuple of (filtered data, filtered bins)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filtered_bins = filter_bins(all_bins, target_bins)
    start_index = all_bins.index(filtered_bins[0])
    length = len(filtered_bins) - 1

    v2, err_tot, stat, reso, fit, frac = read_particle_data(
        file_path, filtered_bins, all_bins, start_index, length, method)
    return [v2, err_tot, stat, reso, fit, frac, filtered_bins], filtered_bins


def _merge_d0_dplus(d0_file, dplus_file, target_bins, all_d0_bins, all_dplus_bins, method, corr_params):
    """Merge D0 and D+ data
    
    Args:
        d0_file: Path to D0 data file
        dplus_file: Path to D+ data file
        target_bins: Target bin range
        all_d0_bins: All D0 bins
        all_dplus_bins: All D+ bins
        method: Data processing method
        corr_params: Systematic correlation parameters
    
    Returns:
        Tuple of (merged data, merged bins)
    """
    d0_data, d0_bins = _read_and_filter_data(d0_file, "D0", all_d0_bins, target_bins, method)
    dplus_data, _ = _read_and_filter_data(dplus_file, "D+", all_dplus_bins, target_bins, method)

    d0_v2, d0_err, d0_stat, d0_reso, d0_fit, d0_frac, _ = d0_data
    dplus_v2, dplus_err, dplus_stat, dplus_reso, dplus_fit, dplus_frac, _ = dplus_data

    # Weighted merging
    combined_v2, combined_err = weighted_merge_data(
        d0_v2, d0_err, dplus_v2, dplus_err, per_element=True,
        ignore_reso=corr_params.ignore_reso,
        stat1=d0_stat, fit1=d0_fit, frac1=d0_frac,
        stat2=dplus_stat, fit2=dplus_fit, frac2=dplus_frac
    )
    
    # Merge error components
    combined_stat = np.sqrt(np.array(d0_stat)**2 + np.array(dplus_stat)** 2) / 2
    combined_reso = np.sqrt(np.array(d0_reso)**2 + np.array(dplus_reso)** 2) / 2 if not corr_params.ignore_reso else np.zeros_like(combined_v2)
    combined_fit = np.sqrt(np.array(d0_fit)**2 + np.array(dplus_fit)** 2) / 2
    combined_frac = np.sqrt(np.array(d0_frac)**2 + np.array(dplus_frac)** 2) / 2

    return [combined_v2, combined_err, combined_stat, combined_reso, combined_fit, combined_frac, d0_bins], d0_bins


def run(method, target_particle='lc', corr_case="Everything UNC.", combine_nonstrange=False):
    """Main workflow for Nsigma calculation
    
    Args:
        method: Calculation method
        target_particle: Particle to analyze ('lc' or 'ds')
        corr_case: Systematic error correlation case
        combine_nonstrange: Whether to combine non-strange particles (D0+D+)
    
    Returns:
        Tuple of (total Nsigma, per-bin Nsigma array)
    """
    print(f"Method: {method}, Particle: {target_particle}, Correlation case: {corr_case}, Combine non-strange: {combine_nonstrange}")
    corr_params = SystCorrelationParams(corr_case)

    # Path and bin configuration
    base_path = "../input-data/lc-d0-data/"
    file_map = {
        'lc': (os.path.join(base_path, 'merged-lc-promptvn_withsyst_r2-0.2%.root'), [2,3,4,5,6,8,12,24], [4,24], 'Lambda_c'),
        'ds': (os.path.join(base_path, 'v2_prompt_wsyst_Ds_3050.root'), [1,2,3,4,5,6,7,8,10,24], [1,5], 'Ds')
    }
    if target_particle not in file_map:
        raise ValueError(f"Unsupported target particle: {target_particle}")
    particle_file, all_bins, target_bins, desc = file_map[target_particle]

    # Read target particle data
    try:
        particle_data, particle_bins = _read_and_filter_data(particle_file, desc, all_bins, target_bins, method)
    except Exception as e:
        print(f"Failed to read {desc} data: {e}")
        return None, None

    # Read reference data
    if combine_nonstrange:
        print(f'Comparison: {desc} vs (D0+D+)')
        ref_data, ref_bins = _merge_d0_dplus(
            os.path.join(base_path, 'v2_prompt_wsyst_D0_3050_finer.root'),
            os.path.join(base_path, 'v2_prompt_wsyst_Dplus_3050.root'),
            target_bins, [0.5,1,1.5,2,2.5,3,3.5,4,5,6,7,8,10,12,16,24],
            [1,1.5,2,2.5,3,3.5,4,5,6,7,8,10,12,16,24], method, corr_params
        )
    else:
        print(f'Comparison: {desc} vs D0')
        ref_data, ref_bins = _read_and_filter_data(
            os.path.join(base_path, 'v2_prompt_wsyst_D0_3050_finer.root'),
            "D0", [0.5,1,1.5,2,2.5,3,3.5,4,5,6,7,8,10,12,16,24], target_bins, method
        )

    # Calculate Nsigma
    print(f'{desc} bins: {particle_bins}')
    print(f'Reference particle bins: {ref_bins}')
    
    total_nsigma, nsigma_arr =  get_nsigma(
        particle_data, ref_data, 
        description=f'{desc} vs {"(d0+dplus)" if combine_nonstrange else "d0"}',
        method=method, corr_params=corr_params
    )
    return total_nsigma, nsigma_arr


def test_all_correlation_cases():
    """Test all systematic error correlation cases"""
    configs = {
        'methods': ['binBybin-difference'],
        'particles': ['lc', 'ds'],
        'cases': [
            "reso CORR, fit & frac UNC",
            "ignoring reso, fit & frac UNC",
            "Everything UNC.",
            "Everything CORR.",
            "only fit and reso CORR.",
            "only fraction and reso as CORR."
        ],
        'combine_options': [True, False]
    }
    results = {}

    for particle in configs['particles']:
        results[particle] = {}
        for method in configs['methods']:
            results[particle][method] = {}
            for case in configs['cases']:
                results[particle][method][case] = {}
                for combine in configs['combine_options']:
                    print(f"\n===== Test combination: Particle={particle}, Method={method}, Case={case}, Combine non-strange={combine} =====")
                    tot_nsigma, nsigma_arr = run(
                        method=method, target_particle=particle,
                        corr_case=case, combine_nonstrange=combine
                    )
                    results[particle][method][case][combine] = {
                        'status': 'success',
                        'nsigma_arr': nsigma_arr,
                        'tot_nsigma': tot_nsigma
                    }
                    print(f"Test result: {'Success' if results[particle][method][case][combine]['status'] == 'success' else 'Failed'}")

    # Summary output
    print("\n" + "="*80)
    print("Summary of all test cases")
    print("="*80)
    for particle in configs['particles']:
        print(f"\n----- Particle: {particle} -----")
        for method in configs['methods']:
            print(f"  Method: {method}")
            for case in configs['cases']:
                print(f"    Correlation case: {case}")
                for combine in configs['combine_options']:
                    res = results[particle][method][case][combine]
                    combine_str = "Combine non-strange" if combine else "Do not combine non-strange"
                    if res['status'] == 'success':
                        nsigma_str = f"Array: {res['nsigma_arr']}, Total Nsigma: {res['tot_nsigma']}"
                        print(f"      {combine_str}: {nsigma_str}")
                    else:
                        print(f"      {combine_str}: Failed ({res['error']})")
    print("\n" + "="*80)
    print("Testing completed")


test_all_correlation_cases()