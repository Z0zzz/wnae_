from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import njit
from sklearn.metrics import roc_curve

# from utilities.utilities import warn


@njit
def _is_sorted(a):
    assert a.ndim == 1
    for i in range(a.size - 1):
        if a[i] > a[i + 1]:
            return False
    return True


@njit
def fast_roc(
    score_sig: np.ndarray,
    score_bkg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the exact ROC curve Tuple([FP],[TP]) from two score sheets. It is assumed that signal are meant to have higher scores. The sample points on the curve are equally distributed on TP axis.
    Parameters:
    -------------------
        score_bkg: 1D np.ndarray containing scores for background
        score_sig: 1D np.ndarray containing scores for signal
    -------------------
    """
    assert score_bkg.ndim == score_sig.ndim == 1
    bkg_n = score_bkg.size
    sig_n = score_sig.size
    if not _is_sorted(score_bkg):
        score_bkg = np.sort(score_bkg)

    if not _is_sorted(score_sig):
        score_sig = np.sort(score_sig)
    ptr_sig = 0
    ptr_bkg = 0

    buf = np.empty((2, 2 * min(score_sig.size, score_bkg.size) + 2))
    buf[:, -1] = 0, 0
    buf_ptr = -2

    while ptr_sig < sig_n and ptr_bkg < bkg_n:
        if score_bkg[ptr_bkg] < score_sig[ptr_sig]:
            while ptr_bkg < bkg_n and score_bkg[ptr_bkg] < score_sig[ptr_sig]:
                ptr_bkg += 1

        elif score_bkg[ptr_bkg] > score_sig[ptr_sig]:
            while ptr_sig < sig_n and score_bkg[ptr_bkg] > score_sig[ptr_sig]:
                ptr_sig += 1
        else:
            if ptr_sig < sig_n and ptr_bkg < bkg_n and score_bkg[ptr_bkg] == score_sig[ptr_sig]:
                ptr_bkg += 1
                ptr_sig += 1

        buf[:, buf_ptr] = ptr_sig / sig_n, ptr_bkg / bkg_n
        buf_ptr -= 1
    buf[:, buf_ptr] = 1, 1
    TP, FP = 1 - buf[:, buf_ptr:]
    return FP, TP


@njit
def fast_roc_sample(sorted_bkg: np.ndarray, sig: np.ndarray, fpr: np.ndarray) -> Tuple[NDArray, NDArray]:
    """Calculate the ROC curve for a given set of signal and background scores.
    Parameters
    ----------
    sorted_bkg : np.ndarray
        Sorted background scores.
    sig : np.ndarray
        Signal scores.
    fpr : np.ndarray
        False positive rates to calculate the ROC curve at.
    Returns
    -------
    tpr : np.ndarray
        True positive rates.
    """

    n_bkg = sorted_bkg.shape[0]
    n_sig = sig.shape[0]
    tpr = np.empty_like(fpr)
    thresholds = np.empty_like(fpr)
    for i, _fp in enumerate(fpr):
        n = int(n_bkg * _fp)
        threshold = sorted_bkg[n] if n < n_bkg else sorted_bkg[-1]
        tpr[i] = np.sum(sig > threshold) / n_sig
        thresholds[i] = threshold
    return tpr, thresholds
