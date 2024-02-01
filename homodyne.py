import streamlit as st
import math
from scipy.special import genlaguerre, laguerre, factorial
import gmpy2
import re
import sqtdiat.qops as sq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import plotly.graph_objs as go
import plotly.express as px
from scipy.interpolate import interp1d, interp2d
from scipy.signal import convolve
from skimage import transform
import time
import pandas as pd
import base64
from functools import partial

def wig_coherent(alpha, xmin, xmax, pmin, pmax, res = 200, g = np.sqrt(2), return_vecs = False):
    """
    Generates a normalized Wigner distribution for a coherent state in phase space.

    Args:
    alpha (complex): Complex number defining the coherent state.
    xmin (float): Minimum value for the x-axis.
    xmax (float): Maximum value for the x-axis.
    pmin (float): Minimum value for the momentum (p) axis.
    pmax (float): Maximum value for the momentum (p) axis.
    res (int, optional): Resolution for the grid. Defaults to 200.
    g (float, optional): Parameter for state representation. Defaults to sqrt(2).
    return_vecs (bool, optional): Whether to return coordinate vectors. Defaults to False.

    Returns:
    numpy.ndarray or tuple: Normalized Wigner distribution for the coherent state or 
                            tuple with distribution and coordinate vectors (if return_vecs=True).
    """
    if xmax - xmin < 3 * np.abs(alpha) or pmax - pmin < 3 * np.abs(alpha):
        st.write(f"Unchangeable dimensions {2 * np.abs(alpha)}")
        xmin, xmax, pmin, pmax = -2 * np.abs(alpha), 2 * np.abs(alpha), -2 * np.abs(alpha), 2 * np.abs(alpha)
    xvec = np.linspace(xmin, xmax, res)
    pvec = np.linspace(pmin, pmax, res)
    X, P = np.meshgrid(xvec, pvec)
    wig = np.exp((-(X - g * np.real(alpha))**2 - (P + g * np.imag(alpha))**2))
    norm_wig = wig / np.sum(wig)
    if return_vecs == True:
        return norm_wig, xvec, pvec
    return norm_wig
    
def wig_vac_squeezed(r, theta, res = 200, return_axes = False):
    """
    Generates the Wigner distribution for a vacuum squeezed state in phase space.

    Args:
    r (float): Squeezing parameter.
    theta (float): Phase angle in degrees.
    res (int, optional): Resolution for the grid. Defaults to 200.
    return_axes (bool, optional): Whether to return coordinate axes. Defaults to False.

    Returns:
    numpy.ndarray or tuple: Wigner distribution for the vacuum squeezed state or 
                            tuple with distribution and coordinate axes (if return_axes=True).
    """
    xv = np.linspace(-10, 10, res)
    X, P = np.meshgrid(xv, xv)
    th = np.deg2rad(theta)
    wig = np.exp(-2 * ((X * np.cos(th) + P * np.sin(th))**2 * np.exp(-2 * (r)) + (-X * np.sin(th) + P * np.cos(th))**2 * np.exp(2 * r))) * 2 / np.pi
    if return_axes == True:
        return wig, xv, xv
    return wig

def wig_loss(wig_dis, eta, xvec, pvec):
    """
    Applies loss to a given Wigner distribution.

    Args:
    wig_dis (numpy.ndarray): Input Wigner distribution.
    eta (float): Loss parameter.
    xvec (numpy.ndarray): x-axis values.
    pvec (numpy.ndarray): p-axis values.

    Returns:
    numpy.ndarray: Wigner distribution after applying loss.
    """
    X, P = np.meshgrid(xvec, pvec)
    s = eta / (1 - eta)
    s_arr = np.exp(-s * (X**2 + P**2))
    s_arr /= np.sum(s_arr)
    return convolve(wig_after_loss(wig_dis, np.sqrt(eta)), s_arr, mode = 'same')

def fact1(n):
    return float(gmpy2.sqrt(gmpy2.fac(n)))

def coh(n, alpha):
    base = np.exp(-np.abs(alpha)**2)
    coh_arr = np.zeros((n), dtype=object)
    for i in range(n):
        coh_arr[i] = alpha**i / gmpy2.sqrt(gmpy2.factorial(i))
    return np.array(np.array(coh_arr, dtype=object) * base, dtype = "complex128")

def coh2(n, alpha):
    base = np.exp(-np.abs(alpha)**2)
    coh_arr = np.zeros((n), dtype=object)
    for i in range(n):
        coh_arr[i] = gmpy2.div(alpha**i, gmpy2.sqrt(fact1(i)))
    return np.array(coh_arr, dtype = object) * base

def rho_input(inp_, type = "state_vec"):
    types = ['state_vec', 'mixed_state', 'coherent', 'cat_states', 'vac_squeezed_states']
    if type not in types:
        raise ValueError("Please enter allowed type")
    else:
        if type == 'state_vec':
            if inp_.shape[0] < 2:
                inp_ = np.array([1, 0])
            inp_ = sq.norm_state_vec(inp_)
            return np.outer(inp_, np.conj(inp_))
        elif type == 'mixed_state':
            # inp_ = 
            pass
        elif type == 'coherent':
            # coh_size = st.radio("")
            N, alpha = inp_
            coh_state = np.array(coh(N, alpha), dtype = 'complex128')
            return np.outer(coh_state, np.conj(coh_state))
        elif type == 'cat_states':
            N, alphas, c_is = inp_
            norm_c_is = sq.norm_state_vec(c_is)
            cat_st = np.zeros((N), dtype = 'complex128')
            for c_i, alpha in zip(norm_c_is, alphas): cat_st += c_i * np.array(coh(N, alpha), dtype = 'complex128')
            return np.outer(cat_st, np.conj(cat_st))
        # elif type == 'coherent':

# def rad(img, )

def wigner_laguerre(rho, x_min = -10, x_max = 10, p_min = -10, p_max = 10, res = 200, return_axes = False, fixed = False):
    """A very basic code for obtaining the wigner distribution from the density matrix. 
    Refs:
    1. Ulf leonhardt - Measuring the Quantum States of Light. Chapter 5, Section 5.2.6 pg.nos 128-129
    2. "Numerical study of Wigner negativity in one-dimensional steady-state resonance fluorescence"
    arXiv: 1909.02395v1 Appendix A
    """
    if type(rho) != np.array:
        rho = np.array(rho)
    if rho.shape[0] == rho.shape[1]:
        x_vec = np.linspace(x_min, x_max, res)
        p_vec = np.linspace(p_min, p_max, res)
        X, P = np.meshgrid(x_vec, p_vec)
        A = X + 1j * P
        B = np.abs(A)**2
        W = np.zeros((res, res))
        for n in range(rho.shape[0]):
            if np.abs(rho[n, n]) > 0:
                W += np.real(rho[n, n] * (-1) ** n * genlaguerre(n, 0)(2 * B))
            for m in range(0, n):
                if np.abs(rho[m, n]) > 0:
                    W += 2 * np.real(rho[m, n] * (-1) ** m * np.sqrt(2 ** (n - m) * factorial(m) / factorial(n)) * genlaguerre(m, n - m)(2 * B) * A ** (n - m))
        W = W * np.exp(-B)  / np.pi
        if return_axes == True: return W / np.sum(W), x_vec, p_vec 
        return W / np.sum(W)
    else:
        raise ValueError("Dim. mismatch between rows and columns")

def sim_homodyne_data(wg, xv, theta_steps = 180, ADC_bits = 8, pts = 100, need_elec_noise = True, elec_var = 0.3, data_res = 10):
    """
    Simulates homodyne data with optional electronic noise and detector losses.

    Args:
    wg (numpy.ndarray): Wigner distribution.
    xv (numpy.ndarray): x-axis values.
    theta_steps (int): Number of quadratures to be measured in one whole period.
    ADC_bits (int): ADC used for sampling.
    pts (int): Number of measurements per quadrature.
    need_elec_noise (bool): Flag to include electronic noise.
    elec_var (float): Relative Variance of Electronic Noise w.r.t Vacuum Noise.
    data_res (int): Spacing between the discrete values obtained.

    Returns:
    numpy.ndarray: Simulated homodyne data.
    """
    # ADC_bits = 8
    thetas = np.linspace(0, 359, theta_steps)
    # pts = 100
    mask = np.array([1 if x % data_res == 0 else 0 for x in range(2**ADC_bits * data_res)])
    all_data = np.zeros((thetas.shape[0] * pts))
    # elec_var = 0.4
    for i, t in enumerate(thetas):
        f = interp1d(xv, transform.rotate(wg, t).sum(0))
        discrete_p = f(np.linspace(xv[0], xv[-1], 2**ADC_bits * data_res)) * mask
        discrete_p = discrete_p / np.sum(np.abs(discrete_p))
        # if np.sum(np.abs(discrete_p)) != 1: print(np.sum(np.abs(discrete_p)))
        data = np.random.choice(np.linspace(xv[0], xv[-1], 2**ADC_bits * data_res), p = np.abs(discrete_p), size = (pts))
        all_data[i * pts: (i + 1) * pts] = data
    if need_elec_noise == True:
        elec_f = np.exp(-(xv)**2 / elec_var)
        elec_fun = interp1d(xv, elec_f)
        elec_p = elec_fun(np.linspace(xv[0], xv[-1], 2**ADC_bits * data_res)) * mask
        elec_p /= np.sum(elec_p)
        elec_noise = np.random.choice(np.linspace(xv[0], xv[-1], 2**ADC_bits * data_res), p = np.abs(elec_p), size = (thetas.shape[0] * pts))
        return all_data + elec_noise
    return all_data

def download_csv(df):
    """
    Downloads simulated data as a CSV file.

    Args:
    df (pandas.DataFrame): Dataframe containing the simulated data.

    Returns:
    str: HTML formatted link to download the CSV file.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sim_data.csv">Simulated data in csv format</a>'
    return href

def perform_interpolation(arr1, arr2, m=100, x_min = -5, x_max = 5, padding=1.0, kind='cubic'):
    # Find non-zero elements and their indices
    non_zero_indices = np.nonzero(arr1)
    filtered_arr1 = arr1[non_zero_indices]
    filtered_arr2 = arr2[non_zero_indices]

    # Create x_new array for interpolation
    x_new = np.linspace(padding * x_min, padding * x_max, m)
    
    # Perform interpolation based on the specified kind (default: cubic)
    interp_func = interp1d(filtered_arr2, filtered_arr1, kind=kind, bounds_error = False, fill_value = 0)
    
    # Generate interpolated data for the x_new array
    interpolated_data = interp_func(x_new)
    
    return x_new, interpolated_data

def meas_data_2_hist(sim_data, theta, data_points, dat_min, dat_max, bins, m = 360):
    # pts = 360
    # padding = 1.2
    # dat_min, dat_max = np.min(sim_data), np.max(sim_data)
    full = np.zeros((m, theta))
    for i in range(theta):
        a, b = np.histogram(sim_data[i * data_points: (i + 1) * data_points], bins = bins)
        _, c = perform_interpolation(a, b, m = m, x_min = dat_min, x_max = dat_max)
        full[:, i] = np.abs(c)[::-1]
    return full

def filter(size):
    """Copied this filter from skimage.transform.iradon - Source code page"""
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    f_ = np.fft.ifft(f)
    return np.real(f_[:, np.newaxis] / np.max(f_))

def inv_rad(sino, threshold = 210):
    """
    Function:
    
    Performs the inverse Radon transform to reconstruct the original 2D image from the Radon transform array.
    
    Arguments:
    
    - sino: The Radon transform array obtained from the `rad` function. It can be a list or a NumPy array.
    - threshold: The threshold value used for filtering. Values below this threshold are set to 0. Defaults to 210.
    
    Returns:
    
    The reconstructed 2D image.
    """
    if type(sino) != np.array: sino = np.array(sino)
    pad_sino = sino.shape[1] // 7
    base = np.zeros((sino.shape[1], sino.shape[1]))
    if sino.shape[0] % 180 == 0:
        res = 360 / sino.shape[0]
        f_filter = [np.abs(x) if x <= threshold else 0 for x in range(sino.shape[1])]
        for theta in [x for x in np.arange(0, 360, res)]:
            temp = []
            org_val = sino[int(theta / res)]
            org_val_fft = np.fft.fft(org_val)
            changed = f_filter * org_val_fft
            poss = [freq for freq in range(len(org_val_fft))]
            org_val_pure = np.fft.ifft(changed)
            for i in range(sino.shape[1]):
                temp.append(np.real(org_val_pure))
            base = base + transform.rotate(np.array(temp), theta, resize = False)
        return base[pad_sino:base.shape[0] - pad_sino, pad_sino:base.shape[1] - pad_sino] / np.sum(base)
    else:
        raise ValueError(f"The input array has {sino.shape[0]} rows, reshape it to be a multiple of 180.")

def irad(hist_2d, thetas):
    all_thetas = np.linspace(0, 360, thetas)
    filt = filter(hist_2d.shape[0])
    filtered_img = np.real(np.fft.ifft(np.fft.fft(hist_2d, axis = 0) * filt, axis = 0))
    final_img = np.zeros((hist_2d.shape[0], hist_2d.shape[0]))
    x, p = np.mgrid[:hist_2d.shape[0], :hist_2d.shape[0]] - hist_2d.shape[0] // 2
    x_arr = np.arange(hist_2d.shape[0]) - hist_2d.shape[0] // 2
    for col, theta in zip(range(hist_2d.shape[0]), all_thetas):
        final_img += partial(np.interp, xp = x_arr, fp = filtered_img[:, col], left = 0, right = 0)(-x * np.sin(np.deg2rad(theta)) - p * np.cos(np.deg2rad(theta)))
    return final_img

def wig_after_loss(arr, eta):
    n = arr.shape[0]
    if eta > 1: raise ValueError("eta must be less than 1.")
    new_n = int((1 * np.sqrt(eta)) * n)
    
    x = np.arange(0, n)
    y = np.arange(0, n)
    a = int((n - 1) / eta)
    b = (a - n) // 2
    new_x = np.linspace(-2 * b, a, n)
    new_y = np.linspace(-2 * b, a, n)
    # new_x = np.linspace(0, int((n - 1) / np.sqrt(eta)), n)
    # new_y = np.linspace(0, int((n - 1) / np.sqrt(eta)), n)
    
    xx, yy = np.meshgrid(x, y)
    new_xx, new_yy = np.meshgrid(new_x, new_y)
    
    interp = interp2d(x, y, arr, kind='cubic')
    
    return interp(new_x, new_y)
