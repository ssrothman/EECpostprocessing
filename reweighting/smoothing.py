import numpy as np
from typing import Protocol

class SmoothingProtocol(Protocol):
    def __call__(self, ratio : np.ndarray, ratioerr : np.ndarray, bin_edges : np.ndarray) -> np.ndarray:
        ...
    
class MovingAverageSmoothing:
    def __init__(self, window_size : int, weight_by_error : bool):
        self.window_size = window_size
        self.weight_by_error = weight_by_error

    def __call__(self, ratio : np.ndarray, ratioerr : np.ndarray, bin_edges : np.ndarray) -> np.ndarray:
        smoothed_ratio = np.zeros_like(ratio)
        for i in range(len(ratio)):
            start = max(0, i - self.window_size)
            end = min(len(ratio), i + self.window_size + 1)
            if self.weight_by_error:
                weights = 1 / ratioerr[start:end]
                weights /= np.sum(weights)
                smoothed_ratio[i] = np.sum(weights * ratio[start:end])
            else:
                smoothed_ratio[i] = np.mean(ratio[start:end])

        return smoothed_ratio
    
class GaussianSmoothing:
    def __init__(self, sigma : float, weight_by_error : bool):
        self.sigma = sigma
        self.weight_by_error = weight_by_error

    def __call__(self, ratio : np.ndarray, ratioerr : np.ndarray, bin_edges : np.ndarray) -> np.ndarray:
        smoothed_ratio = np.zeros_like(ratio)
        for i in range(len(ratio)):
            weights = np.exp(-0.5 * ((bin_edges[:-1] - bin_edges[i]) / self.sigma) ** 2)
            if self.weight_by_error:
                weights /= ratioerr
            weights /= np.sum(weights)
            smoothed_ratio[i] = np.sum(weights * ratio)
        return smoothed_ratio

class SplineSmoothing:
    def __init__(self, degree : int, smoothing_factor : float):
        self.degree = degree
        self.smoothing_factor = smoothing_factor
        self.spline = None

    def __call__(self, ratio : np.ndarray, ratioerr : np.ndarray, bin_edges : np.ndarray) -> np.ndarray:
        from scipy.interpolate import make_splrep
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.spline = make_splrep(bin_centers, ratio, w=1/ratioerr, k=self.degree, s=self.smoothing_factor)
        return np.asarray(self.spline(bin_centers))
    
    def dump(self, filename : str):
        knots = self.spline.t
        coeffs = self.spline.c
        degree = self.spline.k
        np.savez(filename, knots=knots, coeffs=coeffs, degree=degree)