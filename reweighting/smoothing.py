import numpy as np
from typing import List, Protocol
from correctionlib import schemav2

class SmoothingProtocol(Protocol):
    def __call__(self, ratio : np.ndarray, ratioerr : np.ndarray, bin_edges : np.ndarray) -> np.ndarray:
        ...

    def to_correctionlib(self, name : str, varname : str, vardesc : str) -> schemav2.Correction:
        ...

    def evaluate(self, x : np.ndarray) -> np.ndarray:
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

    def to_correctionlib(self, name : str, varname : str, vardesc : str):
        raise NotImplementedError
    
    def evaluate(self, x : np.ndarray) -> np.ndarray:
        raise NotImplementedError

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

    def to_correctionlib(self, name : str, varname : str, vardesc : str):
        raise NotImplementedError

    def evaluate(self, x : np.ndarray) -> np.ndarray:
        raise NotImplementedError

class SplineSmoothing:
    def __init__(self, degree : int, smoothing_factor : float):
        self.degree = degree
        self.smoothing_factor = smoothing_factor
        self.spline = None

    def __call__(self, ratio : np.ndarray, ratioerr : np.ndarray, bin_edges : np.ndarray) -> np.ndarray:
        from scipy.interpolate import make_splrep
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        mask = np.isfinite(ratio) & np.isfinite(ratioerr) & (ratioerr > 0) & np.isfinite(bin_centers)
        if np.sum(mask) < self.degree + 1:
            raise ValueError("Not enough valid points to fit the spline! Need at least %d, but got %d." % (self.degree + 1, np.sum(mask)))

        self.spline = make_splrep(bin_centers[mask], ratio[mask], w=1/ratioerr[mask], k=self.degree, s=self.smoothing_factor) # type: ignore
        return np.asarray(self.spline(bin_centers))
    
    def evaluate(self, x : np.ndarray) -> np.ndarray:
        assert(self.spline is not None), "Spline has not been fitted yet"
        return self.spline(x)

    def dump(self, filename : str):
        assert(self.spline is not None), "Spline has not been fitted yet"
        knots = self.spline.t
        coeffs = self.spline.c
        degree = self.spline.k
        np.savez(filename, knots=knots, coeffs=coeffs, degree=degree)

    def get_ppoly(self):
        from scipy.interpolate import PPoly
        assert(self.spline is not None), "Spline has not been fitted yet"
        result = PPoly.from_spline(self.spline)

        # for some reason the spline can produce redundant knots at the edges
        # this causes the ppoly to have zero-length segments
        # we want to clean these up to aoivd issues in the eventual correctionlib implementation
        unique_knots, indices = np.unique(result.x, return_index=True)
        unique_coeffs = result.c[:, indices[:-1]]
        return PPoly(unique_coeffs, unique_knots)
    
    def to_correctionlib(self, name : str, varname : str, vardesc : str):
        # we implement the spline as a piecewise polynominal in correctionlib
        ppoly = self.get_ppoly()
        from correctionlib import schemav2

        binning_edges = ppoly.x.tolist()
        binning_content : List[schemav2.Content]= [
                    schemav2.FormulaRef(
                        nodetype='formularef',
                        index=0,
                        parameters=[ppoly.x[i], ppoly.c[3, i], ppoly.c[2, i], ppoly.c[1, i], ppoly.c[0, i]]
                    )
                    for i in range(len(ppoly.x) - 1)
                ]
        # add fixed underflow and overflow bins for the correct clamping behavior
        binning_edges = [binning_edges[0] - 1] + binning_edges + [binning_edges[-1] + 1]
        binning_content = [
            float(ppoly(binning_edges[1])) # underflow
        ] + binning_content + [
            float(ppoly(binning_edges[-2])) # overflow
        ]

        correction = schemav2.Correction(
            name = name,
            description = 'Spline-smoothed reweighting factor',
            version = 1,
            inputs = [
                schemav2.Variable(name=varname, type='real', description=vardesc)
            ],
            output = schemav2.Variable(name='weight', type='real'),
            generic_formulas = [
                schemav2.Formula(
                    nodetype='formula',
                    variables = [varname],
                    parser='TFormula',
                    expression = '[1] + [2] * (x - [0]) + [3] * (x - [0])^2 + [4] * (x - [0])^3' # third-order polynomial
                )
            ],
            data = schemav2.Binning(
                nodetype='binning',
                input=varname,
                edges=binning_edges,
                flow='clamp',
                content=binning_content
            )
        )
        return correction