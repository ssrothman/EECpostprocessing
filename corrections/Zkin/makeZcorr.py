from samples.latest import SAMPLE_LIST
import numpy as np
from plotting.util import config
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.optimize

data = "DATA_2018UL"

signal = 'DYJetsToLL'

backgrounds = [
    'WW', 'ZZ', 'WZ',
    'TTTo2L2Nu',
    'ST_tW_top', 'ST_tW_antitop',
    'ST_t_top_5f', 'ST_t_antitop_5f'
]

slicing = (slice(1,None), slice(1,25))

Hdata = SAMPLE_LIST.get_hist(data, 'Kin', ['tight'])['HZ']
edges_Zpt = np.asarray(Hdata.axes['Zpt'].edges.tolist() + [np.inf])
edges_Zy = Hdata.axes['Zy'].edges[:25]
err2data = Hdata.project("Zpt", "Zy").variances(flow=True)[slicing]
Hdata = Hdata.project("Zpt", 'Zy').values(flow=True)[slicing]

Hsignal = SAMPLE_LIST.get_hist(signal, 'Kin', ['tight'])
Hreweighted = SAMPLE_LIST.get_hist(signal, 'Kin', ['tight', 'Zreweight'])

xsec_signal = config.xsecs.DYJetsToLL
factor_signal = 1000 * config.totalLumi * xsec_signal / Hsignal['sumwt']
Hsignal = Hsignal['HZ'].project("Zpt", 'Zy')
Hreweighted = Hreweighted['HZ'].project("Zpt", 'Zy')
err2signal = factor_signal*factor_signal*Hsignal.variances(flow=True)[slicing]
Hsignal = factor_signal*Hsignal.values(flow=True)[slicing]
Hreweighted = factor_signal*Hreweighted.values(flow=True)[slicing]

Hbackground = np.zeros_like(Hdata)
err2background = np.zeros_like(Hdata)

for background in backgrounds:
    if background.endswith("5f"):
        xsec = vars(config.xsecs)[background[:-3]]
    else:
        xsec = vars(config.xsecs)[background]

    back = SAMPLE_LIST.get_hist(background, 'Kin', ['tight'])

    factor = 1000 * config.totalLumi * xsec / back['sumwt']

    Hback = back['HZ'].project("Zpt", 'Zy').values(flow=True)[slicing]
    err2back = back['HZ'].project("Zpt", 'Zy').variances(flow=True)[slicing]
    Hbackground = Hbackground + factor*Hback
    err2background = err2background + factor*factor*err2back

Hdata_sub = Hdata - Hbackground
err2data_sub = err2data + err2background

ratio = np.nan_to_num(Hdata_sub / Hsignal)
ratio = np.where(ratio > 5, 5, ratio)
ratio = np.where(ratio < 0, 0, ratio)

ratioerr = np.nan_to_num(np.sqrt(err2data_sub/np.square(Hdata_sub) \
                   + err2signal/np.square(Hsignal))) * ratio + 0.05
#ratio[24, 22] = 0

def fitfunc(x, b, c):
    return (1+ b*x + c*x*x)/(1 + b*x)


#plt.show()


ratio_smooth = ratio.copy()
for iPT in range(ratio.shape[0]):
    for iY in range(ratio.shape[1]):
        nNeighbors = 0
        sumNeighbors = 0
        
        if iPT > 15:
            for dx in [0]:
                for dy in [-1, 0, 1]:
                    #if dx*dy != 0:
                    #    continue
                    if iPT + dx < 0 or iPT + dx >= ratio.shape[0]:
                        continue
                    if iY + dy < 0 or iY + dy >= ratio.shape[1]:
                        continue
                    if ratio[iPT+dx, iY+dy] != 0:
                        if dx==0 and dy==0:
                            nNeighbors += 10
                            sumNeighbors += 10*ratio[iPT+dx, iY+dy]
                        elif dx * dy ==0:
                            nNeighbors += 2
                            sumNeighbors += 2*ratio[iPT+dx, iY+dy]
                        else:
                            nNeighbors += 1
                            sumNeighbors += ratio[iPT+dx, iY+dy]

            if nNeighbors > 0:
                ratio_smooth[iPT, iY] = sumNeighbors / nNeighbors
            else:
                ratio_smooth[iPT, iY] = 0
        else:
            ratio_smooth[iPT, iY] = ratio[iPT, iY]

corrMC = ratio_smooth * Hsignal


print("w.r.t. y:")
print(np.sum(Hdata_sub, axis=0)/np.sum(corrMC, axis=0))
print("w.r.t. pT:")
print(np.sum(Hdata_sub, axis=1)/np.sum(corrMC, axis=1))

#ratio = np.where(ratio >5, 5, ratio)
#ratio = np.where(ratio <0, 0, ratio)

#fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20, 5))
#im0 = ax0.imshow(Hdata, origin='lower', aspect='auto',
#                 norm=matplotlib.colors.LogNorm())
#im1 = ax1.imshow(Hsignal, origin='lower', aspect='auto',
#                 norm=matplotlib.colors.LogNorm())
#im2 = ax2.imshow(ratio, origin='lower', aspect='auto',
#                 norm=matplotlib.colors.LogNorm(0.1, 2.2))
#im3 = ax3.imshow(ratio_smooth, origin='lower', aspect='auto',
#                 norm=matplotlib.colors.LogNorm(0.1, 2.2))
#fig.colorbar(im0, ax=ax0)
#fig.colorbar(im1, ax=ax1)
#fig.colorbar(im2, ax=ax2)
#fig.colorbar(im3, ax=ax3)
#ax0.set_title("DATA")
#ax1.set_title("MC")
#ax2.set_title("DATA/MC")
#ax3.set_title("DATA/MC smoothed")
#ax0.set_xlabel("Z |y| bin")
#ax1.set_xlabel("Z |y| bin")
#ax2.set_xlabel("Z |y| bin")
#ax3.set_xlabel("Z |y| bin")
#
#ax0.set_ylabel("Z pT bin")
#plt.tight_layout()
#plt.show()

import correctionlib.schemav2 as cs

corr = cs.Correction(
    name="Zwt",
    version=1,
    description="Z kinematics data/MC weight",
    inputs=[
        cs.Variable(name="Zpt", type="real", description="Z pT [GeV]"),
        cs.Variable(name="Zy", type="real", description="Z |y|"),
    ],
    output=cs.Variable(name="weight", type="real", description="Z kinematics data/MC weight"),
    data = cs.MultiBinning(
        nodetype="multibinning",
        inputs=["Zpt", 'Zy'],
        edges = [edges_Zpt.tolist(), edges_Zy.tolist()],
        content = ratio.ravel().tolist(),
        flow='clamp'
    )
)
corr_ratioerr = cs.Correction(
    name="Zwt_err",
    version=1,
    description="Z kinematics data/MC weight uncertainty",
    inputs=[
        cs.Variable(name="Zpt", type="real", description="Z pT [GeV]"),
        cs.Variable(name="Zy", type="real", description="Z |y|"),
    ],
    output=cs.Variable(name="weight", type="real", description="Z kinematics data/MC weight uncertainty"),
    data = cs.MultiBinning(
        nodetype="multibinning",
        inputs=["Zpt", 'Zy'],
        edges = [edges_Zpt.tolist(), edges_Zy.tolist()],
        content = ratioerr.ravel().tolist(),
        flow='clamp'
    )
)

import rich
rich.print(corr)
print(edges_Zpt)
print(edges_Zy)

cset = cs.CorrectionSet(
        schema_version=2, 
        corrections=[corr]
)

print("RATIO")
print(ratio[20])
print("CORR/REAL")
print((Hreweighted/Hsignal)[20])
print()
print("Evaluated")
ratio_eval = corr.to_evaluator()
ratioerr_eval = corr_ratioerr.to_evaluator()
print(ratio_eval.evaluate(edges_Zpt[20]+1, edges_Zy+0.001))
print()
print()

with open("corrections/Zkin/Zwt.json", "w") as f:
    f.write(cset.json(exclude_unset=True, indent=4))

PTcenters = (edges_Zpt[1:] + edges_Zpt[:-1])/2
PTcenters[-1] = 5000
Ycenters = (edges_Zy[1:] + edges_Zy[:-1])/2
Y, PT = np.meshgrid(Ycenters, PTcenters)
PT = PT.reshape((1, -1))
Y = Y.reshape((1, -1))
xdata = np.vstack((PT, Y))
ratiovals = ratio_eval.evaluate(xdata[0], xdata[1])
ratioerrvals = ratioerr_eval.evaluate(xdata[0], xdata[1])

mask = ratiovals > 0
xdata = xdata[:, mask]
ratiovals = ratiovals[mask]
ratioerrvals = ratioerrvals[mask]

from scipy import interpolate
PTgrid, Ygrid = np.meshgrid(PTcenters, Ycenters, indexing='ij')
print(PTgrid.shape)
print(PTgrid[:,0])
print(Ygrid.shape)
print(Ygrid[0,:])
print(ratio.shape)
print(ratioerr.shape)

#ratioerr = np.nan_to_num(np.sqrt(err2data_sub)/Hdata_sub)

#ratioerr[Hdata_sub == 0] = 0
#ratioerr[Hsignal == 0] = 0
#ratioerr[ratio == 0] = 0
#ratioerr[ratio > 2] = 0
#ratioerr[ratio < 0.5] = 0

mask = (ratio == 0) | (Hdata_sub==0) | (Hsignal==0)

w = np.ones_like(ratio)

ratiogrid = ratio.copy()

PTgrid = PTgrid.ravel()
Ygrid = Ygrid.ravel()
ratio = ratio.ravel()
w= w.ravel()
mask = ~mask.ravel()

PTgrid = PTgrid[mask]
Ygrid = Ygrid[mask]
ratio = ratio[mask]
w = w[mask]

tck = interpolate.bisplrep(PTgrid, Ygrid, ratio,
                           kx=3, ky=3,
                           w=w, s=2)
interpvals = interpolate.bisplev(PTcenters, Ycenters, tck)
print(interpvals.shape)

fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3, 2, figsize=(10, 15))
ax0.set_title("DATA")
im0 = ax0.imshow(Hdata_sub, origin='lower', aspect='auto',
                 norm=matplotlib.colors.LogNorm())
ax1.set_title("MC")
im1 = ax1.imshow(Hsignal, origin='lower', aspect='auto',
                 norm=matplotlib.colors.LogNorm())
ax2.set_title("DATA/MC")
im2 = ax2.imshow(ratiogrid, origin='lower', aspect='auto',
                 norm=matplotlib.colors.Normalize(0, 2))
ax3.set_title("Interpolated")
im3 = ax3.imshow(interpvals, origin='lower', aspect='auto',
                 norm=matplotlib.colors.Normalize(0, 2))
corrMC_raw = ratiogrid * Hsignal
corrMC_interp = interpvals * Hsignal
ax4.set_title("DATA/(ratio*MC)")
im4 = ax4.imshow(Hdata_sub/corrMC_raw - 1, origin='lower', aspect='auto',
                 norm=matplotlib.colors.Normalize(-0.5, +0.5),
                 cmap='coolwarm')
ax5.set_title("DATA/(interp*MC)")
im5 = ax5.imshow(Hdata_sub/corrMC_interp - 1, origin='lower', aspect='auto',
                 norm=matplotlib.colors.Normalize(-0.5, +0.5),
                 cmap='coolwarm')

fig.colorbar(im1, ax=ax1)
fig.colorbar(im3, ax=ax3)
fig.colorbar(im5, ax=ax5)
plt.tight_layout()
plt.show()
asfdkladfskl



def polynomial(x, *pars):
    ans = 0
    for i, par in enumerate(pars):
        ans += par * np.power(x, i)
    return ans

def rational(x, pars_top, pars_bottom):
    return polynomial(x, *pars_top)/polynomial(x, *pars_bottom)

def make_rational(order_top, order_bottom):
    return lambda x, *pars: rational(x, pars[:order_top], 
                                     [1] + list(pars[order_top:]))

def double_rational(x, pars_top_PT, pars_bottom_PT,
                    pars_top_Y, pars_bottom_Y,
                    xy_top, xy_bottom):

    top_PT = polynomial(x[0], *pars_top_PT)
    bottom_PT = polynomial(x[0], 1, *pars_bottom_PT)
    top_Y = polynomial(x[1], *pars_top_Y)
    bottom_Y = polynomial(x[1], 1, *pars_bottom_Y)

    top = top_PT + top_Y
    bottom = bottom_PT + bottom_Y

    if xy_top is not None:
        top += xy_top * x[0] * x[1]
    if xy_bottom is not None:
        bottom += xy_bottom * x[0] * x[1]
    
    return top/bottom


def make_fitfunc(x, order_top_PT, order_bottom_PT,
                    order_top_Y, order_bottom_Y,
                    doXYtop, doXYbottom):
    s0 = 0
    s1 = order_top_PT
    s2 = s1 + order_bottom_PT
    s3 = s2 + order_top_Y
    s4 = s3 + order_bottom_Y

    fitfunc = lambda x, *pars: double_rational(x,
                                    pars[s0:s1], pars[s1:s2],
                                    pars[s2:s3], pars[s3:s4],
                                    pars[s4] if doXYtop else None,
                                    pars[s4+1] if doXYbottom else None)

    p0 = [0 for i in range(s4)]
    p0[s0] = 1
    p0[s2] = 1
    if doXYtop:
        p0 += [0]
    if doXYbottom:
        p0 += [0]

    return fitfunc, p0

from scipy.optimize import curve_fit


'''
for order_PT_top in [2, 3]:
    for order_PT_bot in [2, 3, 4]:
        for order_Y_top in [2,3]:
            for order_Y_bot in [2,3,4]:
                print(order_PT_top, order_PT_bot, order_Y_top, order_Y_bot)
                for doXY in [False, True]:
                    fitfunc, p0 = make_fitfunc(xdata, 
                                               order_PT_top, order_PT_bot,
                                               order_Y_top, order_Y_bot,
                                               doXY, doXY)
                    try:
                        popt, pcov = curve_fit(fitfunc, xdata, ratiovals, 
                                               p0=p0, sigma=ratioerrvals, 
                                               absolute_sigma=True)
                        pred = fitfunc(xdata, *popt)
                        residuals = (ratiovals - pred)/ratioerrvals
                        residual2 = np.square(residuals)
                        nDOF = len(ratiovals) - len(popt)
                        chi2 = np.sum(residual2)/nDOF
                        print("\tpopt:", popt)
                        print("\tchi2:", chi2)
                    except:
                        print("\tFailed")
'''

import numpy.polynomial.polynomial
fitfunc, p0 = make_fitfunc(xdata, 3, 4, 3, 4, True, True)
def make_polyfunc(order):
    fitfunc=lambda x, *pars : numpy.polynomial.polynomial.polyval2d(
                                x[0], x[1], 
                                np.asarray(pars).reshape((order, order)))
    p0 = np.zeros((order, order))
    p0[0,0] = 1

    return fitfunc, p0

fitfunc, p0 = make_polyfunc(5)
print(p0.shape)
popt, pcov = curve_fit(fitfunc, xdata, ratiovals, 
                       p0=p0, sigma=ratioerrvals, 
                       absolute_sigma=True)

for pt in [10, 100, 500]:
    xdata_test = np.zeros((2, 24))
    xdata_test[0, :] = pt
    xdata_test[1, :] = Ycenters
    print(xdata_test)
    truth = ratio_eval.evaluate(xdata_test[0], xdata_test[1])
    plt.scatter(Ycenters, truth)

    xdata_test_fine = np.zeros((2, 100))
    xdata_test_fine[0, :] = pt
    xdata_test_fine[1, :] = np.linspace(0, np.max(Ycenters), 100)
    plt.plot(xdata_test_fine[1], fitfunc(xdata_test_fine, *popt))
    plt.show()

