import numpy as np
import os
from datetime import datetime, timedelta, date
import numpy.ma as ma
import time
import random
from numba import jit

########################################################################################################################
# Purpose: IPA.py performs the error estimation algorithm for producing the variance and length-scale ratio files.
# For a specific variable type, Typ, based on the chosen seasons in SeasonList and N% observations in Nlist a new .npy
# file is created, and stored under the BaseDir, with ErrorEstimation\Typ\Season\N\Data.
#
# Prerequisites:
#       - Seasonal coarse grid. BaseDir\Preprocessed\Season
#       - Innovations for entire season. BaseDir\Preprocessed\Innovations_Typ
#       - Seasonal Rossby radius. BaseDir\Preprocessed\Season
#
# Output:
#       - Background error variance, IPA_Var.npy
#       - Observation error variance, IPA_Obs.npy
#       - Background error length-scale ratio (wgt1), IPA_Lsr.npy
#       - Script diagnostics, time run, observations used, etc.
#
########################################################################################################################


def ipamain(basedir, typ):
    print('--' * 60)
    print('Running IPA.py script. \n'
          'Runs the Inner product analysis algorithm to produce the standard deviation, length-scale ratio and \n'
          'observation error variance files, on the coarse grid. Outputs.py will produce model grid and .jpg files to\n'
          'view the error estimation results.')
    print('Time: %s ' % datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    print('--' * 60)

    maxkm = 900
    km2d = 40000/360
    maxradius = maxkm/km2d
    s = time.time()

    @jit('Tuple((f8[:], f8[:,:], f8))(f8[:], f8[:], f8[:], f4, i8, i8)',
         parallel=True, forceobj=True)
    def time_loop(xti, yti, zti, i0, i1, dc):
        toti = np.array((0.0, 0.0), dtype=np.float64)
        vi = np.array((0.0, 0.0), dtype=np.float64)
        ei = np.array([(0.0, 0.0), (0.0, 0.0)], dtype=np.float64)
        for tt in range(0, dc):
            # Set array of 'today's' innovations

            xxi = xti[tt]
            yyi = yti[tt]
            zzi = zti[tt]
            rr = (xxi*xxi + yyi*yyi)
            v0idx = np.argmin(rr)
            idd = rr < maxradius * maxradius
            idd[v0idx] = False
            toti, ei = inner(rr[idd], zzi[idd], zzi[v0idx], ei, toti, i0, i1)
            vi[0] += zzi[v0idx]**2
            vi[1] += 1
        return toti, ei, vi[0]/vi[1]

    # Setup coarse grid model domain. ~30km resolution.
    latlim = [8, 32]
    lonlim = [45, 75]
    dlat = 0.274
    dlon = 0.3

    xedges, yedges, x2d, y2d = prepare_grid(latlim, lonlim, dlat, dlon)

    ycenter = 0.5 * (yedges[1:] + yedges[:-1])
    xcenter = 0.5 * (xedges[1:] + xedges[:-1])

    seasonlist = ["1-DJF", "2-MAM", "3-JJA", "4-SON"]

    a1 = 4.0
    size = []
    tottime = []
    dt = timedelta(hours=24)
    print('Functions and parameters defined. %4.2fs' % (time.time()-s))

    for Season in seasonlist:
        print('Now running for the %s season. %4.2fs' % (Season, time.time()-s))

        s1 = time.time()
        z = []
        y = []
        x = []

        if Season == '1-DJF':
            start = date(2013, 12, 1)
            end = date(2014, 3, 1)
        elif Season == '2-MAM':
            start = date(2014, 3, 1)
            end = date(2014, 6, 1)
        elif Season == '3-JJA':
            start = date(2014, 6, 1)
            end = date(2014, 9, 1)
        elif Season == '4-SON':
            start = date(2014, 9, 1)
            end = date(2014, 12, 1)
        else:
            end = 0
            start = 0
            exit(0)

        daycount = (end - start).days
        os.chdir('%s\\ErrorEstimation\\Preprocessed\\Innovations_%s' % (basedir, typ))

        random.seed(100)
        ze = []
        for t in range(daycount):
            day = start + t * dt
            # Open and save all innovations,
            # Observations chosen as a random 1/N % of the whole data set. Seeded with 100.

            dat = np.load('%i\\%04i-%02i-%02i.npz' % (day.year, day.year, day.month, day.day))

            z.append(dat['zi'][:, 0])
            x.append(dat['xi'])
            y.append(dat['yi'])
            ze.append(len(dat['zi'][:, 0]))
        size.append(np.sum(ze))
        os.chdir('../%s/' % Season)

        # Load coarse grid for domain and mask
        grid = np.load('coarse_grid_%s.npz' % typ)
        gridz = grid['zi']
        gridz = ma.masked_invalid(gridz)

        #  Load rossby radius in terms of the coarse grid.
        ross = np.load('rossby_%s.npy' % typ)
        ross = (ross / 1000) / km2d

        var, lsr, obs = np.array(np.zeros_like(gridz)), np.array(np.zeros_like(gridz)), np.array(np.zeros_like(gridz))
        print('Allocated arrays and created list of observations. %4.2fs' % (time.time()-s))

        # Loop for x and y, if the mask is true, skip this value. Produces VAR, LSR and OBS, for all non-masked elements
        for i, yj in enumerate(ycenter):
            if np.sum(gridz.mask[i, :]) == 99:  # If all values on this i are masked, skip the rest of the loop.
                continue

            xy = []
            yy = []
            zy = []

            for t in range(daycount):
                idx = (abs(y[t] - yj) <= maxradius)
                xy.append(x[t][idx])
                yy.append(y[t][idx])
                zy.append(z[t][idx])

            for j, xj in enumerate(xcenter):
                if np.sum(gridz.mask[i, j]) == 1:
                    continue

                a0 = ross[i, j]
                xi = []
                yi = []
                zi = []
                for t in range(0, daycount):  # 'Box Cut', removes all values more than 9 degrees in x or y.
                    idx = (abs(xy[t] - xj) <= maxradius)
                    if len(xy[t][idx]) <= 1:
                        continue
                    xi.append((xy[t][idx]-xj))
                    yi.append((yy[t][idx]-yj))
                    zi.append(zy[t][idx])
                count = len(xi)
                tot, e, v = time_loop(xi, yi, zi, a0, a1, count)  # Time loop, contains inner loop.
                mm = np.linalg.solve(e, np.asmatrix(tot).T)

                af = mm[0]/(mm[1]+mm[0])  # w1 = m0 / (m0+m1)
                mf = mm[1]+mm[0]          # V = m0 + m1
                var[i, j] = mf
                lsr[i, j] = af
                obs[i, j] = v - mf
        print('Produced Var, Lsr and Obs for %s. %4.2fs' % (Season, time.time()-s))

        lsr[lsr == 0] = np.nan
        var[var == 0] = np.nan
        obs[obs == 0] = np.nan

        tottime.append(time.time()-s1)
        os.chdir('%s\\ErrorEstimation\\%s\\%s' % (basedir, typ.upper(), Season))
        if os.path.isdir('Data') is False:
            os.makedirs('Data')
        print("Data\\IPA_Var.npy has been made" )
        np.save("Data\\IPA_Var.npy", var)
        print("Data\\IPA_Obs.npy has been made")
        np.save("Data\\IPA_Obs.npy", obs)
        print("Data\\IPA_Lsr.npy has been made")
        np.save("Data\\IPA_Lsr.npy", lsr)
        print('--' * 60)

    print('Finished producing error estimation values using IPA. Writing diagnostics. %4.2fs' % (time.time()-s))
    os.chdir('..\\')
    txt = open('IPA Diagnostics-Real.txt', 'w+')
    txt.write('    Time    |    Season    |    Obs    \n')

    for p2 in range(len(seasonlist)):
        txt.write('  %7.2f   |     %s    |    %i\n' % (tottime[p2], seasonlist[p2], int(size[p2])))
    txt.close()
    print('==' * 60)


@jit('Tuple((f8[:], f8[:], f8[:, :], f8[:, :]))(i8[:], i8[:], f8, f8)', forceobj=True)
def prepare_grid(latlim, lonlim, dla, dlo):
    # Vectorised coarse grid.
    sy = int((latlim[1] - latlim[0]) / dla)
    gridy = np.linspace(latlim[0], latlim[1], int(sy))
    sx = int((lonlim[1] - lonlim[0]) / dlo)
    gridx = np.linspace(lonlim[0], lonlim[1], int(sx))

    # 2D matrix coarse grid.
    x2 = np.zeros((len(gridx) - 1, len(gridy) - 1))
    for p in range(0, len(gridy) - 1, 1):
        x2[:, p] = gridx[:-1]
    x2 = np.transpose(x2)

    y2 = np.zeros((len(gridy) - 1, len(gridx) - 1))
    for p in range(0, len(gridx) - 1, 1):
        y2[:, p] = gridy[:-1]
    return gridx, gridy, x2, y2


@jit('Tuple((f8[:], f8[:,:]))(f8[:], f4[:], f8, f8[:,:], f8[:], f8, f8)',
     forceobj=False, parallel=True, fastmath=True)
def inner(d, z, v0, eo, toto, i0, i1):
    exp, sm = np.exp, np.sum
    if len(d) <= 1:
        return toto, eo
    ya = exp(-d / (2 * i0 * i0))
    yb = exp(-d / (2 * i1 * i1))

    eo[0, 0] += sm(ya * ya)
    eo[1, 0] += sm(ya * yb)
    eo[0, 1] += sm(ya * yb)
    eo[1, 1] += sm(yb * yb)

    toto[0] += v0*sm(ya * z)
    toto[1] += v0*sm(yb * z)

    return toto, eo
