import numpy as np
import os
import time
from datetime import datetime
import shutil
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from netCDF4 import Dataset
from numba import jit


def outputsmain(basedir, predir, typ, typelist):
    print('--' * 60)
    print('Running Outputs.py script.')
    print('After running the required error estimation scripts, the outputs .png files are computed. For coarse grid '
          'just the \n.png files will be made, for the model grid .nc and .png files are computed after applying '
          'interpolation.')
    print('Time: %s ' % datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    print('--' * 60)
    s = time.time()

    def cgrid2mgrid(variable, sigma, dst, gridx2, gridy2, sea):

        cwdi = os.getcwd()
        os.chdir(predir)

        mo = Dataset('S010_1d_20111130_20111201_gridT.nc', 'r')
        mo2 = Dataset('Bmod_sdv_mxl_rea_t.nc', 'r')
        ob = Dataset('%s_err_obs_sd.nc' % typ, 'r')

        lat = np.array(mo.variables['nav_lat'])
        lon = np.array(mo.variables['nav_lon'])
        mo.close()

        mod = mo2.variables['sdv_mxl'][:]
        mo2.close()

        omod = ob.variables['Err_rep_%s_sig' % typ.upper()][:]
        ob.close()

        os.chdir(cwdi)

        point = np.array((gridx2.flatten(), gridy2.flatten()))

        if dst[-12:-9] == 'Sdv':
            data = variable[~np.isnan(variable)]
            point = point[:, ~np.isnan(variable.flatten())]
            griddz = interpolate.griddata(point.T, data, (lon, lat), 'nearest')
            griddz[mod.mask[0, :, :] == True] = np.nan
            griddz = smooth_var_array(griddz, 6 * sigma)
            griddz = ma.masked_invalid(griddz)
            nnz = griddz[~np.isnan(griddz)]

            plt.figure()
            plt.title('Background error standard deviation')
            plt.pcolormesh(griddz, cmap='jet', vmin=ll, vmax=ul)
            plt.colorbar()
            plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(nnz, 5), np.percentile(nnz, 95)))
            plt.ylabel(label)
            plt.tight_layout()
            plt.savefig('%s.png' % (dst[:-3]))
            plt.close()

            shutil.copyfile('V:\\EEPrerequisites\\Bmod_sdv_mxl_rea_t.nc', 'Data\\%s' % dst)

            sdv = Dataset('Data\\%s' % dst, 'r+')
            sdv['sdv_mxl'][:] = griddz
            sdv.close()

        if dst[-12:-9] == 'Obs':
            data = variable[~np.isnan(variable)]
            point = point[:, ~np.isnan(variable.flatten())]
            griddz = interpolate.griddata(point.T, data, (lon, lat), 'nearest')
            griddz[omod.mask[0, :, :] == True] = np.nan
            griddz = smooth_var_array(griddz, 6 * sigma)
            griddz = ma.masked_invalid(griddz)
            nnz = griddz[~np.isnan(griddz)]

            plt.figure()
            plt.title('Observation error standard deviation')
            plt.pcolormesh(griddz, cmap='jet', vmin=0, vmax=0.5)
            plt.colorbar()
            plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(nnz, 5), np.percentile(nnz, 95)))
            plt.ylabel(label)
            plt.tight_layout()
            plt.savefig('%s.png' % (dst[:-3]))
            plt.close()

            shutil.copyfile('V:\\EEPrerequisites\\%s_err_obs_sd.nc' % typ, 'Data\\%s' % dst)

            sdv = Dataset('Data\\%s' % dst, 'r+')
            sdv['Err_rep_%s_sig' % typ.upper()][:] = np.sqrt(griddz)
            sdv.close()

        if dst[-12:-9] == 'Lsr':
            data = variable[~np.isnan(variable)]
            point = point[:, ~np.isnan(variable.flatten())]
            griddz = interpolate.griddata(point.T, data, (lon, lat), 'nearest')
            griddz[mod.mask[0, :, :] == True] = np.nan
            griddz = smooth_var_array(griddz, 6 * sigma)
            griddz = ma.masked_invalid(griddz)
            nnz = griddz[~np.isnan(griddz)]

            plt.figure()
            plt.title('Short length-scale ratio')
            plt.pcolormesh(griddz, cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(nnz, 5), np.percentile(nnz, 95)))
            plt.ylabel(label)
            plt.tight_layout()
            plt.savefig('%s.png' % (dst[:-3]))
            plt.close()

            w1 = griddz.copy()

            w1 = ma.masked_array(w1, mod.mask)
            w2 = ma.masked_array(1 - w1, mod.mask)

            p = Dataset('%s/ErrorEstimation/Preprocessed/%s/TGradient.nc' % (basedir, sea), 'r')
            w2t = w2 * (p['wgt2'][:])
            w1t = w1 * (p['wgt1'][:])
            p.close()

            wgt2 = w2t / (w2t + w1t)

            shutil.copyfile('V:\\EEPrerequisites\\Bmod_sdv_wgt_rea_t.nc', 'Data\\%s' % dst)

            r = Dataset('Data\\%s' % dst, 'r+')
            r['wgt2'][:] = np.sqrt(wgt2)
            r['wgt1'][:] = np.sqrt(1 - wgt2)
            r.close()

        return

    if typ == 'sst':
        label = 'Temperature'
        ll = 0.2
        ul = 0.8
    elif typ == 'sla':
        label = 'Sea level anomoly'
        ll = 0
        ul = 0.3
    else:
        label = 'False'
        ll = 0
        ul = 1

    # Load coarse grid for domain and mask
    os.chdir('%s\\ErrorEstimation\\Preprocessed\\1-DJF' % basedir)
    grid = np.load('coarse_grid_%s.npz' % typ)
    gridz1 = grid['zi']
    gridz1 = ma.masked_invalid(gridz1)
    mask = gridz1.mask

    # Setup coarse grid model domain. ~30km resolution.
    latlim = [8, 32]
    lonlim = [45, 75]
    dlat = 0.274
    dlon = 0.3

    xedges, yedges, x2d, y2d = prepare_grid(latlim, lonlim, dlat, dlon)

    seasonlist = ['1-DJF', '2-MAM', '3-JJA', '4-SON']
    print('Beginning to produce .png and .nc files for model and coarse grid outputs, for all options in: \n%s with %s.'
          ' %2.2fs' % (seasonlist, typelist, time.time()-s))

    for Type in typelist:
        for Season in seasonlist:
            os.chdir('%s\\ErrorEstimation\\%s\\%s' % (basedir, typ.upper(), Season))

            if os.path.isfile('Data\\%s_Lsr.npy' % Type):
                lsr = np.load('Data\\%s_Lsr.npy' % Type)
                lsr2 = lsr[~np.isnan(lsr)]

                plt.figure()
                plt.pcolormesh(xedges, yedges, lsr, cmap='jet', vmin=0, vmax=1)
                plt.title('Length-scale ratio (wgt1)')
                plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(lsr2, 5), np.percentile(lsr2, 95)))
                plt.colorbar()
                plt.tight_layout()
                plt.savefig('%s_Lsr.png' % Type)
                plt.close()

                lsr[np.isnan(lsr)] = 0.0
                lidd1 = np.where(lsr <= 0.0)
                lidd2 = np.where(lsr > 1)
                lsr[lidd1] = np.nan
                lsr[lidd2] = np.nan
                x2 = lsr[~np.isnan(lsr)]
                points2 = np.zeros((2, len(x2)))
                points = np.array((x2d.flatten(), y2d.flatten()))
                points2[0, :] = points[0, (~np.isnan(lsr)).flatten()]
                points2[1, :] = points[1, (~np.isnan(lsr)).flatten()]

                limlsr = interpolate.griddata(points2.T, x2.flatten(), (x2d, y2d), 'nearest')

                limlsr[mask == True] = np.nan

                cgrid2mgrid(limlsr, 1.2, '%s_Lsr_Model.nc' % Type, x2d, y2d, Season)
                sink = '\\\\POFCDisk1\\PhD_Lewis\\ocean\\OPERATIONAL_SUITE_V5.3\\AS20v28\\errorcovs_%s\\others\\' \
                       '%s_Lsr_%s.nc' % (Season[2:].lower(), Type, typ)
                shutil.copyfile('Data\\%s_Lsr_Model.nc' % Type, sink)
                print('Model error length-scale ratio for %s method in %s finish. \n%s. \n%2.2fs'
                      % (Type, Season, sink, time.time()-s))

            if os.path.isfile('Data\\%s_Var.npy' % Type):
                var = np.load('Data\\%s_Var.npy' % Type)
                sdv = var.copy()
                sdv[np.isnan(sdv)] = 0.0
                sdv[np.where(sdv <= 0.0)] = np.nan
                sdv = np.sqrt(sdv)
                var2 = var[~np.isnan(var)]

                plt.figure()
                plt.pcolormesh(xedges, yedges, sdv, cmap='jet', vmin=ll, vmax=ul)
                plt.title('Standard Deviation')
                plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(sdv[~np.isnan(sdv)], 5),
                                                          np.percentile(sdv[~np.isnan(sdv)], 95)))
                plt.colorbar()
                plt.tight_layout()
                plt.savefig('%s_Sdv.png' % Type)
                plt.close()

                numval = len(var2)

                limvar = var.copy()
                lim = sorted(range(len(var2)), key=lambda m: var2[m], reverse=True)[:int(2 * numval / 100)]
                lim += sorted(range(len(var2)), key=lambda m: var2[m], reverse=False)[:int(2 * numval / 100)]

                for j in lim:
                    limvar[var == var2[j]] = np.nan
                limstd = np.sqrt(limvar)
                y2 = limstd[~np.isnan(limstd)]
                points = np.array((x2d.flatten(), y2d.flatten()))
                points2 = np.zeros((2, len(y2)))
                points2[0, :] = points[0, (~np.isnan(limstd)).flatten()]
                points2[1, :] = points[1, (~np.isnan(limstd)).flatten()]

                limstd = interpolate.griddata(points2.T, y2.flatten(), (x2d, y2d), 'nearest')

                limstd[mask == True] = np.nan

                cgrid2mgrid(limstd, 1.2, '%s_Sdv_Model.nc' % Type, x2d, y2d, Season)
                sink = '\\\\POFCDisk1\\PhD_Lewis\\ocean\\OPERATIONAL_SUITE_V5.3\\AS20v28\\errorcovs_%s\\others\\' \
                       '%s_Lsr_%s.nc' % (Season[2:].lower(), Type, typ)
                shutil.copyfile('Data\\%s_Sdv_Model.nc'  % Type, sink)
                print('Model error standard deviation for %s method in %s finish. \n%s. \n%2.2fs'
                      % (Type, Season, sink, time.time()-s))

            if os.path.isfile('Data\\%s_Obs.npy' % Type):
                obs = np.load('Data\\%s_Obs.npy' % Type)
                obs2 = obs[~np.isnan(obs)]
                sdv = obs.copy()
                sdv[np.isnan(sdv)] = 0.0
                sdv[np.where(sdv <= 0.0)] = np.nan
                sdv = np.sqrt(sdv)

                plt.figure()
                plt.pcolormesh(xedges, yedges, sdv, cmap='jet', vmin=0.1, vmax=0.5)
                plt.title('Standard Deviation')
                plt.xlabel('5%% = %3.5f, 95%% = %3.5f' % (np.percentile(sdv[~np.isnan(sdv)], 5),
                                                          np.percentile(sdv[~np.isnan(sdv)], 95)))
                plt.colorbar()
                plt.tight_layout()
                plt.savefig('%s_Obs.png' % Type)
                plt.close()

                numval = len(obs2)

                limobs = obs.copy()
                lim = sorted(range(len(obs2)), key=lambda m: obs2[m], reverse=True)[:int(2 * numval / 100)]
                lim += sorted(range(len(obs2)), key=lambda m: obs2[m], reverse=False)[:int(2 * numval / 100)]

                for j in lim:
                    limobs[obs == obs2[j]] = np.nan

                limstd = np.sqrt(limobs)
                y2 = limstd[~np.isnan(limstd)]
                points = np.array((x2d.flatten(), y2d.flatten()))
                points2 = np.zeros((2, len(y2)))
                points2[0, :] = points[0, (~np.isnan(limstd)).flatten()]
                points2[1, :] = points[1, (~np.isnan(limstd)).flatten()]

                limstd = interpolate.griddata(points2.T, y2.flatten(), (x2d, y2d), 'nearest')

                limstd[mask == True] = np.nan

                cgrid2mgrid(limstd, 1.2, '%s_Obs_Model.nc' % Type, x2d, y2d, Season)
                sink = '\\\\POFCDisk1\\PhD_Lewis\\ocean\\OPERATIONAL_SUITE_V5.3\\AS20v28\\errorcovs_%s\\others\\' \
                       '%s_Lsr_%s.nc' % (Season[2:].lower(), Type, typ)
                shutil.copyfile('Data\\%s_Obs_Model.nc' % Type, sink)
                print('Observation error standard deviation for %s method in %s finish. \n%s. \n%2.2fs'
                      % (Type, Season, sink, time.time()-s))
                print(60*'--')


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


@jit('(f8[:, :])(f8[:, :], f8)', forceobj=True)
def smooth_var_array(data, sigma):

    data[data == 0] = np.nan
    data = ma.masked_invalid(data)
    masks = 1 - data.mask.astype(np.float)
    wsum = ndimage.gaussian_filter(masks*data, sigma)
    gsum = ndimage.gaussian_filter(masks,      sigma)
    gsum[gsum == 0] = np.nan
    gfilter = wsum/gsum

    gfilter = ma.masked_array(gfilter, mask=data.mask)

    return gfilter
