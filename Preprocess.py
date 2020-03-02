import time
import os
import calendar
import numpy as np
import numpy.ma as ma
from scipy import ndimage
import scipy.interpolate as interpolate
from datetime import datetime, date, timedelta
from netCDF4 import Dataset
from numba import jit
from shutil import copyfile

########################################################################################################################
# Purpose: Preprocess.py, before running the IPA and H-L scripts we store the innovations in a uniform format. We also
# produce a coarse grid for the AS20 domain, the mask for this grid is determined based on where innovations do occur.
# We also prepare the Rossby radius following this coarse grid and mask.
#
# Prerequisites:
#       - BaseDir available.
#       - YYYYMMDDT0000Z_xxx_qc_BiasCorrfb_oo_qc_fdbk.nc, for the set date and variable under the BaseDir.
#       - rossy_radii.nc, from NEMOVar suite. (Original is in ocean\OPERATIONAL_SUITE_V5.3\...)
#       - Model run output. Bmod_sdv_mxl... and Bmod_sdv_wgt... (This is to format .nc files)
#       - Unzipped assimilated model runs. e.g. diaopfoam files (This is for depth gradient). UZ.sh script will unzip.
#       - BaseDir should contain ocean\OPERATIONAL_SUITE_V5.3\...
#
# Output: Stored innovations from Start to End-1 day. Coarse masked grid, Rossby radius and TGradient for Seasons.
#
########################################################################################################################


def preprocessmain(basedir, predir, var):
    print('--' * 60)
    print('Running Preprocess.py script.')
    print('Produces all required files for the error estimation scripts. Stores the innovations and coarse grid for '
          '"var" produces \nRossby radius and TGradient for each season, and makes any necessary subdirectories.')
    print('Time: %s ' % datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    print('--' * 60)

    # Setup coarse grid model domain. ~30km resolution.
    latlim = [8, 32]
    lonlim = [45, 75]
    dlat = 0.274
    dlon = 0.3

    xedges, yedges, x2d, y2d = prepare_grid(latlim, lonlim, dlat, dlon)

    startlist = []
    endlist = []
    start = date(2014, 1, 1)
    end = date(2015, 1, 1)
    daycount = (end - start).days
    dt = timedelta(hours=24)

    startlist.append(date(2013, 12, 1))
    startlist.append(date(2014, 3, 1))
    startlist.append(date(2014, 6, 1))
    startlist.append(date(2014, 9, 1))

    endlist.append(date(2014, 3, 1))
    endlist.append(date(2014, 6, 1))
    endlist.append(date(2014, 9, 1))
    endlist.append(date(2014, 12, 1))

    typ = '%s_qc_BiasCorrfb_oo_qc_fdbk' % var

    s1 = time.time()

    if os.path.isdir('%s\\ErrorEstimation\\Preprocessed' % basedir) is False:
        os.makedirs('%s\\ErrorEstimation\\Preprocessed' % basedir)
    os.chdir('%s\\ErrorEstimation\\Preprocessed' % basedir)

    if os.path.isdir('Innovations_%s\\2014' % var) is False:
        os.makedirs('Innovations_%s\\2014' % var)

    # Store the innovations for each day, to be accessed easier in later scripts.
    for t in range(daycount):
        day = start + t * dt
        if os.path.isdir('%s\\as20_obs_files~\\as20_obs_files\\%s' % (predir, start.year)):
            os.chdir('%s\\as20_obs_files~\\as20_obs_files\\%s' % (predir, start.year))
        else:
            print('"%s\\as20_obs_files~\\as20_obs_files\\%s" is not available.' % (predir, start.year))
            print('Please ensure correct feedback files are available under Base Directory.')
            exit(0)

        [lat, lon, obs] = read_observation(day, var)

        os.chdir('%s\\ErrorEstimation\\Preprocessed\\Innovations_%s\\2014' % (basedir, var))
        np.savez(('%s' % day), xi=lon, yi=lat, zi=obs)

    os.chdir('%s\\ErrorEstimation\\Preprocessed' % basedir)
    for Name in os.listdir('Innovations_%s\\2014' % var):
        if Name[5:7] == '12':
            if os.path.isdir('Innovations_%s\\2013' % var) is False:
                os.makedirs('Innovations_%s\\2013' % var)
            copyfile('Innovations_%s\\2014\\%s' % (var, Name), 'Innovations_%s\\2013\\%s' % (var, Name))

    if os.path.isdir('Innovations_%s\\2013' % var) is False:
        os.makedirs('Innovations_%s\\2013' % var)
    os.chdir('Innovations_%s\\2013' % var)
    for Name in os.listdir('.'):
        if Name[0:4] == '2013':
            os.remove(Name)
        else:
            os.rename(Name, Name.replace('2014', '2013'))

    print('Innovations have been successfully stored in ErrorEstimation\\Preprocessed\\Innovations_%s. %4.2f s'
          % (var, time.time() - s1))

    # Prepare Rossby radius.
    os.chdir(predir)
    r = Dataset('rossby_radii.nc', 'r')
    ross = np.array(r.variables['rossby_r'][:])
    rlat = np.array(r.variables['lat'][:])
    rlon = np.array(r.variables['lon'][:])
    r.close()

    rlat = rlat[np.where(np.array(ross) <= 200000)]
    rlon = rlon[np.where(np.array(ross) <= 200000)]
    ros = ross[np.where(ross <= 200000)]

    # Prepare TGradient dimensions and mask.
    os.chdir('rose_as20_assim_mo\\diaopfoam_files_2013_DEC\\dec')
    dia = Dataset('20131201T0000Z_diaopfoam_fcast.grid_T/20131201T0000Z_diaopfoam_fcast.grid_T.nc')
    depth = np.array(dia.variables['deptht_bounds'][:])
    mask = ma.getmask(ma.masked_equal(np.array(dia.variables['votemper'][0, :, :, :]), 0))
    dia.close()
    #

    # Make the seasonal coarse grids.
    for Start, End in zip(startlist, endlist):
        print('--' * 60)
        daycount = (End - Start).days
        innovationcount = 0
        count = 0
        dayidx = np.zeros(daycount + 1, np.int64)

        if Start.month == 12:
            season = '1-DJF'
        elif Start.month == 3:
            season = '2-MAM'
        elif Start.month == 6:
            season = '3-JJA'
        elif Start.month == 9:
            season = '4-SON'
        else:
            season = 'fail'

        os.chdir('%s\\ErrorEstimation\\Preprocessed' % basedir)

        for t in range(daycount):
            day = Start + t * dt
            cg = np.load('Innovations_%s\\%s\\%s.npz' % (typ[:3], day.year, day))
            dayidx[t] = innovationcount
            innovationcount += cg['xi'].size
        dayidx[daycount] = innovationcount

        sst = np.zeros((innovationcount, 1))
        lat = np.zeros(innovationcount)
        lon = np.zeros(innovationcount)
        daynum = np.zeros(innovationcount, np.int)

        for t in range(daycount):
            day = Start + t * dt

            cg = np.load('Innovations_%s\\%s\\%s.npz' % (typ[:3], day.year, day))
            x, y, z = cg['xi'], cg['yi'], cg['zi']

            datasize = x.size
            sst[count:count + datasize] = z
            lat[count:count + datasize] = y
            lon[count:count + datasize] = x
            daynum[count:count + datasize] = t
            count += datasize

        if os.path.isdir('%s' % season) is False:
            os.makedirs('%s' % season)
        os.chdir('%s' % season)
        latidx = np.digitize(lat, yedges) - 1
        lonidx = np.digitize(lon, xedges) - 1

        val = np.zeros((len(yedges) - 1, len(xedges) - 1), np.float)

        griddedvalues = griddata(daycount, dayidx, latidx.flatten(), lonidx.flatten(), sst.flatten(), val)
        zi = np.ma.masked_invalid(griddedvalues)

        zi2 = np.zeros([len(yedges) - 1, len(xedges) - 1])
        for i in range(0, len(yedges) - 1, 1):
            for j in range(0, len(xedges) - 1, 1):
                zi2[i, j] = np.mean(zi[i, j, :])
        zi2[zi2 == 0] = np.nan
        np.savez('coarse_grid_%s' % typ[:3], xi=xedges, yi=yedges, zi=zi2)
        print('Coarse grid have successfully been stored in ErrorEstimation\\Preprocessed\\%s. %4.2f s'
              % (season, time.time() - s1))

        # Make the seasonal Rossy Radius npy file.
        gridz = np.ma.masked_invalid(zi2)
        ross = interpolate.griddata((rlon, rlat), np.array(ros), (x2d, y2d), 'nearest')
        ross[gridz.mask == True] = 0
        np.save('rossby_%s' % var, ross)
        print('Rossby radius on the coarse grid has been stored successfully in ErrorEstimation\\Preprocessed\\%s. '
              '%4.2f s' % (season, time.time() - s1))

        # Make the seasonal TGradient (the depth profile for length-scale ratio model file).
        tgrad = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
        os.chdir('%s\\rose_as20_assim_mo' % predir)
        for t in range(daycount):
            day = Start + t*dt
            year = day.year
            month = calendar.month_abbr[day.month]
            yyyymmdd = '%i%02i%02i' % (year, day.month, day.day)
            diadir = 'diaopfoam_files_%i_%s\\%s\\' % (year, month.upper(), month.lower())

            mod = Dataset('%s%sT0000Z_diaopfoam_fcast.grid_T\\%sT0000Z_diaopfoam_'
                          'fcast.grid_T.nc' % (diadir, yyyymmdd, yyyymmdd), 'r')

            tmp = np.array(mod.variables['votemper'][0, :, :, :])
            mod.close()
            for x in range(tmp.shape[0]-1):
                tgrad[x, :, :] += 10*(tmp[x, :, :] - tmp[x+1, :, :])/(depth[x+1, 0]-depth[x, 0])

        tgrad = tgrad/daycount
        tgrad = abs(tgrad)
        tgrad[tgrad > 1.5] = 1.5
        tgrad[tgrad < 0.07] = 0.07
        for x in np.linspace(0, 10, 11):
            tval = np.where(tgrad[int(x), :, :] < 0.5)
            for i in range(0, len(tval[0])):
                tgrad[int(x), tval[0][i], tval[1][i]] = 0.5

        for i in range(tgrad.shape[1]):
            for j in range(tgrad.shape[2]):
                tgrad[:, i, j] = smooth_var_array(tgrad[:, i, j], 2)

        os.chdir('%s\\ErrorEstimation\\Preprocessed\\%s' % (basedir, season))

        tgrad = ma.masked_array(tgrad, mask)
        tgradi = np.ones_like(tgrad)
        if os.path.isfile('TGradient.nc') is False:
            copyfile('%s\\Bmod_sdv_wgt_rea_t.nc' % predir, 'TGradient.nc')
        tnc = Dataset('TGradient.nc', 'r+')
        tnc['wgt2'][:] = tgrad/1.5
        tnc['wgt1'][:] = (tgradi - tgrad/1.5)
        tnc.close()
        print('The depth gradient files have been produced successfully from model data. %4.2f s' % (time.time() - s1))

        # Make the sub-level directories for future runs.
        os.chdir('%s\\ErrorEstimation' % basedir)
        if os.path.isdir('%s' % var.upper()) is False:
            os.makedirs('%s' % var.upper())
        os.chdir('%s' % var.upper())

        if os.path.isdir('%s' % season) is False:
            os.makedirs('%s' % season)
        os.chdir('%s' % season)
        print('Required subdirectories successfully created. %4.2f s' % (time.time() - s1))
    print(60*'==')


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


@jit('(f8[:,:])(f8[:,:], i8)', forceobj=True)
def smooth_var_array(data, sigma):
    data[data == 0] = np.nan
    data = ma.masked_invalid(data)
    masks = 1 - data.mask.astype(np.float)
    wsum = ndimage.gaussian_filter(masks * data, sigma)
    gsum = ndimage.gaussian_filter(masks, sigma)

    gfilter = wsum / gsum

    gfilter = ma.masked_array(gfilter, mask=data.mask)

    return gfilter


@jit('Tuple((f8[:,:],f8[:,:],f8[:,:]))(i8, u1)', forceobj=True)
def read_observation(s, var):
    cwd = os.getcwd()
    ymd = s.strftime("%Y%m%d")
    name = '%sT0000Z_%s_qc_BiasCorrfb_oo_qc_fdbk.nc' % (ymd, var)

    data = Dataset(name, 'r')
    lat = data.variables['LATITUDE'][:]
    lon = data.variables['LONGITUDE'][:]
    obs = data.variables['%s_OBS' % var.upper()][:]
    mod = data.variables['%s_Hx' % var.upper()][:]
    obs = obs - mod
    data.close()

    os.chdir(cwd)
    return lat, lon, obs


@jit('(f8[:,:,:])(i8, i8[:], i8[:], i8[:], f8[:], f8[:,:])', forceobj=True)
def griddata(daycount, dayidx, latidx, lonidx, sst, gridmean):
    # Combines all the data into a 3D grid that has the size of the computational grid times the number of days.
    # The values at each entry is the average of all the values (minus the global mean) in one day that fall into
    # each grid cell.

    allgriddedvalues = np.zeros((gridmean.shape[0], gridmean.shape[1], daycount,))

    for day in range(daycount):
        valcount = np.zeros(gridmean.shape)
        for k in range(dayidx[day], dayidx[day + 1]):
            laidx2 = latidx[k]
            loidx2 = lonidx[k]

            allgriddedvalues[laidx2, loidx2, day] += sst[k]
            valcount[laidx2, loidx2] += 1
        valcount[valcount == 0] = 1
        allgriddedvalues[:, :, day] = allgriddedvalues[:, :, day] / valcount
        allgriddedvalues[:, :, day][np.isnan(gridmean)] = np.nan

    return allgriddedvalues
