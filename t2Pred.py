import numpy as np

class t2PredSegment:
    """
    To be used in t2PredSet
    coeffs: numpy 2d array
    mjd_start, mjd_end: MJD objects
    """
    def __init__(self, coeffs, mjd_start, mjd_end, disp_const=0.):
        self.coeffs = np.array(coeffs)
        self.mjd_start = MJD(mjd_start)
        self.mjd_end = MJD(mjd_end)
        self.disp_const = disp_const

class t2PredSet:
    """
    segments: list of t2PredSegment objects
    """
    def __init__(self, t2predfile="t2pred.dat"):
        pred = np.loadtxt(t2predfile, dtype=str, delimiter='$$$')

        self.nsegs = int(pred[0].split()[1])
        self.psrname = pred[2].split()[1]
        self.sitename = pred[3].split()[1]
        self.freq_start = float(pred[5].split()[1])
        self.freq_end = float(pred[5].split()[2])
        self.ncoeff_mjd = int(pred[7].split()[1])
        self.ncoeff_freq = int(pred[8].split()[1])
        self.segments = []
        self.segstarts = []

        lines_per_COEFFS = int(np.ceil(self.ncoeff_freq/3.))
        lines_per_time = lines_per_COEFFS * self.ncoeff_mjd

        fullsegments = np.split(pred[1:], self.nsegs)
        for fullsegment in fullsegments:
            mjd_start = MJD(fullsegment[3].split()[1])
            mjd_end = MJD(fullsegment[3].split()[2])
            disp_const = float(fullsegment[5].split()[1])
            coeff_lines = fullsegment[8:8+lines_per_time]
            split_coeff_lines = np.split(coeff_lines, self.ncoeff_mjd)
            chunk_list = []
            for chunk in split_coeff_lines:
                oneline = ''
                for line in chunk: oneline += line
                split_oneline = np.array(oneline.split()[1:])
                chunk_list.append(split_oneline.astype(float))
            t2pred_seg = t2PredSegment(chunk_list, mjd_start, mjd_end,\
                disp_const)
            self.segments.append(t2pred_seg)
            self.segstarts.append(t2pred_seg.mjd_start)

        self.segstarts = np.array(self.segstarts)
        self.mjd_start_set = self.segments[0].mjd_start
        self.mjd_end_set = self.segments[-1].mjd_end

    def find_segment(self, mjd):
        if mjd < self.segments[0].mjd_start:
            print "Warning: MJD falls outside prediction range"
            return self.segments[0]
        elif mjd > self.segments[-1].mjd_end:
            print "Warning: MJD falls outside prediction range"
            return self.segments[-1]
        else:
            for segment in self.segments:
                if mjd >= segment.mjd_start:
                    if mjd <= segment.mjd_end:
                        return segment

def get_phases(nbins, tres, freq, t2pred_set, start_offset=0.):
    """
    Returns an array of phases that correspond to the Tempo2 prediction info
    in t2pred_set.
    
    nbins: number of phase bins required
    tres: time resolution in seconds
    freq: observing frequency
    t2pred_set: a t2PredSet object containing Tempo2 prediction information
    start_offset: time (in days) added to the start of t2pred_set to offset the
        starting point of the output phase array
    """
    dtstarts = (t2pred_set.segstarts - t2pred_set.mjd_start_set).astype(float)
    dt_split = []

    if start_offset:
        next_start = start_offset
        dt_end = start_offset + nbins*tres/86400.
    else:
        next_start = dtstarts[0]
        dt_end = dtstarts[0] + (nbins-1)*tres/86400.

    for ii in range(t2pred_set.nsegs-1):
        if dtstarts[ii+1] < dt_end:
            seg = np.arange(next_start, dtstarts[ii+1], tres/86400.)
            dt_split.append(seg)
            try: next_start = seg[-1] + tres/86400.
            except: pass
        else:
            seg = np.arange(next_start, dt_end, tres/86400.)
            dt_split.append(seg)
            next_start = dt_end

    dt_split.append(np.arange(next_start, dt_end, tres/86400.))

    map_freq = -1.+2.*(freq-t2pred_set.freq_start)/\
        (t2pred_set.freq_end-t2pred_set.freq_start)
    phases = np.array([], dtype=float)
    for ii in range(t2pred_set.nsegs):
        dt_seg = dt_split[ii]
        if len(dt_seg):
            t2p_seg = t2pred_set.segments[ii]
            numerator = dt_seg - dtstarts[ii]
            denominator = float(t2p_seg.mjd_end - t2p_seg.mjd_start)
            map_mjd = -1.+2.*numerator/denominator
            phases = np.append(phases, chebval2d_fix(map_mjd, map_freq,\
                t2p_seg.coeffs) + t2p_seg.disp_const/(freq*freq))
    return phases

#def get_frequency(mjd, freq, t2pred_set):
#    seg = t2pred_set.find_segment(mjd)
#    map_mjd = -1. + 2.*(mjd - seg.mjd_start)/(seg.mjd_end - seg.mjd_start)
#    map_freq = -1. + 2.*(freq - t2pred_set.freq_start)/(t2pred_set.freq_end - t2pred_set.freq_start)
#    return chebval2d_fix(map_mjd, map_freq, seg.coeffs) * 2./(seg.mjd_end-seg.mjd_start) / 86400.

def chebval_fix(x, c, tensor=True):
    """
    This is just numpy's chebval, but it subtracts off half of the first
    coefficient.
    """
    c = np.array(c, ndmin=1, copy=1)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
       c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1 :
        c0 = c[0]
        c1 = 0
    elif len(c) == 2 :
        c0 = c[0]
        c1 = c[1]
    else :
        x2 = 2*x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1) :
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x - 0.5*c[0]

def chebval2d_fix(x, y, c):
    """
    This is just numpy's chebval2d, but it uses chebval_fix, which subtracts
    off half of the first coefficient.
    """
    try:
        x, y = np.array((x, y), copy=0)
    except:
        raise ValueError('x, y are incompatible')

    c = chebval_fix(x, c)
    c = chebval_fix(y, c, tensor=False)
    return c

### MJD is copy-pasted from TimeSeriesEditor.  It's quite possible I've
### changed it there and not here.

class MJD():
    """
    MJD class that contains the integer and fractional portions of the date
    separately to increase precision.
    """
    def __init__(self, day_int, day_frac=None):
        if day_frac is not None:
            self.day_int = int(day_int)
            self.day_frac = float(day_frac)
        elif type(day_int) is not str:
            if isinstance(day_int, MJD):
                self.day_int = day_int.day_int
                self.day_frac = day_int.day_frac
            else:
                self.day_int = int(day_int)
                self.day_frac = np.floor((day_int-np.floor(day_int))*1.e11)/\
                    (1.e11)
        else:
            val = eval(day_int)
            if type(val) is int:
                self.day_int = val
                self.day_frac = 0.0
            else:
                self.day_int = eval(day_int.split('.')[0])
                self.day_frac = eval('.'+day_int.split('.')[1])

    def add(self, time):
        if time < 0.0:
            return self.subtract(-time)
        elif self.day_frac + time < 1.0:
            return MJD(self.day_int, self.day_frac + time)
        else:
            int_add = int(time)
            frac_add = time - np.floor(time)
            return self.add_MJD(MJD(int_add, frac_add))

    def subtract(self, time):
        if time < 0.0:
            return self.add(-time)
        elif self.day_frac - time >= 0.0:
            return MJD(self.day_int, self.day_frac - time)
        else:
            int_sub = int(time)
            frac_sub = time - np.floor(time)
            return self.subtract_MJD(MJD(int_sub, frac_sub))

    def add_MJD(self, other_MJD):
        if self.day_frac + other_MJD.day_frac < 1.0:
            return MJD(self.day_int + other_MJD.day_int,\
                       self.day_frac + other_MJD.day_frac)
        else:
            return MJD(self.day_int + other_MJD.day_int + 1,\
                       self.day_frac + other_MJD.day_frac - 1.0)

    def __add__(self, days):
        if isinstance(days, MJD): return self.add_MJD(days)
        else: return self.add(days)

    def subtract_MJD(self, other_MJD):
        if self > other_MJD:
            if self.day_frac - other_MJD.day_frac >= 0.0:
                return MJD(self.day_int - other_MJD.day_int,\
                    self.day_frac - other_MJD.day_frac)
            else:
                return MJD(self.day_int - other_MJD.day_int - 1,\
                    self.day_frac - other_MJD.day_frac + 1.0)
        else:
            int_part = float(self.day_int - other_MJD.day_int)
            frac_part = self.day_frac - other_MJD.day_frac
            return int_part + frac_part

    def __sub__(self, days):
        if isinstance(days, MJD): return self.subtract_MJD(days)
        else: return self.subtract(days)

    def __rsub__(self, days):
        return days + -self

    def __neg__(self):
        if self.day_int >= 1:
            return -float(self)

    def as_float(self):
        return self.day_int + self.day_frac

    def show(self):
        return repr(self.day_int)+repr(self.day_frac)[1:]

    def __repr__(self):
        return self.show()

    def __float__(self):
        return self.as_float()

    def __eq__(self, other):
        other = MJD(other)
        return (self.day_int == other.day_int) and\
            (self.day_frac == other.day_frac)

    def __ne__(self, other):
        other = MJD(other)
        return (self.day_int != other.day_int) or\
            (self.day_frac != other.day_frac)

    def __lt__(self, other):
        other = MJD(other)
        if (self.day_int == other.day_int):
            return self.day_frac < other.day_frac
        else:
            return self.day_int < other.day_int

    def __gt__(self, other):
        other = MJD(other)
        if (self.day_int == other.day_int):
            return self.day_frac > other.day_frac
        else:
            return self.day_int > other.day_int

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

