# The MIT License (MIT)
# Copyright (c) 2016 by the ECT Development Team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Description
===========

CLI/API data visualization operations

Components
==========

## plot_map

Plots a geospatial data slice on a world map. General python prompt usage:

```python
import matplotlib.pyplot as plt
from ect.ops.plot import plot_map

plot_map(dset)
plt.show()
```

General jupyter notebook usage:
```python
import matplotlib.pyplot as plt
from ect.ops.plot import plot_map

%matplotlib inline
plot_map(dset)
```
If a file path is given, the plot is saved.
Supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff

"""
import matplotlib

matplotlib.use('agg')
# https://github.com/matplotlib/matplotlib/issues/3466/#issuecomment-213678376
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

from ect.core.op import op_input, op


@op(tags=['graphical', 'plot', 'map'])
@op_input('ds', description="A dataset from which to create the plot")
@op_input('variable', description="The geophysical quantity (dataset variable) to plot")
@op_input('time', description="Point in time to plot")
@op_input('lat_min', description="Minimum latitude extent to plot")
@op_input('lat_max', description="Maximum latitude extent to plot")
@op_input('lon_min', description="Minimum longitude extent to plot")
@op_input('lon_max', description="Maximum longitude extent to plot")
@op_input('path', description="Full path where to save the plot")
def plot_map(ds: xr.Dataset,
             variable: str = None,
             time=None,
             lat_min: float = None,
             lat_max: float = None,
             lon_min: float = None,
             lon_max: float = None,
             path: str = None) -> None:
    """
    Plot the given variable from the given dataset on a map with coastal lines.
    In case no variable name is given, the first encountered variable in the dataset
    is plotted. In case no time index is given, the first time slice is taken. It is
    also possible to set extents of the plot. If no extents are given, a global plot
    is created.

    The plot can either be shown using pyplot functionality, or saved, if a path is given.
    The following file formats for saving the plot are supported:
    eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff

    :param ds: xr.Dataset to plot
    :param variable: variable name in the dataset to plot
    :param time: time slice index to plot
    :param lat_min: minimum latitude extent to plot
    :param lat_max: maximum latitude extent to plot
    :param lon_min: minimum longitude extent to plot
    :param lon_max: maximum longitude extent to plot
    :param path: path where to save the plot
    """
    if not isinstance(ds, xr.Dataset):
        raise NotImplementedError('Only raster datasets are currently supported')

    if not variable:
        for key in ds.data_vars.keys():
            variable = key
            break

    if not time:
        time = 0

    # Sanity check
    if lat_min is None:
        lat_min = -90.0

    if lat_max is None:
        lat_max = 90.0

    if lon_min is None:
        lon_min = -180.0

    if lon_max is None:
        lon_max = 180.0

    if not _extents_sane(lat_min, lat_max, lon_min, lon_max):
        raise ValueError('Provided plot extents do not form a valid bounding box\
                within [-90.0,90.0,-180.0,180.0]')

    extents = [lon_min, lon_max, lat_min, lat_max]

    array_slice = ds[variable].isel(time=time)
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extents, ccrs.PlateCarree())

    ax.coastlines()
    array_slice.plot.contourf(ax=ax, transform=ccrs.PlateCarree())

    if path:
        fig.savefig(path)


def _extents_sane(lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    """ 
    Check if the provided [lat_min, lat_max, lon_min, lon_max] extents
    are sane.
    """
    if lat_min >= lat_max:
        return False

    if lon_min >= lon_max:
        return False

    if lat_min < -90.0:
        return False

    if lat_max > 90.0:
        return False

    if lon_min < -180.0:
        return False

    if lon_max > 180.0:
        return False

    return True