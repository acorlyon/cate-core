# The MIT License (MIT)
# Copyright (c) 2016, 2017 by the ESA CCI Toolbox development team and contributors
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

import json
import os.path
from _threading_local import local
from abc import ABCMeta
from typing import List, Tuple
from types import LambdaType

import xarray as xr
from xarray import Dataset
import pandas as pd

from cate.core.ds import DATA_STORE_REGISTRY, DataStore, DataSource
from cate.core.objectio import OBJECT_IO_REGISTRY, ObjectIO
from cate.core.op import OP_REGISTRY, op_input, op
from cate.util.monitor import Monitor
#from cate.ds.esa_cci_odp import EsaCciOdpDataStore
from cate.ds.local import LocalFilePatternDataStore


@op(tags=['input'])
@op_input('ds_name')
@op_input('start_date')
@op_input('end_date')
@op_input('sync')
@op_input('protocol')
def open_dataset(ds_name: str,
                 start_date: str = None,
                 end_date: str = None,
                 sync: bool = False,
                 protocol: str=None,
                 monitor: Monitor=Monitor.NONE) -> xr.Dataset:
    """
    Open a dataset from a data source identified by *ds_name*.

    :param ds_name: The name of data source.
    :param start_date: Optional start date of the requested dataset.
    :param end_date: Optional end date of the requested dataset.
    :param sync: Whether to synchronize local and remote data files before opening the dataset.
    :param protocol: Name of protocol used to access dataset
    :param monitor: a progress monitor, used only if *sync* is ``True``.
    :return: An new dataset instance.
    """
    import cate.core.ds
    return cate.core.ds.open_dataset(ds_name, start_date=start_date, end_date=end_date,
                                     protocol=protocol, sync=sync, monitor=monitor)


# noinspection PyShadowingBuiltins
@op(tags=['output'], no_cache=True)
@op_input('ds')
@op_input('file')
@op_input('format')
def save_dataset(ds: xr.Dataset, file: str, format: str = None):
    """
    Save a dataset to NetCDF file.

    :param ds: The dataset
    :param file: File path
    :param format: NetCDF format flavour, one of 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC'.
    """
    ds.to_netcdf(file, format=format)


# noinspection PyShadowingBuiltins
@op(tags=['input'])
@op_input('ds')
def create_local_synced_datasource(name: str, selected_variables: List[str]= None, time_range: Tuple[int, int] = None,
                                   lat_lon: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                                   local_datastore_name: str = None, local_path: str = None,
                                   compression_enabled: bool = True, compression_level: int = 9,
                                   monitor: Monitor = Monitor.NONE):
    """

    :param name:
    :param selected_variables:
    :param time_range:
    :param lat_lon:
    :param local_datastore_name:
    :param local_path:
    :param compression_enabled:
    :param compression_level:
    :param monitor:
    :return:
    """
    from xarray.backends.netCDF4_ import NetCDF4DataStore
    from cate.ds.local import get_data_store_path

    local_dstore = DATA_STORE_REGISTRY.get_data_store('local')  # type: LocalFilePatternDataStore
    if not isinstance(local_dstore, LocalFilePatternDataStore):
        raise TypeError("Couldn't find local data store")

    remote_dstore = DATA_STORE_REGISTRY.get_data_store('esa_cci_odp')  # type: DataStore
    if not isinstance(local_dstore, DataStore):
        raise TypeError("Couldn't find remote data store")

    remote_dsources = remote_dstore.query(name, monitor)
    if remote_dsources is None or len(remote_dsources) == 0:
        raise ValueError("Could't find specified data source: {}".format(name))
    remote_dsource = remote_dsources[0]  # type: DataSource

    if not local_path:
        local_path = get_data_store_path()

    if compression_enabled:
        encoding_update = {'zlib': True}
        if compression_level:
            encoding_update.setdefault('complevel', compression_level)

    local_dsources = local_dstore.query('local.{}'.format(local_datastore_name), monitor)  # type: [DataSource]
    if local_dsources:
        raise ValueError("Could't create specified local data source: {}".format(local_datastore_name))
    local_dstore.add_opendap_pattern(local_datastore_name, name, selected_variables, time_range, lat_lon)
    local_dsource = local_dstore.query('local.{}'.format(local_datastore_name), monitor)  # type: DataSource

    remote_datasets_uri = remote_dsource.get_datasets_uri(time_range=time_range, protocol='OPENDAP')

    for dataset_uri in remote_datasets_uri:  # type: str

        local_filepath = os.path.join(local_path, dataset_uri.rsplit('/', 1)[1])
        print(local_filepath)
        remote_netcdf = NetCDF4DataStore(dataset_uri)
        local_netcdf = NetCDF4DataStore(local_filepath, mode='w', persist=True)

        local_netcdf.set_attributes(remote_netcdf.get_attrs())

        if selected_variables is None:
            selected_variables = [var_name for var_name in remote_netcdf.variables.keys()]

        remote_dataset = xr.Dataset.load_store(remote_netcdf)

        if lat_lon:
            remote_dataset = remote_dataset.sel(drop=False,
                                                lat=slice(lat_lon[0][0], lat_lon[0][1]),
                                                lon=slice(lat_lon[1][0], lat_lon[1][1]))

        for sel_var_name in selected_variables:
            print(sel_var_name)
            var_dataset = remote_dataset.drop(
                [var_name for var_name in remote_dataset.variables.keys() if var_name != sel_var_name])
            if compression_enabled:
                var_dataset.variables.get(sel_var_name).encoding.update(encoding_update)
            local_netcdf.store_dataset(var_dataset)

        local_netcdf.sync()

        remote_netcdf.close()
        local_netcdf.close()

        local_dsource.add_dataset(local_filepath, local_netcdf.get('start_time'))

    local_dsource.save()

    return local_datastore_name


# noinspection PyShadowingBuiltins
@op(tags=['input'])
@op_input('file')
@op_input('format')
def read_object(file: str, format: str = None) -> object:
    """
    Read a data object from a file.

    :param file: The file path.
    :param format: Optional format name.
    :return: The data object.
    """
    import cate.core.objectio
    obj, _ = cate.core.objectio.read_object(file, format_name=format)
    return obj


# noinspection PyShadowingBuiltins
@op(tags=['output'], no_cache=True)
@op_input('obj')
@op_input('file')
@op_input('format')
def write_object(obj, file: str, format: str = None):
    """
    Read a data object from a file.

    :param file: The file path.
    :param format: Optional format name.
    :return: The data object.
    """
    import cate.core.objectio
    cate.core.objectio.write_object(obj, file, format_name=format)


@op(tags=['input'])
@op_input('file')
@op_input('encoding')
def read_text(file: str, encoding: str = None) -> str:
    """
    Read a string object from a text file.

    :param file: The text file path.
    :param encoding: Optional encoding, e.g. "utc-8".
    :return: The string object.
    """
    if isinstance(file, str):
        with open(file, 'r', encoding=encoding) as fp:
            return fp.read()
    else:
        # noinspection PyUnresolvedReferences
        return file.read()


@op(tags=['output'], no_cache=True)
@op_input('obj')
@op_input('file')
@op_input('encoding')
def write_text(obj: object, file: str, encoding: str = None):
    """
    Write an object as string to a text file.

    :param obj: The data object.
    :param file: The text file path.
    :param encoding: Optional encoding, e.g. "utc-8".
    """
    if isinstance(file, str):
        with open(file, 'w', encoding=encoding) as fp:
            fp.write(str(obj))
    else:
        # noinspection PyUnresolvedReferences
        return file.write(str(obj))


@op(tags=['input'])
@op_input('file')
@op_input('encoding')
def read_json(file: str, encoding: str = None) -> object:
    """
    Read a data object from a JSON text file.

    :param file: The JSON file path.
    :param encoding: Optional encoding, e.g. "utc-8".
    :return: The data object.
    """
    if isinstance(file, str):
        with open(file, 'r', encoding=encoding) as fp:
            return json.load(fp)
    else:
        return json.load(file)


@op(tags=['output'], no_cache=True)
@op_input('obj')
@op_input('file')
@op_input('encoding')
@op_input('indent')
def write_json(obj: object, file: str, encoding: str = None, indent: str = None):
    """
    Write a data object to a JSON text file. Note that the data object must be JSON-serializable.

    :param obj: A JSON-serializable data object.
    :param file: The JSON file path.
    :param encoding: Optional encoding, e.g. "utf-8".
    :param indent: indent used in the file, e.g. "  " (two spaces).
    """
    if isinstance(file, str):
        with open(file, 'w', encoding=encoding) as fp:
            json.dump(obj, fp, indent=indent)
    else:
        return json.dump(obj, file, indent=indent)


@op(tags=['input'])
@op_input('file')
def read_csv(file: str, **kwargs) -> pd.DataFrame:
    """
    Read comma-separated values from plain text csv file into Pandas DataFrame

    :param file: The csv file path.
    :param kwargs: Optional pandas.read_csv() parameters
    :return: The DataFrame object.
    """
    return pd.read_csv(file, **kwargs)


@op(tags=['input'])
@op_input('file')
@op_input('drop_variables')
@op_input('decode_cf')
@op_input('decode_times')
@op_input('engine')
def read_netcdf(file: str,
                drop_variables: str = None,
                decode_cf: bool = True, decode_times: bool = True,
                engine: str = None) -> xr.Dataset:
    """
    Read a dataset from a netCDF 3/4 or HDF file.

    :param file: The netCDF file path.
    :param drop_variables: List of variables to be dropped.
    :param decode_cf: Whether to decode CF attributes and coordinate variables.
    :param decode_times: Whether to decode time information (convert time coordinates to ``datetime`` objects).
    :param engine: Optional netCDF engine name.
    """
    return xr.open_dataset(file, drop_variables=drop_variables,
                           decode_cf=decode_cf, decode_times=decode_times, engine=engine)


@op(tags=['output'], no_cache=True)
@op_input('obj')
@op_input('file')
@op_input('engine')
def write_netcdf3(obj: xr.Dataset, file: str, engine: str = None):
    """
    Write a data object to a netCDF 3 file. Note that the data object must be netCDF-serializable.

    :param obj: A netCDF-serializable data object.
    :param file: The netCDF file path.
    :param engine: Optional netCDF engine to be used
    """
    obj.to_netcdf(file, format='NETCDF3_64BIT', engine=engine)


@op(tags=['output'], no_cache=True)
@op_input('obj')
@op_input('file')
@op_input('engine')
def write_netcdf4(obj: xr.Dataset, file: str, engine: str = None):
    """
    Write a data object to a netCDF 4 file. Note that the data object must be netCDF-serializable.

    :param obj: A netCDF-serializable data object.
    :param file: The netCDF file path.
    :param engine: Optional netCDF engine to be used
    """
    obj.to_netcdf(file, format='NETCDF4', engine=engine)


# noinspection PyAbstractClass
class TextObjectIO(ObjectIO):
    @property
    def description(self):
        return "Plain text format"

    @property
    def format_name(self):
        return 'TEXT'

    @property
    def filename_ext(self):
        return '.txt'

    @property
    def read_op(self):
        return OP_REGISTRY.get_op('read_text')

    @property
    def write_op(self):
        return OP_REGISTRY.get_op('write_text')

    def read_fitness(self, file):
        # Basically every object can be written to a text file: str(obj)
        return 1 if isinstance(file, str) and os.path.isfile(file) else 0

    def write_fitness(self, obj):
        return 1000 if isinstance(obj, str) else 1


# noinspection PyAbstractClass
class JsonObjectIO(ObjectIO):
    @property
    def description(self):
        return 'JSON format (plain text, UTF8)'

    @property
    def format_name(self):
        return 'JSON'

    @property
    def filename_ext(self):
        return '.json'

    @property
    def read_op(self):
        return OP_REGISTRY.get_op('read_json')

    @property
    def write_op(self):
        return OP_REGISTRY.get_op('write_json')

    def read_fitness(self, file):
        return 1 if isinstance(file, str) and os.path.isfile(file) else 0

    def write_fitness(self, obj):
        return 1000 if isinstance(obj, str) \
                       or isinstance(obj, float) \
                       or isinstance(obj, int) \
                       or isinstance(obj, list) \
                       or isinstance(obj, dict) else 0


class NetCDFObjectIO(ObjectIO, metaclass=ABCMeta):
    @property
    def filename_ext(self):
        return '.nc'

    def read_fitness(self, file):
        # noinspection PyBroadException
        try:
            dataset = self.read(file)
            dataset.close()
            return 100000
        except Exception:
            return -1

    def write_fitness(self, obj):
        # TODO (forman, 20160905): add support for numpy-like arrays
        return 100000 if isinstance(obj, xr.Dataset) or (hasattr(obj, 'to_netcdf') and callable(obj.to_netcdf)) \
            else 0


# noinspection PyAbstractClass
class NetCDF3ObjectIO(NetCDFObjectIO):
    @property
    def description(self):
        return "netCDF 3 file format, which fully supports 2+ GB files."

    @property
    def format_name(self):
        return 'NETCDF3'

    @property
    def read_op(self):
        return OP_REGISTRY.get_op('read_netcdf')

    @property
    def write_op(self):
        return OP_REGISTRY.get_op('write_netcdf3')


# noinspection PyAbstractClass
class NetCDF4ObjectIO(NetCDFObjectIO):
    @property
    def description(self):
        return "netCDF 4 file format (HDF5 file format, using netCDF 4 API features)"

    @property
    def format_name(self):
        return 'NETCDF4'

    @property
    def read_op(self):
        return OP_REGISTRY.get_op('read_netcdf')

    @property
    def write_op(self):
        return OP_REGISTRY.get_op('write_netcdf4')


OBJECT_IO_REGISTRY.object_io_list.extend([
    TextObjectIO(),
    JsonObjectIO(),
    NetCDF4ObjectIO(),
    NetCDF3ObjectIO()
])
