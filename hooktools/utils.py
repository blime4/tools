from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import os
import sys
import yaml
from pathlib import Path

from hooktools.trace_pb2 import TensorProto
import hooktools.trace_mapping as trace_mapping

from six import text_type

def convert_endian(tensor):  # type: (TensorProto) -> None
    """
    call to convert endianess of raw data in tensor.
    @params
    TensorProto: TensorProto to be converted.
    """
    tensor_dtype = tensor.data_type
    np_dtype = trace_mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]
    tensor.raw_data = np.frombuffer(tensor.raw_data, dtype=np_dtype).byteswap().tobytes()

def combine_pairs_to_complex(fa):  # type: (Sequence[int]) -> Sequence[np.complex64]
    return [complex(fa[i * 2], fa[i * 2 + 1]) for i in range(len(fa) // 2)]

def from_array(arr, name=None):  # type: (np.ndarray[Any], Optional[Text]) -> TensorProto
    """Converts a numpy array to a tensor def.

    Inputs:
        arr: a numpy array.
        name: (optional) the name of the tensor.
    Returns:
        tensor_def: the converted tensor def.
    """
    tensor = TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    if arr.dtype == np.object:
        # Special care for strings.
        tensor.data_type = trace_mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
        # TODO: Introduce full string support.
        # We flatten the array in case there are 2-D arrays are specified
        # We throw the error below if we have a 3-D array or some kind of other
        # object. If you want more complex shapes then follow the below instructions.
        # Unlike other types where the shape is automatically inferred from
        # nested arrays of values, the only reliable way now to feed strings
        # is to put them into a flat array then specify type astype(np.object)
        # (otherwise all strings may have different types depending on their length)
        # and then specify shape .reshape([x, y, z])
        flat_array = arr.flatten()
        for e in flat_array:
            if isinstance(e, text_type):
                tensor.string_data.append(e.encode('utf-8'))
            elif isinstance(e, np.ndarray):
                for s in e:
                    if isinstance(s, text_type):
                        tensor.string_data.append(s.encode('utf-8'))
                    elif isinstance(s, bytes):
                        tensor.string_data.append(s)
            elif isinstance(e, bytes):
                tensor.string_data.append(e)
            else:
                raise NotImplementedError(
                    "Unrecognized object in the object array, expect a string, or array of bytes: ", str(type(e)))
        return tensor
    
    # For numerical types, directly use numpy raw bytes.
    try:
        dtype = trace_mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
    except KeyError:
        raise RuntimeError(
            "Numpy data type not understood yet: {}".format(str(arr.dtype)))
    tensor.data_type = dtype
    tensor.raw_data = arr.tobytes()  # note: tobytes() is only after 1.9.
    if sys.byteorder == 'big':
        # Convert endian from big to little
        convert_endian(tensor)

    return tensor

def to_array(tensor):  # type: (TensorProto) -> np.ndarray[Any]
    """Converts a tensor def object to a numpy array.

    Inputs:
        tensor: a TensorProto object.
    Returns:
        arr: the converted array.
    """
    if tensor.HasField("segment"):
        raise ValueError(
            "Currently not supporting loading segments.")
    if tensor.data_type == TensorProto.UNDEFINED:
        raise TypeError("The element type in the input tensor is not defined.")

    tensor_dtype = tensor.data_type
    np_dtype = trace_mapping.TENSOR_TYPE_TO_NP_TYPE[tensor_dtype]
    storage_type = trace_mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[tensor_dtype]
    storage_np_dtype = trace_mapping.TENSOR_TYPE_TO_NP_TYPE[storage_type]
    storage_field = trace_mapping.STORAGE_TENSOR_TYPE_TO_FIELD[storage_type]
    dims = tensor.dims

    if tensor.data_type == TensorProto.STRING:
        utf8_strings = getattr(tensor, storage_field)
        ss = list(s.decode('utf-8') for s in utf8_strings)
        return np.asarray(ss).astype(np_dtype).reshape(dims)

    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        if sys.byteorder == 'big':
            # Convert endian from little to big
            convert_endian(tensor)
        return np.frombuffer(
            tensor.raw_data,
            dtype=np_dtype).reshape(dims)
    else:
        data = getattr(tensor, storage_field),  # type: Sequence[np.complex64]
        if (tensor_dtype == TensorProto.COMPLEX64
                or tensor_dtype == TensorProto.COMPLEX128):
            data = combine_pairs_to_complex(data)
        # F16 is stored as int32; Need view to get the original value
        if tensor_dtype == TensorProto.FLOAT16:
            return (
                np.asarray(
                    tensor.int32_data,
                    dtype=np.uint16)
                .reshape(dims)
                .view(np.float16))
        # Otherwise simply use astype to convert; e.g., int->float, float->float
        else:
            return (
                np.asarray(
                    data,
                    dtype=storage_np_dtype)
                .astype(np_dtype)
                .reshape(dims)
            )
            
def check_suffix(file="demo.yaml", suffix=('.yaml,'), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def handle_config(config):
    if isinstance(config, str):
        check_suffix(config, suffix=('.yaml', '.yml'))
        with open(config, 'r', encoding='utf-8') as file:
            file_data = file.read()
            config = yaml.load(file_data, Loader=yaml.FullLoader)

    assert isinstance(config, dict), f"unacceptable cofiguration! "
    return config

def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj
    
def get_file_list(path, endswith="pkl"):
    return [os.path.join(path, pb) for pb in sorted(os.listdir(path)) if pb.endswith(endswith)]
