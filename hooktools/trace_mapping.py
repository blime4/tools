# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

from .  import trace_pb2
# from trace_pb2 import trace_pb2.TensorProto
import numpy as np  # type: ignore

TENSOR_TYPE_TO_NP_TYPE = {
    int(trace_pb2.TensorProto.FLOAT): np.dtype('float32'),
    int(trace_pb2.TensorProto.UINT8): np.dtype('uint8'),
    int(trace_pb2.TensorProto.INT8): np.dtype('int8'),
    int(trace_pb2.TensorProto.UINT16): np.dtype('uint16'),
    int(trace_pb2.TensorProto.INT16): np.dtype('int16'),
    int(trace_pb2.TensorProto.INT32): np.dtype('int32'),
    int(trace_pb2.TensorProto.INT64): np.dtype('int64'),
    int(trace_pb2.TensorProto.BOOL): np.dtype('bool'),
    int(trace_pb2.TensorProto.FLOAT16): np.dtype('float16'),
    int(trace_pb2.TensorProto.DOUBLE): np.dtype('float64'),
    int(trace_pb2.TensorProto.COMPLEX64): np.dtype('complex64'),
    int(trace_pb2.TensorProto.COMPLEX128): np.dtype('complex128'),
    int(trace_pb2.TensorProto.UINT32): np.dtype('uint32'),
    int(trace_pb2.TensorProto.UINT64): np.dtype('uint64'),
    int(trace_pb2.TensorProto.STRING): np.dtype(np.object)
}

NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}

TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = {
    int(trace_pb2.TensorProto.FLOAT): int(trace_pb2.TensorProto.FLOAT),
    int(trace_pb2.TensorProto.UINT8): int(trace_pb2.TensorProto.INT32),
    int(trace_pb2.TensorProto.INT8): int(trace_pb2.TensorProto.INT32),
    int(trace_pb2.TensorProto.UINT16): int(trace_pb2.TensorProto.INT32),
    int(trace_pb2.TensorProto.INT16): int(trace_pb2.TensorProto.INT32),
    int(trace_pb2.TensorProto.INT32): int(trace_pb2.TensorProto.INT32),
    int(trace_pb2.TensorProto.INT64): int(trace_pb2.TensorProto.INT64),
    int(trace_pb2.TensorProto.BOOL): int(trace_pb2.TensorProto.INT32),
    int(trace_pb2.TensorProto.FLOAT16): int(trace_pb2.TensorProto.UINT16),
    int(trace_pb2.TensorProto.BFLOAT16): int(trace_pb2.TensorProto.UINT16),
    int(trace_pb2.TensorProto.DOUBLE): int(trace_pb2.TensorProto.DOUBLE),
    int(trace_pb2.TensorProto.COMPLEX64): int(trace_pb2.TensorProto.FLOAT),
    int(trace_pb2.TensorProto.COMPLEX128): int(trace_pb2.TensorProto.DOUBLE),
    int(trace_pb2.TensorProto.UINT32): int(trace_pb2.TensorProto.UINT32),
    int(trace_pb2.TensorProto.UINT64): int(trace_pb2.TensorProto.UINT64),
    int(trace_pb2.TensorProto.STRING): int(trace_pb2.TensorProto.STRING),
}

STORAGE_TENSOR_TYPE_TO_FIELD = {
    int(trace_pb2.TensorProto.FLOAT): 'float_data',
    int(trace_pb2.TensorProto.INT32): 'int32_data',
    int(trace_pb2.TensorProto.INT64): 'int64_data',
    int(trace_pb2.TensorProto.UINT16): 'int32_data',
    int(trace_pb2.TensorProto.DOUBLE): 'double_data',
    int(trace_pb2.TensorProto.COMPLEX64): 'float_data',
    int(trace_pb2.TensorProto.COMPLEX128): 'double_data',
    int(trace_pb2.TensorProto.UINT32): 'uint64_data',
    int(trace_pb2.TensorProto.UINT64): 'uint64_data',
    int(trace_pb2.TensorProto.STRING): 'string_data',
    int(trace_pb2.TensorProto.BOOL): 'int32_data',
}

# STORAGE_ELEMENT_TYPE_TO_FIELD = {
#     int(SequenceProto.TENSOR): 'tensor_values',
#     int(SequenceProto.SPARSE_TENSOR): 'sparse_tensor_values',
#     int(SequenceProto.SEQUENCE): 'sequence_values',
#     int(SequenceProto.MAP): 'map_values',
#     int(OptionalProto.OPTIONAL): 'optional_value'
# }

# OPTIONAL_ELEMENT_TYPE_TO_FIELD = {
#     int(OptionalProto.TENSOR): 'tensor_value',
#     int(OptionalProto.SPARSE_TENSOR): 'sparse_tensor_value',
#     int(OptionalProto.SEQUENCE): 'sequence_value',
#     int(OptionalProto.MAP): 'map_value',
#     int(OptionalProto.OPTIONAL): 'optional_value'
# }

# half_tolerance, float_tolerance
DEFAULT_TYPE_TOLERANCE = (1e-2, 1e-3,)

