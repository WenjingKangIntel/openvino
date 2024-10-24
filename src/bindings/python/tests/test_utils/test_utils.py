# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Function
from openvino.runtime.impl import Shape, Type
from openvino.runtime.impl.op import Parameter
import openvino.runtime.opset8 as ops


def get_test_function():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    relu = ops.relu(param)
    func = Function([relu], [param], "test")
    assert func is not None
    return func


def test_compare_functions():
    try:
        from openvino.test_utils import compare_functions
        func = get_test_function()
        status, msg = compare_functions(func, func)
        assert status
    except RuntimeError:
        print("openvino.test_utils.compare_functions is not available")
