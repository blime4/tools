from contextlib import contextmanager
import torch
import io
import os
import copy

import time
import json
import hashlib
import yaml
import pickle

from hooktools.trace_pb2 import HookData, MetaData
from hooktools.utils import from_array, handle_config


class PickleHookData(object):

    def __init__(self, module, input, output):
        self.module_name = str(module)
        self.input = input
        self.output = output

    def __repr__(self) -> str:
        return f"module_name : {self.module_name}, input.type : {type(self.input)}, output.type : {type(self.output)},"


class TracerBase(object):

    def __init__(self, config):
        """
        config : yaml type configuration
        """
        config = handle_config(config)

        self.log_dir = config.get('log_dir', "./tmp")
        self.tracer_name = config.get('tracer_name', "")
        self.trace_mode = config.get('trace_mode', 0)
        # trace mode :
        # 0 : "NOTRACE"
        # 1 : "FORWARD"
        # 2 : "BACKWARD"
        # 3 : "ALLTRACE"
        self.only_input = config.get('only_input', False)
        self.only_output = config.get('only_output', False)
        self.timestamp = str(int(time.time()))
        self.forward_hook = self.trace_mode == 1 or self.trace_mode == 3
        self.backward_hook = self.trace_mode == 2 or self.trace_mode == 3

        if self.forward_hook:
            self.forward_log_path = os.path.join(
                self.log_dir, self.tracer_name + "_forward_hook_"+self.timestamp)
        if self.backward_hook:
            self.backward_log_path = os.path.join(
                self.log_dir, self.tracer_name + "_backward_hook_"+self.timestamp)

        self.epoch = -1
        self.step = -1

        self.register_hooks = config.get('register_hooks', [])
        self.current_save_path = self.log_dir

    def trace(self, epoch=-1, step=-1):
        """
        epoch : integer number of epoch
        step : integer number of step
        """
        if self.trace_mode == 0:
            return
        print("tracing !!!!, epoch=%d, step=%d" % (epoch, step))
        self.epoch = epoch
        self.step = step

        if self.forward_hook:
            self.forward_handle = torch.nn.modules.module.register_module_forward_hook(
                self.hook_forward_fn)
        if self.backward_hook:
            # self.backward_handle = torch.nn.modules.module.register_module_full_backward_hook(self.hook_backward_fn) # still have an RuntimeError : Module backward hook for grad_input is called before the grad_output one. This happens because the gradient in your nn.M odule flows to the Module’s input without passing through the Module’s output
            self.backward_handle = torch.nn.modules.module.register_module_backward_hook(
                self.hook_backward_fn)  # this api will work in torchvision==0.8.2, ==0.9.0

    def untrace(self):
        if self.trace_mode == 0:
            return
        if self.forward_hook:
            self.forward_handle.remove()
        if self.backward_hook:
            self.backward_handle.remove()
        self.epoch = -1
        self.step = -1

    def hook_forward_fn(self, module, input, output):
        pass

    def hook_backward_fn(self, module, grad_input, grad_output):
        pass
    
    def _set_current_save_path(self, mode="Forward"):
        self.current_save_path = self.forward_log_path if mode == "Forward" else self.backward_log_path
        if self.epoch != -1:
            self.current_save_path = os.path.join(
                self.current_save_path, "epoch" + str(self.epoch))
        if self.step != -1:
            self.current_save_path = os.path.join(
                self.current_save_path, "step" + str(self.step))
        if not os.path.exists(self.current_save_path):
            os.makedirs(self.current_save_path)


class DumpPbFileTracer(TracerBase):

    def __init__(self, config):
        super().__init__(config)
        self.dump_pb_hook_options = config.get('dump_pb_hook_options', {})
        self.max_number_of_modules_in_a_single_pb_file = self.dump_pb_hook_options.get(
            'max_number_of_modules_in_a_single_pb_file', 5)
        if self.forward_hook:
            self.ForwardMetaData = MetaData()
            self.forward_number = 0
            self.save_forward_number = 0
        if self.backward_hook:
            self.BackwardMetaData = MetaData()
            self.backward_number = 0
            self.save_backward_number = 0

    def hook_forward_fn(self, module, input, output):
        super().hook_forward_fn(module, input, output)
        self._hook_impl(module, input, output, mode="Forward")

    def hook_backward_fn(self, module, grad_input, grad_output):
        super().hook_backward_fn(module, grad_input, grad_output)
        self._hook_impl(
            module, grad_input, grad_output, mode="Backward")

    def trace(self, epoch=-1, step=-1):
        return super().trace(epoch, step)

    def untrace(self):
        super().untrace()
        if self.forward_hook:
            self._save_and_reinit_metadata("Forward")
        if self.backward_hook:
            self._save_and_reinit_metadata("Backward")

    def _hook_impl(self, module, input, output, mode="Forward"):
        hook_data = HookData()
        hook_data.module_name = str(module)

        if not self.only_input:
            if isinstance(output, torch.Tensor):
                self._serialize_hook_outputs(output, hook_data)
            elif isinstance(output, list):
                for out in output:
                    self._serialize_hook_outputs(out, hook_data)

        if not self.only_output:
            for inp in input:  # type input is tuple
                if isinstance(inp, list):
                    for i in inp:
                        self._serialize_hook_inputs(i, hook_data)
                if isinstance(inp, torch.Tensor):
                    self._serialize_hook_inputs(inp, hook_data)

        self._set_current_save_path(mode)

        if mode == "Forward":
            forward_data = self.ForwardMetaData.datas.add()
            forward_data.CopyFrom(hook_data)
            self.forward_number = self.forward_number + 1
            number = self.forward_number % self.max_number_of_modules_in_a_single_pb_file
            self.ForwardMetaData.datas.append(forward_data)
            if number == 0:
                self._save_and_reinit_metadata(mode)
        else:
            backward_data = self.BackwardMetaData.datas.add()
            backward_data.CopyFrom(hook_data)
            self.backward_number = self.backward_number + 1
            number = self.backward_number % self.max_number_of_modules_in_a_single_pb_file
            self.BackwardMetaData.datas.append(backward_data)
            if number == 0:
                self._save_and_reinit_metadata(mode)

    def _tensor_to_numpy(self, tensor):
        try:
            array = tensor.data.detach().cpu().numpy()
        except:
            array = tensor.data.detach().numpy()
        return array

    def _serialize_hook_inputs(self, input, hook_data):
        hook_inputs = hook_data.inputs.add()
        array = self._tensor_to_numpy(input)
        tensor = from_array(array)
        hook_inputs.tensor.CopyFrom(tensor)
        if input.grad_fn.__class__.__name__ != "NoneType":
            hook_inputs.grad_fn = input.grad_fn.__class__.__name__
        hook_data.inputs.append(hook_inputs)

    def _serialize_hook_outputs(self, output, hook_data):
        hook_outputs = hook_data.outputs.add()
        array = self._tensor_to_numpy(output)
        tensor = from_array(array)
        hook_outputs.tensor.CopyFrom(tensor)
        hook_outputs.grad_fn = output.grad_fn.__class__.__name__
        hook_data.outputs.append(hook_outputs)

    def _save_and_reinit_metadata(self, mode="Forward"):
        if mode == "Forward":
            pb_file = os.path.join(self.current_save_path, "forward_metadata_" +
                                   str(self.save_forward_number).zfill(6) + ".pb")
            self.save_forward_number = self.save_forward_number + 1
            with open(pb_file, "wb+") as f:
                bytesAsString = self.ForwardMetaData.SerializeToString()
                f.write(bytesAsString)
                self.ForwardMetaData = MetaData()
        else:
            pb_file = os.path.join(self.current_save_path, "backward_metadata_" +
                                   str(self.save_backward_number).zfill(6) + ".pb")
            self.save_backward_number = self.save_backward_number + 1
            with open(pb_file, "wb+") as f:
                bytesAsString = self.BackwardMetaData.SerializeToString()
                f.write(bytesAsString)
                self.BackwardMetaData = MetaData()


class DumpPickleFileTracer(TracerBase):

    def __init__(self, config):
        super().__init__(config)
        self.dump_pickle_hook_options = config.get(
            'dump_pickle_hook_options', {})
        if self.forward_hook:
            self.save_forward_number = 0
        if self.backward_hook:
            self.save_backward_number = 0

    def hook_forward_fn(self, module, input, output):
        super().hook_forward_fn(module, input, output)
        self._hook_impl(module, input, output, "Forward")

    def hook_backward_fn(self, module, grad_input, grad_output):
        super().hook_backward_fn(module, grad_input, grad_output)
        self._hook_impl(module, grad_input, grad_output, "Backward")

    def trace(self, epoch=-1, step=-1):
        return super().trace(epoch, step)

    def untrace(self):
        return super().untrace()

    def _hook_impl(self, module, input, output, mode="Forward"):
        if self.only_input:
            output = None
        if self.only_output:
            input = None
        self._set_current_save_path(mode)
        hook_data = PickleHookData(module, input, output)
        self._save_pickle_data(hook_data, mode=mode)
    
    def _save_pickle_data(self, data, mode="Forward"):
        if mode == "Forward":
            pkl_file = os.path.join(self.current_save_path, "forward_" +
                                    str(self.save_forward_number).zfill(6) + ".pkl")
            self.save_forward_number = self.save_forward_number + 1
            with open(pkl_file, "wb+") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        else:
            pkl_file = os.path.join(self.current_save_path, "backward_" +
                                    str(self.save_backward_number).zfill(6) + ".pkl")
            self.save_backward_number = self.save_backward_number + 1
            with open(pkl_file, "wb+") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                
    # TODO : (optional) save module_name to pkl_file map

class Tracer(object):

    def __init__(self, config):
        """
        config : yaml type configuration
        """
        config = handle_config(config)

        self.register_hooks = config.get('register_hooks', [])

        self.trace_fns = []
        self.untrace_fns = []

        if "dump_pb_hook" in self.register_hooks:
            self.dump_pb_hook = DumpPbFileTracer(config)
            self.trace_fns.append(self.dump_pb_hook.trace)
            self.untrace_fns.append(self.dump_pb_hook.untrace)
        if "dump_pickle_hook" in self.register_hooks:
            self.dump_pkl_hook = DumpPickleFileTracer(config)
            self.trace_fns.append(self.dump_pkl_hook.trace)
            self.untrace_fns.append(self.dump_pkl_hook.untrace)
        if "check_nan_hook" in self.register_hooks:
            pass

        print(config)

    def trace(self, epoch=-1, step=-1):
        for trace_fn in self.trace_fns:
            trace_fn(epoch, step)

    def untrace(self):
        try:
            for untrace_fn in self.untrace_fns:
                untrace_fn()
        except:
            pass
