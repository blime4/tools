from contextlib import contextmanager
import torch
import io
import os
import copy

import time
import json
import hashlib
import yaml
from pathlib import Path

from hooktools.trace_pb2 import HookData, MetaData
from hooktools.trace_utils import from_array


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


class TracerBase(object):

    def __init__(self, config):
        """
        config : yaml type configuration
        """
        config = handle_config(config)

        self.log_dir = config.get('log_dir', "./tmp")
        self.tracer_name = config.get('trace_dir', "")
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

    def trace(self, epoch=-1, step=-1):
        """
        epoch : integer number of epoch
        step : integer number of step
        """
        if self.trace_mode == 0:
            return
        print("tracing !!!!")
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


class DumpPbFileTracer(TracerBase):

    def __init__(self, config):
        super().__init__(config)
        self.dump_pb_hook_options = config.get('dump_pb_hook_options', {})
        self.max_number_of_modules_in_a_single_pb_file = self.dump_pb_hook_options.get(
            'max_number_of_modules_in_a_single_pb_file', 5)
        if self.forward_hook:
            self.ForWardMetaData = MetaData()
            self.forward_number = 0
            self.save_forward_number = 0
        if self.backward_hook:
            self.BackwardMetaData = MetaData()
            self.backward_number = 0
            self.save_backward_number = 0
        self.current_save_path = self.log_dir

    def hook_forward_fn(self, module, input, output):
        super().hook_forward_fn(module, input, output)
        self._hook_forward_impl(module, input, output, mode="Forward")

    def hook_backward_fn(self, module, grad_input, grad_output):
        super().hook_backward_fn(module, grad_input, grad_output)
        self._hook_forward_impl(
            module, grad_input, grad_output, mode="Backward")

    def trace(self, epoch=-1, step=-1):
        return super().trace(epoch, step)

    def untrace(self):
        super().untrace()
        if self.forward_hook:
            self._save_and_reinit_metadata("Forward")
        if self.backward_hook:
            self._save_and_reinit_metadata("Backward")

    def _hook_forward_impl(self, module, input, output, mode="Forward"):
        hook_data = HookData()
        hook_data.module_name = str(module)

        if not self.only_input:
            if isinstance(output, torch.Tensor):
                output = [output]
            for out in output:
                self._serialize_hook_outputs(out, hook_data)

        if not self.only_output:
            for inp in input:  # type input is tuple
                if isinstance(inp, torch.Tensor):
                    inp = [inp]
                for i in inp:
                    self._serialize_hook_inputs(i, hook_data)

        self._set_current_save_path(mode)

        if mode == "Forward":
            forward_data = self.ForWardMetaData.datas.add()
            forward_data.CopyFrom(hook_data)
            self.forward_number = self.forward_number + 1
            number = self.forward_number % self.max_number_of_modules_in_a_single_pb_file
            self.ForWardMetaData.datas.append(forward_data)
            if number == 0:
                self._save_and_reinit_metadata(mode)
        else:
            backward_data = self.BackWardMetaData.datas.add()
            backward_data.CopyFrom(hook_data)
            self.backward_number = self.backward_number + 1
            number = self.backward_number % self.max_number_of_modules_in_a_single_pb_file
            self.BackWardMetaData.datas.append(backward_data)
            if number == 0:
                self._save_and_reinit_metadata(mode)

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

    def _save_and_reinit_metadata(self, mode="Forward"):
        if mode == "Forward":
            pb_file = os.path.join(self.current_save_path, "forward_metadata_" +
                                   str(self.save_forward_number).zfill(5) + ".pb")
            self.save_forward_number = self.save_forward_number + 1
            with open(pb_file, "wb+") as f:
                bytesAsString = self.ForWardMetaData.SerializeToString()
                f.write(bytesAsString)
                self.ForWardMetaData = MetaData()
        else:
            pb_file = os.path.join(self.current_save_path, "backward_metadata_" +
                                   str(self.save_backward_number).zfill(5) + ".pb")
            self.save_backward_number = self.save_backward_number + 1
            with open(pb_file, "wb+") as f:
                bytesAsString = self.BackWardMetaData.SerializeToString()
                f.write(bytesAsString)
                self.BackWardMetaData = MetaData()

class Tracer(object):

    def __init__(self, config):
        """
        config : yaml type configuration
        """
        with open(config, 'r', encoding='utf-8') as file:
            file_data = file.read()
            config = yaml.load(file_data, Loader=yaml.FullLoader)

        self.register_hooks = config.get('register_hooks', [])

        self.trace_fns = []
        self.untrace_fns = []
        
        if "dump_pb_hook" in self.register_hooks:
            self.dump_pb_hook = DumpPbFileTracer(config)
            self.trace_fns.append(self.dump_pb_hook.trace)
            self.untrace_fns.append(self.dump_pb_hook.untrace)
        if "check_nan_hook" in self.register_hooks:
            pass
        
        
            
    def trace(self, epoch=-1, step=-1):
        for trace_fn in self.trace_fns:
            trace_fn(epoch, step)
            
    def untrace(self):
        for untrace_fn in self.untrace_fns:
            untrace_fn()
    
    
