import torch
import os
import time
import json
from hooktools.utils import handle_config, is_non_nn_module, NewHookData
from hooktools.hacker import Hacker

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
            print("save directory :  forward_log_path : ", self.forward_log_path)
        if self.backward_hook:
            self.backward_log_path = os.path.join(
                self.log_dir, self.tracer_name + "_backward_hook_"+self.timestamp)
            print("save directory :  backward_log_path : ",
                  self.backward_log_path)

        self.epoch = -1
        self.step = -1

        self.register_hooks = config.get('register_hooks', [])
        self.current_save_path = self.log_dir

        hacker_non_nn_modules = config.get("hacker_non_nn_modules", False)
        self.has_hacker = False
        if hacker_non_nn_modules:
            self.hacker = Hacker(config)
            self.has_hacker = True

        self.module_name_2_file_path = {}

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

    def _save_module_name_file_path_map(self, module_name, file_path):
        module_name = str(module_name)
        file_path = str(file_path)
        if module_name not in self.module_name_2_file_path.keys():
            self.module_name_2_file_path[module_name] = []
            self.module_name_2_file_path[module_name].append(file_path)
        else:
            self.module_name_2_file_path[module_name].append(file_path)

        map_file = os.path.join(
            self.log_dir, "module2path_"+self.timestamp+".json")
        with open(map_file, "w+", encoding='utf-8') as file:
            file.write(json.dumps(
                self.module_name_2_file_path, ensure_ascii=False))


class DumpPtFileTracer(TracerBase):

    def __init__(self, config):
        super().__init__(config)
        self.dump_pt_hook_options = config.get(
            'dump_pt_hook_options', {}
        )
        if self.forward_hook:
            self.save_forward_number = 0
        if self.backward_hook:
            self.save_backward_number = 0

        if self.has_hacker:
            self.hacker.hack(self.hook_forward_fn, self.hook_backward_fn)

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
        classify = "non nn.module" if is_non_nn_module(module) else "nn.module"
        timestamp = str(int(time.time()))
        hook_data = NewHookData(module, input, output, classify, timestamp)
        self._save_pt_data(hook_data, mode=mode)

    def _save_pt_data(self, data, mode="Forward"):
        if mode == "Forward":
            pt_file = os.path.join(self.current_save_path, "forward_" +
                                   str(self.save_forward_number).zfill(6) + ".pt")
            self.save_forward_number = self.save_forward_number + 1
            with open(pt_file, "wb+") as f:
                self._save_module_name_file_path_map(data.module_name, pt_file)
                torch.save(data, f)
        else:
            pt_file = os.path.join(self.current_save_path, "backward_" +
                                   str(self.save_backward_number).zfill(6) + ".pt")
            self.save_backward_number = self.save_backward_number + 1
            with open(pt_file, "wb+") as f:
                self._save_module_name_file_path_map(data.module_name, pt_file)
                torch.save(data, f)


class Tracer(object):

    def __init__(self, config):
        """
        config : yaml type configuration
        """
        config = handle_config(config)

        self.register_hooks = config.get('register_hooks', [])

        self.trace_fns = []
        self.untrace_fns = []

        if "dump_pt_hook" in self.register_hooks:
            self.dump_pt_hook = DumpPtFileTracer(config)
            self.trace_fns.append(self.dump_pt_hook.trace)
            self.untrace_fns.append(self.dump_pt_hook.untrace)

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
