import torch
import os
import time
import json
from hooktools.utils import NewHookData
from hooktools.utils import handle_config
from hooktools.utils import is_need_to_filter_specifiy_modules
from hooktools.hacker import Hacker

class TracerBase(object):

    def __init__(self, config, model):
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

        self.forward_hook = self.trace_mode == 1 or self.trace_mode == 3
        self.backward_hook = self.trace_mode == 2 or self.trace_mode == 3

        self.timestamp = str(int(time.time()))
        self.save_path=os.path.join(self.log_dir, self.tracer_name+"_"+self.timestamp)
        self.forward_path = os.path.join(self.save_path, "forward")
        self.backward_path = os.path.join(self.save_path, "backward")
        self.gradient_path = os.path.join(self.save_path, "gradient")

        self.save_forward_number = 0
        self.save_backward_number = 0
        self.save_gradient_number = 0

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
        self.is_trace = False
        self.trace_granularity = config.get("trace_granularity", 0)
        self.model = model

    def trace(self, epoch=-1, step=-1):
        """
        epoch : integer number of epoch
        step : integer number of step
        """
        if self.trace_mode == 0: return
        print("tracing !!!!, epoch=%d, step=%d" % (epoch, step))
        self.epoch = epoch
        self.step = step

        if self.trace_granularity == 0:
            self.forward_handle = torch.nn.modules.module.register_module_forward_hook(
                                    self.hook_forward_fn) if self.forward_hook else None
            # self.backward_handle = torch.nn.modules.module.register_module_full_backward_hook(self.hook_backward_fn) # still have an RuntimeError : Module backward hook for grad_input is called before the grad_output one. This happens because the gradient in your nn.M odule flows to the Module’s input without passing through the Module’s output
            self.backward_handle = torch.nn.modules.module.register_module_backward_hook(
                                    self.hook_backward_fn)  if self.backward_hook else None
            # this api will work in torchvision==0.8.2, ==0.9.0
        elif self.trace_granularity == 1:
            for name, module in self.model.named_modules():
                if module is not None:
                    self.forward_handle = module.register_forward_hook(self.hook_forward_fn) if self.forward_hook else None
                    self.backward_handle = module.register_backward_hook(self.hook_backward_fn) if self.backward_hook else None

    def untrace(self):
        if self.trace_mode == 0:
            return
        if self.forward_hook:
            self.forward_handle.remove()
        if self.backward_hook:
            self.backward_handle.remove()
        self.epoch = -1
        self.step = -1

    def trace_gradient(self, epoch=-1, step=-1):
        pass

    def hook_forward_fn(self, module, input, output):
        pass

    def hook_backward_fn(self, module, grad_input, grad_output):
        pass

    def _set_current_save_path(self, mode="Forward"):
        if mode == "Forward":
            self.current_save_path = self.forward_path
        elif mode == "Backward":
            self.current_save_path = self.backward_path
        elif mode == "Gradient":
            self.current_save_path = self.gradient_path

        self.current_save_path = os.path.join(
            self.current_save_path, "epoch" + str(self.epoch))
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

    def __init__(self, config, model):
        super().__init__(config, model)
        self.dump_pt_hook_options = config.get(
            'dump_pt_hook_options', {}
        )

        if self.has_hacker:
            self.hacker.hack(self.hook_forward_fn, self.hook_backward_fn)

        self.hook_specifiy_modules = self.dump_pt_hook_options.get("hook_specifiy_modules", {})

    def hook_forward_fn(self, module, input, output):
        self._hook_impl(module, input, output, "Forward")

    def hook_backward_fn(self, module, grad_input, grad_output):
        self._hook_impl(module, grad_input, grad_output, "Backward")

    def trace(self, epoch=-1, step=-1):
        self.is_trace=True
        return super().trace(epoch, step)

    def untrace(self):
        self.is_trace=False
        return super().untrace()

    def trace_gradient(self, epoch=-1, step=-1):
        # usage :
        """_summary_

        Args:
            epoch (int, optional): _description_. Defaults to -1.
            step (int, optional): _description_. Defaults to -1.

            example :

                optimizer.step()
            --> trace.trace_gradient(epoch, step)
                optimizer.zero_grad()

        """

        if self.trace_mode == 0: return
        print("tracing gradient !!!!, epoch=%d, step=%d" % (epoch, step))
        self.epoch = epoch
        self.step = step
        for name, param in self.model.named_parameters():
            if param is not None:
                if self._check_if_need_to_save(name):
                    self._set_current_save_path("Gradient")
                    hook_data = NewHookData(module=name, gradient=param, gradient_grad=param.grad)
                    self._save_pt_data(hook_data, mode="Gradient")
                    # print(f"[epoch-{epoch}][step-{step}][grad] : {param.grad}")

    def _hook_impl(self, module, input, output, mode="Forward"):
        if self.is_trace:
            if self._check_if_need_to_save(module):
                self._set_current_save_path(mode)
                hook_data = NewHookData(module=module, input=input, output=output)
                self._save_pt_data(hook_data, mode=mode)

    def _save_pt_data(self, data, mode="Forward"):
        counter_dict = {
            "Forward": "save_forward_number",
            "Backward": "save_backward_number",
            "Gradient": "save_gradient_number"
        }
        counter = getattr(self, counter_dict[mode])
        pt_file = os.path.join(self.current_save_path, mode.lower() +
                            "_" + str(counter).zfill(6) + ".pt")
        setattr(self, counter_dict[mode], counter + 1)
        with open(pt_file, "wb+") as f:
            self._save_module_name_file_path_map(data.module_name, pt_file)
            torch.save(data, f)

    def _check_if_need_to_save(self, module_name):
        return is_need_to_filter_specifiy_modules(module_name, self.hook_specifiy_modules)


class Tracer(object):

    def __init__(self, config, model):
        """
        config : yaml type configuration
        """
        config = handle_config(config)

        self.register_hooks = config.get('register_hooks', [])

        self.trace_fns = []
        self.untrace_fns = []

        if "dump_pt_hook" in self.register_hooks:
            self.dump_pt_hook = DumpPtFileTracer(config, model)
            self.trace_fns.append(self.dump_pt_hook.trace)
            self.untrace_fns.append(self.dump_pt_hook.untrace)

        print(config)

    def trace(self, epoch=-1, step=-1):
        for trace_fn in self.trace_fns:
            trace_fn(epoch, step)

    def trace_gradient(self, epoch=-1, step=-1):
        if self.dump_pt_hook:
            self.dump_pt_hook.trace_gradient(epoch=epoch, step=step)

    def untrace(self):
        try:
            for untrace_fn in self.untrace_fns:
                untrace_fn()
        except:
            pass


# think about if we need untrace_gradient api