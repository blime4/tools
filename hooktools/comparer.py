import os
from natsort import natsorted
from hooktools.utils import NewHookData
from hooktools.utils import handle_config
from hooktools.utils import get_file_list
from hooktools.utils import is_need_to_filter_specifiy_modules
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import defaultdict
from operator import itemgetter
import transformers
import atexit
import signal


class Utils(object):
    def __init__(self) -> None:
        self.topk_dict = defaultdict(list)
        self.detail = ""
        self.prefix = ""
        self.current_files = ()
        self.state = defaultdict()
        self.latest_conclusion_pk_filename = ""

    def print_raw_data(self, module, actual, desired):
        print("-------↓")
        print(f"[module] : {module}")
        if module is None or module == "":
            print(f"something error. in {self.get_detail()}, module is None")
        print(f"[actual] : {actual}")
        print(f"[desired] : {desired}")
        print("-------↑\n")

    def add_topk(self, fn_name="", module="", error=None, actual=None, desired=None):
        self.topk_dict[fn_name].append(
            {
                "module": module,
                "error": error,
                "actual": actual,
                "desired": desired,
                "detail": self.get_detail()
            }
        )

    def print_topk(self, k=100):
        for fn_name, lst in self.topk_dict.items():
            sorted_lst = sorted(lst, reverse=True, key=lambda x: x["error"])
            topk_lst = sorted_lst[:k]
            print(f"Top {k} errors for function '{fn_name}':")
            for index, item in enumerate(topk_lst):
                module, error, actual, desired, detail = item["module"], item["error"], item["actual"], item["desired"], item["detail"]
                print(
                    f"\t{index} = {detail}\n {module}: {error:.6f} \n\t\t atual : {actual}, \n\t\t desired : {desired}\n\n")

    def conclusion(self, k=100, save_pk=True):
        if save_pk:
            import time
            for fn_name, lst in self.topk_dict.items():
                sorted_lst = sorted(lst, reverse=True, key=lambda x: x["error"])
                self.latest_conclusion_pk_filename = "topk_"+fn_name+"_"+str(int(time.time()))+".pk"
                torch.save(sorted_lst[:k], self.latest_conclusion_pk_filename)
        print("\n\n")
        print("conclusion".center(20, '-'))
        print(self.print_topk(k=k))

    def set_state(self, folder, epoch, step):
        self.state = {
            "folder": folder,
            "epoch": epoch,
            "step": step,
        }

    def set_prefix(self, prefix):
        self.prefix = prefix

    def get_detail(self):
        try:
            r = f"[{self.state['folder']}][{self.state['epoch']}][{self.state['step']}]"
        except:
            r = ""
        return r+f"[{self.prefix}]\n[current_files] : [{self.current_files}]"

    def get_latest_conclusion_pk_filename(self):
        return self.latest_conclusion_pk_filename

    def _check_if_need_to_compare(self, module_name):
        return is_need_to_filter_specifiy_modules(module_name, self.compare_specifiy_modules)

    def _check_path_exists(self, path):
        if not os.path.exists(path):
            print(path)
            raise path + "not exists!"

    def _get_both_filelist(self, filelist_1, filelist_2):
        if not filelist_1 or not filelist_2:
            return [], []

        pbl1 = set([os.path.basename(pb) for pb in filelist_1])
        pbl2 = set([os.path.basename(pb) for pb in filelist_2])

        only_in_pb_file_list_1 = pbl1 - pbl2
        only_in_pb_file_list_2 = pbl2 - pbl1
        both_pb_file_list = sorted(list(pbl1 | pbl2))

        if only_in_pb_file_list_1 != set():
            print("These files exist only in pb_file_list_1 : ",
                  only_in_pb_file_list_1)
        if only_in_pb_file_list_2 != set():
            print("These files exist only in pb_file_list_2 : ",
                  only_in_pb_file_list_2)

        both_file_list_1 = [os.path.join(os.path.dirname(
            filelist_1[0]), pb) for pb in both_pb_file_list]
        both_file_list_2 = [os.path.join(os.path.dirname(
            filelist_2[0]), pb) for pb in both_pb_file_list]

        return both_file_list_1, both_file_list_2

    def set_current_files(self, current_files):
        self.current_files = current_files

class Filter(Utils):

    def __init__(self, config):
        super().__init__()
        config = handle_config(config)
        self.filter_config = config.get('filter', {})
        self.show_max_error_only = config.get("show_max_error_only", False)
        if "global_filter" in self.filter_config:
            setattr(self, "global_filter", eval(
                str(self.filter_config["global_filter"])))
        else:
            self.global_filter = None

        for name in ["L1_filter", "AE_filter", "CS_filter", "MSE_filter", "MAE_filter", "RMSE_filter", "MAPE_filter"]:
            if name in self.filter_config:
                setattr(self, name, self.filter_config[name])

        self.registersi_signal = config.get('registersi_signal', True)
        atexit.register(self.conclusion)
        if self.registersi_signal:
            signal.signal(signal.SIGINT, self.conclusion)

    def filter(self, data=None, fn_name="", prefix="",  module="", actual=None, desired=None, attr=None):

        if fn_name == "L1":
            self.filter(data=data[0], fn_name="l1_error", prefix=prefix,
                           module=module, actual=actual, desired=desired, attr="L1_filter")
            # self.filter(data=data[1], fn_name="rel_error", prefix=prefix,
            #                module=module, actual=actual, desired=desired, attr="L1_filter")
        else:
            max_data = torch.max(data)
            attr = "{}_filter".format(fn_name) if attr is None else attr
            if attr is not None and hasattr(self, attr):
                filter_error = eval(str(getattr(self, attr)))
            elif self.global_filter is not None:
                filter_error = self.global_filter
            else:
                filter_error = None

            if filter_error is None or max_data > filter_error:
                self.detail = self.get_detail()
                data_to_print = max_data if self.show_max_error_only else data
                print(f"[detail] : {self.detail}", end='\n')
                print(f"[metric_name] : {fn_name}", end='\n')
                print(f"[error] : {data_to_print}", end='\n')

                self.print_raw_data(module, actual, desired)
                # todo : think about max_data or data
                self.add_topk(fn_name, module, max_data, actual, desired)

class Evaluator(Filter):

    def __init__(self, config):
        super().__init__(config)
        config = handle_config(config)
        self.evaluation_metrics = config.get('evaluation_metrics', [])

        self.evaluator_verbose = 'evaluator_verbose' in self.evaluation_metrics
        self.skip_nn_module = 'skip_nn_module' in self.evaluation_metrics
        self.skip_non_nn_module = 'skip_non_nn_module' in self.evaluation_metrics
        self.current_module = ""
        self.current_files = ()

        self.registered_evaluations = dict()
        if "L1" in self.evaluation_metrics:
            self.register_evaluation("L1", self.evaluate_l1_loss)
        if "AE" in self.evaluation_metrics:
            self.register_evaluation("AE", self.evaluate_absolute_error)
        if "MAE" in self.evaluation_metrics:
            self.register_evaluation("MAE", self.evaluate_mean_absolute_error)
        if "CS" in self.evaluation_metrics:
            self.register_evaluation("CS", self.evaluate_cosine_similarity)
        if "MSE" in self.evaluation_metrics:
            self.register_evaluation("MSE", self.evaluate_mean_squared_error)
        if "RMSE" in self.evaluation_metrics:
            self.register_evaluation(
                "RMSE", self.evaluate_root_mean_squared_error)
        if "MAPE" in self.evaluation_metrics:
            self.register_evaluation(
                "MAPE", self.evaluate_mean_absolute_percentage_error)
        self.compare_specifiy_modules = config.get('compare_specifiy_modules', {})

    def register_evaluation(self, fn_name, evaluation_fn):
        self.registered_evaluations[fn_name] = evaluation_fn

    def evaluate_cosine_similarity(self, actual, desired):
        return torch.nn.functional.cosine_similarity(
            actual, desired, dim=0)

    def evaluate_mean_squared_error(self, actual, desired):
        return torch.mean((actual - desired) ** 2)

    def evaluate_root_mean_squared_error(self, actual, desired):
        return torch.sqrt(torch.mean((actual - desired) ** 2))

    def evaluate_mean_absolute_percentage_error(self, actual, desired):
        return 100 * torch.mean(torch.abs((actual - desired) / actual))

    def evaluate_mean_absolute_error(self, actual, desired):
        return torch.mean(torch.abs(actual - desired))

    def evaluate_l1_loss(self, actual, desired):
        # siyi's way
        try:
            actual = actual.cpu() if actual.is_cuda else actual
            desired = desired.cpu() if desired.is_cuda else desired
            # 解决 inf 问题
            finite_mask = torch.isfinite(actual) | torch.isfinite(desired)
            filtered_actual = actual[finite_mask]
            filtered_desired = desired[finite_mask]
            l1_error = (filtered_actual - filtered_desired).float().abs().mean()
            rel_error = l1_error / (filtered_actual.abs().float().mean())

            if l1_error * rel_error > 10:
                print(f'\n###\n should checked! : l1_error * rel_error > 10, current_module is : [{self.current_module}] , {self.get_detail()}\n###\n')
            return (l1_error.detach(), rel_error.detach())

        except Exception as e:
            print("ERROR : ", e)
            raise "failed."

    def evaluate_absolute_error(self, actual, desired):
        absolute_error = torch.abs(actual - desired)
        return absolute_error

    def evalute_(self, actual, desired, prefix=""):
        assert type(actual) == type(desired), f"type(actual) is {type(actual)} which is not same with type(desired): {type(desired)}"

        if isinstance(actual, NewHookData):
            assert actual.module_name==desired.module_name, f"module_name must be same : actual : {actual.module_name} , desired : {desired.module_name}"
            self.current_module = actual.module_name
            if hasattr(actual, "input"):
                self.evalute_(actual.input, desired.input, prefix+'[  input ]')
            if hasattr(actual, "output"):
                self.evalute_(actual.output, desired.output,
                              prefix+'[ output ]')
            if hasattr(actual, "gradient"):
                self.evalute_(actual.gradient, desired.gradient,
                              prefix+'[gradient]')
            if hasattr(actual, "gradient_grad"):
                self.evalute_(actual.gradient_grad,
                              desired.gradient_grad, prefix+'[gradient_grad]')
        elif isinstance(actual, torch.Tensor):
            for fn_name, evaluation_fn in self.registered_evaluations.items():
                if not torch.is_floating_point(actual):
                    actual = actual.double()
                    desired = desired.double()
                error = evaluation_fn(actual, desired)
                self.set_prefix(prefix)
                if self.current_module is None or self.current_module == "":
                    print(f'For debug : self.current_module == "" in here.!!!!, detail is {self.get_detail()}.')
                    print('actual is : ', actual)
                    print('desired is : ', desired)

                self.filter(
                    error, fn_name, prefix, self.current_module, actual, desired)

        elif isinstance(actual,transformers.modeling_outputs.BaseModelOutputWithPast):
            if hasattr(actual, "last_hidden_state"):
                self.evalute_(actual.last_hidden_state, desired.last_hidden_state, prefix+'[BaseModelOutputWithPast][last_hidden_state]')
            if hasattr(actual, "past_key_values"):
                self.evalute_(actual.past_key_values, desired.past_key_values, prefix+'[BaseModelOutputWithPast][past_key_values]')

        elif isinstance(actual, transformers.modeling_outputs.CausalLMOutputWithPast):
            if hasattr(actual, "logits"):
                self.evalute_(actual.logits, desired.logits, prefix+'[CausalLMOutputWithPast][logits]')
            if hasattr(actual, "past_key_values"):
                self.evalute_(actual.past_key_values, desired.past_key_values, prefix+'[CausalLMOutputWithPast][past_key_values]')

        elif isinstance(actual, (list, tuple)):
            for idx, (val1, val2) in enumerate(zip(actual, desired)):
                self.evalute_(val1, val2, prefix+f'[idx-{idx}]\t')

        elif isinstance(actual, (int, float, bool, str)):
            # non nn.module will have some input, ouput data, which type in (int, float, bool, str)
            if (actual != desired or self.evaluator_verbose):
                print("[underlying type data not match]", prefix, "\nactual:\t", actual, "\ndesired:\t", desired)
        else:
            if actual is None and desired is None:
                pass
            else:
                print(f"for debug : actual : {actual}, dir(actual) : {dir(actual)}, type(actual) : {type(actual)}")
                print(f"for debug : desired : {desired}, dir(desired) : {dir(desired)}, type(desired) : {type(desired)}")
                torch.save(actual, f"actual-{type(actual)}.pk")
                torch.save(desired, f"actual-{type(desired)}.pk")
                raise TypeError(f"Unsupported data type : {type(actual)}")

    def evalute(self, data_1, data_2):
        if self.skip_nn_module and data_1.classify == "nn.module" and data_2.classify == "nn.module":
            return
        if self.skip_non_nn_module and data_1.classify == "non nn.module" and data_2.classify == "non nn.module":
            return

        if data_1.module_name == data_2.module_name:
            if self.evaluator_verbose:
                print("module_name: ", data_1.module_name)
        else:
            print(self.get_detail())
            print("module_name: \ndata_1 :", data_1.module_name,
                  "\ndata_2 : ", data_2.module_name)
            print(f"current_files is : {self.current_files}.")
            raise f'\n###\n should checked!: module_name not match \n###\n'

        if self._check_if_need_to_compare(data_1.module_name):
            self.evalute_(data_1, data_2)

class Comparer(Evaluator):

    def __init__(self, config):
        """
        config : yaml type configuration
        """
        super().__init__(config)
        config = handle_config(config)
        self.comparer_name = config.get('comparer_name', "")
        self.compare_mode = config.get('compare_mode', 0)
        self.file_type = config.get('file_type', 'pt')
        self.compare_non_nn_module = config.get('compare_non_nn_module', False)
        self.compare_nn_module = config.get('compare_nn_module', False)
        # compare mode:
        # 0 : compare directory
        # 1 : compare file
        # 2 : compare filelist
        self.evaluator = Evaluator(config)

        self.auto_conclusion = config.get("auto_conclusion", True)

        if self.compare_mode == 0:
            compare_options = config.get('compare_directory_options', {})
            self.compare_epochs = compare_options.get('compare_epochs', [])
            self.compare_steps = compare_options.get('compare_steps', [])
            def _trans(untrans):
                if untrans.startswith("[") and untrans.endswith(")"):
                    # [start, end)
                    start = int(untrans[1:-1].split(",")[0])
                    end = int(untrans[1:-1].split(",")[1])
                    return [num for num in range(start, end)]
                elif isinstance(eval(untrans), int):
                    return [eval(untrans)]
                else:
                    assert isinstance(eval(untrans), list)
                    return eval(untrans)
            self.compare_epochs = _trans(self.compare_epochs)
            self.compare_steps = _trans(self.compare_steps)

            self.only_compare_input = compare_options.get(
                'only_compare_input', False)
            self.only_compare_output = compare_options.get(
                'only_compare_output', False)

            self.compared_directory_1 = compare_options.get(
                'compared_directory_1')
            self.compared_directory_2 = compare_options.get(
                'compared_directory_2')
            self.compare_folder_name = compare_options.get(
                'compare_folder_name', [])

            self.compared_file_1 = ""
            self.compared_file_2 = ""

        elif self.compare_mode == 1:
            compare_options = config.get('compare_file_options', {})
            self.compared_file_1 = compare_options.get("compared_file_1")
            self.compared_file_2 = compare_options.get("compared_file_2")

        elif self.compare_mode == 2:
            compare_options = config.get('compare_filelist_options', {})
            self.compared_filelist_1 = compare_options.get(
                "compared_filelist_1")
            self.compared_filelist_2 = compare_options.get(
                "compared_filelist_2")

        self.compare_by_order = config.get("compare_by_order", False)
        self.compare_both_file = not self.compare_by_order or config.get(
            "compare_both_file", False)
        self.current_files = ()
        self.compare_verbose = compare_options.get('compare_verbose', False)
        if self.compare_verbose:
            print(config)

    # TODO: complete compare mode and compare precision way.

    def compare(self):
        if self.compare_mode == 0:
            self._check_path_exists(self.compared_directory_1)
            self._check_path_exists(self.compared_directory_2)
            self.compare_directory()
        elif self.compare_mode == 1:
            self.compare_file(self.compared_file_1, self.compared_file_2)
        elif self.compare_mode == 2:
            filelist_1 = get_file_list(
                path=self.compared_filelist_1, endswith=self.file_type)
            filelist_2 = get_file_list(
                path=self.compared_filelist_2, endswith=self.file_type)
            self.compare_filelist(filelist_1, filelist_2)
        if self.auto_conclusion:
            self.conclusion()

    def compare_directory(self):
        # if compare_epochs or compare_steps is not set,
        # default compare all epochs or steps in the compared directory.
        # ├── backward
        # │   └── epoch0
        # │       └── step1
        # ├── forward
        # │   └── epoch0
        # │       └── step1
        # └── gradient
        #     └── epoch0
        #         ├── step0
        #         ├── step1
        folder_name = [f for f in os.listdir(
            self.compared_directory_1) if f in self.compare_folder_name]
        # folder --------------------------------
        with tqdm(folder_name) as folder_pbar:
            for folder in folder_pbar:
                folder_path_1 = os.path.join(self.compared_directory_1, folder)
                folder_path_2 = os.path.join(self.compared_directory_2, folder)
                folder_pbar.set_description(desc=f"Processing : {folder_path_1}, {folder_path_2}", refresh=True)
                if not self.compare_epochs:
                    epochs = natsorted(os.listdir(folder_path_1))
                else:
                    epochs = ["epoch" + str(i) for i in self.compare_epochs]

                # epoch --------------------------------
                with tqdm(epochs) as epoch_pbar:
                    for epoch in epoch_pbar:
                        epoch_pbar.set_description(desc=f"Processing : {epoch}",  refresh=True)
                        epoch_path_1 = os.path.join(folder_path_1, epoch)
                        epoch_path_2 = os.path.join(folder_path_2, epoch)

                        if not self.compare_steps:
                            steps = natsorted(os.listdir(epoch_path_1))
                        else:
                            steps = ["step" + str(i)
                                     for i in self.compare_steps]

                        # step --------------------------------
                        with tqdm(steps) as step_pbar:
                            for step in step_pbar:
                                step_pbar.set_description(desc=f"Processing : {step}",  refresh=True)
                                step_path_1 = os.path.join(epoch_path_1, step)
                                step_path_2 = os.path.join(epoch_path_2, step)

                                filelist_1 = get_file_list(
                                    path=step_path_1, endswith=self.file_type)
                                filelist_2 = get_file_list(
                                    path=step_path_2, endswith=self.file_type)
                                self.set_state(
                                    folder, epoch, step)
                                self.compare_filelist(filelist_1, filelist_2)

    def compare_filelist(self, filelist_1, filelist_2):
        if self.compare_both_file:
            both_file_list_1, both_file_list_2 = self._get_both_filelist(
                filelist_1, filelist_2)
            for file1, file2 in tqdm(zip(both_file_list_1, both_file_list_2)):
                # if self.compare_verbose:
                #     print("filelist_1 : ", filelist_1)
                #     print("filelist_2 : ", filelist_2)
                self.compare_file(file1, file2)
        elif self.compare_by_order:
            with tqdm(zip(filelist_1, filelist_2)) as file_pbar:
                for file1, file2 in file_pbar:
                    file_pbar.set_description(desc=f"[{file1}]",  refresh=True)
                    # if self.compare_verbose:
                    #     print("filelist_1 : ", filelist_1)
                    #     print("filelist_2 : ", filelist_2)
                    self.compare_file(file1, file2)

    def compare_file(self, file_path_1, file_path_2):
        if self.file_type == "pt":
            if self.compare_verbose:
                print("file_path_1 ", file_path_1)
                print("file_path_2 ", file_path_2)
            self.compare_pt_file(file_path_1, file_path_2)

    def compare_pt_file(self, file_path_1, file_path_2):
        assert file_path_1 != file_path_2
        with open(file_path_1, "rb") as f1:
            f1.seek(0)
            pt_data_1 = torch.load(f1, map_location="cpu")
        with open(file_path_2, "rb") as f2:
            f2.seek(0)
            pt_data_2 = torch.load(f2, map_location="cpu")
        self.set_current_files((file_path_1, file_path_2))
        self.evalute(pt_data_1, pt_data_2)

    def set_current_files(self, current_files):
        self.current_files = current_files

    def _check_path_exists(self, path):
        if not os.path.exists(path):
            print(path)
            raise path + "not exists!"

    def _get_both_filelist(self, filelist_1, filelist_2):

        if not filelist_1 or not filelist_2:
            return [], []

        pbl1 = set([os.path.basename(pb) for pb in filelist_1])
        pbl2 = set([os.path.basename(pb) for pb in filelist_2])

        only_in_pb_file_list_1 = pbl1 - pbl2
        only_in_pb_file_list_2 = pbl2 - pbl1
        both_pb_file_list = sorted(list(pbl1 | pbl2))

        if only_in_pb_file_list_1 != set():
            print("These files exist only in pb_file_list_1 : ",
                  only_in_pb_file_list_1)
        if only_in_pb_file_list_2 != set():
            print("These files exist only in pb_file_list_2 : ",
                  only_in_pb_file_list_2)

        both_file_list_1 = [os.path.join(os.path.dirname(
            filelist_1[0]), pb) for pb in both_pb_file_list]
        both_file_list_2 = [os.path.join(os.path.dirname(
            filelist_2[0]), pb) for pb in both_pb_file_list]

        return both_file_list_1, both_file_list_2


# TODO:
# 4. 按照算子, 比较算子的误差，变化