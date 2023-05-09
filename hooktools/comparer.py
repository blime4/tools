import os
from hooktools.utils import handle_config, get_file_list
from hooktools.utils import NewHookData
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn


class Comparer(object):

    def __init__(self, config):
        """
        config : yaml type configuration
        """
        config = handle_config(config)
        self.log_dir = config.get('log_dir', "./tmp")
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

        if self.compare_mode == 0:
            compare_options = config.get('compare_directory_options', {})
            self.compare_epochs = compare_options.get('compare_epochs', [])
            self.compare_steps = compare_options.get('compare_steps', [])

            self.only_compare_forward = compare_options.get(
                'only_compare_forward', False)
            self.only_compare_backward = compare_options.get(
                'only_compare_backward', False)
            self.only_compare_input = compare_options.get(
                'only_compare_input', False)
            self.only_compare_output = compare_options.get(
                'only_compare_output', False)

            self.compared_directory_1 = compare_options.get(
                'compared_directory_1')
            self.compared_directory_2 = compare_options.get(
                'compared_directory_2')
            self._check_path_exists(self.compared_directory_1)
            self._check_path_exists(self.compared_directory_2)

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
        self.verbose = compare_options.get('verbose', False)
        if self.verbose:
            print(config)

    # TODO: complete compare mode and compare precision way.

    def compare(self):
        if self.compare_mode == 0:
            self.compare_directory()
        elif self.compare_mode == 1:
            self.compare_file()
        elif self.compare_mode == 2:
            filelist_1 = get_file_list(
                path=self.compared_filelist_1, endswith=self.file_type)
            filelist_2 = get_file_list(
                path=self.compared_filelist_2, endswith=self.file_type)
            self.compare_filelist(filelist_1, filelist_2)

    def compare_directory(self):
        # if compare_epochs or compare_steps is not set,
        # default compare all epochs or steps in the compared directory.
        if not self.compare_epochs:
            epochs = sorted(os.listdir(self.compared_directory_1))
        else:
            epochs = ["epoch" + str(i) for i in self.compare_epochs]

        # epoch --------------------------------
        for epoch in tqdm(epochs, desc="epoch : "):
            epoch_path_1 = os.path.join(self.compared_directory_1, epoch)
            epoch_path_2 = os.path.join(self.compared_directory_2, epoch)

            if not self.compare_steps:
                steps = sorted(os.listdir(epoch_path_1))
            else:
                steps = ["step" + str(i) for i in self.compare_steps]

            # step --------------------------------
            for step in tqdm(steps, desc="step : "):
                step_path_1 = os.path.join(epoch_path_1, step)
                step_path_2 = os.path.join(epoch_path_2, step)

                filelist_1 = get_file_list(
                    path=step_path_1, endswith=self.file_type)
                filelist_2 = get_file_list(
                    path=step_path_2, endswith=self.file_type)
                self.compare_filelist(filelist_1, filelist_2)

    def compare_filelist(self, filelist_1, filelist_2):
        if self.compare_both_file:
            both_file_list_1, both_file_list_2 = self._get_both_filelist(
                filelist_1, filelist_2)
            for file1, file2 in tqdm(zip(both_file_list_1, both_file_list_2)):
                self.compare_file(file1, file2)
        elif self.compare_by_order:
            for file1, file2 in tqdm(zip(filelist_1, filelist_2), desc="Processing files...\n"):
                self.compare_file(file1, file2)

    def compare_file(self, file_path_1, file_path_2):
        if not file_path_1 and not file_path_2:
            file_path_1 = self.compared_file_1
            file_path_2 = self.compared_file_2
        if self.file_type == "pt":
            if self.verbose:
                print("file_path_1 ", file_path_1)
                print("file_path_2 ", file_path_2)
            self.compare_pt_file(file_path_1, file_path_2)

    def compare_pt_file(self, file_path_1, file_path_2):
        with open(file_path_1, "rb") as f1:
            f1.seek(0)
            pt_data_1 = torch.load(f1)
        with open(file_path_2, "rb") as f2:
            f2.seek(0)
            pt_data_2 = torch.load(f2)
        self.evaluator.evalute(pt_data_1, pt_data_2)

    def _check_path_exists(self, path):
        if not os.path.exists(path):
            raise path + "not exists!"

    def _get_both_filelist(self, filelist_1, filelist_2):

        if not filelist_1 or not filelist_2:
            return [], []

        pbl1 = [os.path.basename(pb) for pb in filelist_1]
        pbl2 = [os.path.basename(pb) for pb in filelist_2]

        only_in_pb_file_list_1 = pbl1 - pbl2
        only_in_pb_file_list_2 = pbl2 - pbl1
        both_pb_file_list = pbl1 | pbl2

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


class MetricData(object):

    def __init__(self):
        self.input = []
        self.output = []

    def __repr__(self) -> str:
        return f"input: {self.input} , output: {self.output}"


class Evaluator(object):

    def __init__(self, config):
        config = handle_config(config)
        self.evaluation_metrics = config.get('evaluation_metrics', [])

        self.verbose = 'verbose' in self.evaluation_metrics
        self.skip_nn_module = 'skip_nn_module' in self.evaluation_metrics
        self.skip_non_nn_module = 'skip_non_nn_module' in self.evaluation_metrics

        self.registered_evaluations = dict()
        if "L1" in self.evaluation_metrics:
            self.register_evaluation("l1", self.evaluate_l1_loss)
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

    def register_evaluation(self, fn_name, evaluation_fn):
        self.registered_evaluations[fn_name] = evaluation_fn

    def evaluate_cosine_similarity(self, data_1, data_2):
        def _get_tensor_cosine_similarity(tensor_1, tensor_2, data_1, data_2):
            try:
                tensor_1 = tensor_1.reshape(1, -1)
                tensor_2 = tensor_2.reshape(1, -1)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                return cos(tensor_1, tensor_2).tolist()
            except:
                print("data_1 : ", data_1)
                print("data_2 : ", data_2)
                print("tensor_1.shape : ", tensor_1.shape)
                print("tensor_2.shape : ", tensor_2.shape)
                tensor_1 = tensor_1.reshape(1, -1)
                tensor_2 = tensor_2.reshape(1, -1)
                min_len = min(tensor_1.shape[1], tensor_2.shape[1])
                tensor_1 = tensor_1[:, :min_len]
                tensor_2 = tensor_2[:, :min_len]
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                return cos(tensor_1, tensor_2).tolist()

        def _get_struct_cosine_similarity(data_1, data_2):
            metric = MetricData()
            for input_1, input_2 in zip(data_1.input, data_2.input):
                if isinstance(input_1, torch.Tensor):
                    cos = _get_tensor_cosine_similarity(
                        input_1, input_2, data_1, data_2)
                    metric.input.append(cos)
                elif isinstance(input_1, list):
                    cos_list = []
                    for inp1, inp2 in zip(input_1, input_2):
                        cos_list.append(_get_tensor_cosine_similarity(
                            inp1, inp2, data_1, data_2))
                    metric.input.append(cos_list)
                else:
                    print("Invalid input type : ", type(input_1))
            if isinstance(data_1.output, torch.Tensor):
                cos = _get_tensor_cosine_similarity(
                    data_1.output, data_2.output, data_1, data_2)
                metric.output.append(cos)
            elif isinstance(data_1.output, list) or isinstance(data_1.output, tuple):
                cos_list = []
                for out_1, out_2 in zip(data_1.output, data_2.output):
                    cos_list.append(_get_tensor_cosine_similarity(
                        out_1, out_2, data_1, data_2))
                metric.output.append(cos_list)
            else:
                print("Invalid output type : ", type(data_1.output))
            return metric
        print(_get_struct_cosine_similarity(data_1, data_2))

    def evaluate_mean_squared_error(self, data_1, data_2):
        data_1 = data_1.reshape(1, -1)
        data_2 = data_2.reshape(1, -1)
        loss = nn.MSELoss()
        return loss(data_1, data_2)

    def evaluate_root_mean_squared_error(self, data_1, data_2):
        pass

    def evaluate_mean_absolute_percentage_error(self, data_1, data_2):
        pass

    def evaluate_l1_loss(self, data_1, data_2):
        # siyi way
        def compare(actual, desired, prefix=""):
            if isinstance(actual, (list, tuple)):
                if actual:
                    for idx in range(len(actual)):
                        try:
                            compare(actual[idx], desired[idx],
                                    prefix=prefix+f"{idx}")
                        except Exception as e:
                            print("ERROR : ", e)
                            raise prefix + "failed"
            elif isinstance(actual, torch.Tensor) and isinstance(desired, torch.Tensor):
                try:
                    l1_error = (actual - desired).float().abs().mean()
                    rel_error = l1_error / (actual.abs().float().mean())
                    print(prefix, 'l1_error: ', l1_error.detach().numpy(),
                          'rel_error', rel_error.detach().numpy())
                    if l1_error * rel_error > 10:
                        print('\n###\n', prefix, 'should checked!', '\n###\n')

                except Exception as e:
                    print("ERROR : ", e)
                    raise prefix + "failed."
            elif isinstance(actual, NewHookData):
                print("↓"*30)
                print("input:")
                compare(actual.input, desired.input)
                print("output:")
                compare(actual.output, desired.output)
                print("↑"*30)
            # non nn.module function's data's type have : int, float, bool, str
            elif isinstance(actual, (int, float, bool)):
                if(actual != desired or self.verbose):
                    print("actual : ", actual, "desired : ", desired,
                          "abs(actual - desired)", abs(actual-desired))
            elif isinstance(actual, str):
                print(f"str : {actual}, {desired}")
            else:
                raise TypeError(f"Unsupported data type : {type(actual)}")

        return compare(data_1, data_2)

    def evalute(self, data_1, data_2):
        if self.skip_nn_module and data_1.classify == "nn.module" and data_2.classify == "nn.module":
            return
        if self.skip_non_nn_module and data_1.classify == "non nn.module" and data_2.classify == "non nn.module":
            return

        if data_1.module_name == data_2.module_name:
            print("module_name: ", data_1.module_name)
        else:
            print("module_name: data_1 :", data_1.module_name,
                  "\tdata_2 : ", data_2.module_name)
            print('\n###\n should checked! \n###\n')
        metrics = {}
        for fn_name, evaluation_fn in self.registered_evaluations.items():
            metrics[fn_name] = evaluation_fn(data_1, data_2)
        return metrics
