from contextlib import contextmanager
import io
import os
import copy

import time
import json
import hashlib
from hooktools.utils import handle_config, get_file_list
from pathlib import Path
from tqdm import tqdm
from hooktools.trace_pb2 import HookData, MetaData, TensorProto, IOData
from hooktools.utils import to_array
from hooktools.tracer import PickleHookData
import pickle
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
        self.file_type = config.get('file_type', 0)
        # compare mode:
        # 0 : compare directory
        # 1 : compare protobuf file
        self.evaluator = Evaluation(config)

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
            self.compared_filelist_1 = compare_options.get("compared_filelist_1")
            self.compared_filelist_2 = compare_options.get("compared_filelist_2")
            self.compare_by_order = compare_options.get("compare_by_order", False)
            self.compare_both_file = not self.compare_by_order or compare_options.get("compare_both_file", False)

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
            filelist_1 = get_file_list(self.compared_filelist_1)
            filelist_2 = get_file_list(self.compared_filelist_2)
            self.compare_filelist(filelist_1, filelist_2)

    def compare_directory(self):
        # if compare_epochs or compare_steps is not set,
        # default compare all epochs or steps in the compared directory.
        if not self.compare_epochs:
            epochs = sorted(os.listdir(self.compared_directory_1))
        else:
            epochs = ["epoch" + str(i) for i in self.compare_epochs]

        # epoch --------------------------------
        for epoch in tqdm(epochs):
            epoch_path_1 = os.path.join(self.compared_directory_1, epoch)
            epoch_path_2 = os.path.join(self.compared_directory_2, epoch)

            if not self.compare_steps:
                steps = sorted(os.listdir(epoch_path_1))
            else:
                steps = ["step" + str(i) for i in self.compare_steps]

            # step --------------------------------
            for step in tqdm(steps):
                step_path_1 = os.path.join(epoch_path_1, step)
                step_path_2 = os.path.join(epoch_path_2, step)

                filelist_1 = get_file_list(step_path_1)
                filelist_2 = get_file_list(step_path_2)

                self.compare_filelist(filelist_1, filelist_2)

    def compare_filelist(self, filelist_1, filelist_2):
        if self.compare_both_file:
            both_file_list_1, both_file_list_2 = self._get_both_filelist(
                filelist_1, filelist_2)
            for file1, file2 in tqdm(zip(both_file_list_1, both_file_list_2)):
                self.compare_file(file1, file2)
        elif self.compare_by_order:
            for file1, file2 in tqdm(zip(filelist_1, filelist_2)):
                self.compare_file(file1, file2)

    def compare_file(self, file_path_1, file_path_2):
        if not file_path_1 and not file_path_2:
            file_path_1 = self.compared_file_1
            file_path_2 = self.compared_file_2
        if self.file_type == 0:
            self.compare_pickle_file(file_path_1, file_path_2)
        elif self.file_type == 1:
            pb_tensor_1 = self._parse_pb_path(file_path_1)
            pb_tensor_2 = self._parse_pb_path(file_path_2)
            self.compare_pb_tensor(pb_tensor_1, pb_tensor_2)
            
    def compare_pickle_file(self, file_path_1, file_path_2):
        # try:
        with open(file_path_1, "rb") as f1:
            f1.seek(0)
            pkl_data_1 = pickle.load(f1)
        with open(file_path_2, "rb") as f2:
            f2.seek(0)
            pkl_data_2 = pickle.load(f2)    
        print(self.evaluator.evalute(pkl_data_1, pkl_data_2))
        # except:
        #     print("file_path_1 : ",file_path_1)
        #     print("file_path_2 : ",file_path_2)
            

    def compare_pb_tensor(self, pb_tensor_1, pb_tensor_2):
        print("Not implemented yet")
        pass

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

    def _parse_from_string(self, pb_file, proto):
        # Load the TensorProto
        with open(pb_file, "rb") as f:
            proto.ParseFromString(f.read())
        return proto

    def _parse_pb_path(self, pb_file_path):
        pb = MetaData()
        proto_tensor = self._parse_from_string(pb_file_path, pb)
        return proto_tensor


class MetricData(object):
    
    def __init__(self):
        self.input = []
        self.output = []
        
    def __repr__(self) -> str:
        return f"input: {self.input} , output: {self.output}"

class Evaluation(object):
    
    def __init__(self, config):
        config = handle_config(config)
        self.evaluation_metrics = config.get('evaluation_metrics', [])
        self.registered_evaluations = dict()
        if "CS" in self.evaluation_metrics:
            self.register_evaluation("CS", self.evaluate_cosine_similarity)
        if "MSE" in self.evaluation_metrics:
            self.register_evaluation("MSE", self.evaluate_mean_squared_error)
        if "RMSE" in self.evaluation_metrics:
            self.register_evaluation("RMSE", self.evaluate_root_mean_squared_error)
        if "MAPE" in self.evaluation_metrics:
            self.register_evaluation("MAPE", self.evaluate_mean_absolute_percentage_error)
        
    def register_evaluation(self, fn_name, evaluation_fn):
        self.registered_evaluations[fn_name] = evaluation_fn
    
    def evaluate_cosine_similarity(self, data_1, data_2):
        # module_name : Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), input.type : <class 'tuple'>, output.type : <class 'torch.Tensor'>
        
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
                    cos = _get_tensor_cosine_similarity(input_1, input_2, data_1, data_2)
                    metric.input.append(cos)
                elif isinstance(input_1, list):                    
                    cos_list = []
                    for inp1, inp2 in zip(input_1, input_2):
                        cos_list.append(_get_tensor_cosine_similarity(inp1, inp2, data_1, data_2))
                    metric.input.append(cos_list)
                # else:
                #     print("Invalid input type : ", type(input_1))
            if isinstance(data_1.output, torch.Tensor):
                cos = _get_tensor_cosine_similarity(data_1.output, data_2.output, data_1, data_2)
                metric.output.append(cos)
            elif isinstance(data_1.output, list) or isinstance(data_1.output, tuple):
                cos_list = []
                for out_1, out_2 in zip(data_1.output, data_2.output):
                    cos_list.append(_get_tensor_cosine_similarity(out_1, out_2, data_1, data_2))
                metric.output.append(cos_list)
            else:
                print("Invalid output type : ", type(data_1.output))
            return metric
        return _get_struct_cosine_similarity(data_1, data_2)
        
    def evaluate_mean_squared_error(self, data_1, data_2):
        data_1 = data_1.reshape(1, -1)
        data_2 = data_2.reshape(1, -1)
        loss = nn.MSELoss()
        return loss(data_1, data_2)
    
    def evaluate_root_mean_squared_error(self, data_1, data_2):
        pass
    
    def evaluate_mean_absolute_percentage_error(self, data_1, data_2):
        pass
    
    def evalute(self, data_1, data_2):
        metrics = {}
        for fn_name, evaluation_fn in self.registered_evaluations.items():
            metrics[fn_name] = evaluation_fn(data_1, data_2)
        return metrics
