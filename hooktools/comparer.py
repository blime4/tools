from contextlib import contextmanager
import io
import os
import copy

import time
import json
import hashlib
from hooktools.utils import handle_config, get_pb_file_list
from pathlib import Path
from tqdm import tqdm
from hooktools.trace_pb2 import HookData, MetaData, TensorProto, IOData
from hooktools.utils import to_array


class Comparer(object):

    def __init__(self, config):
        """
        config : yaml type configuration
        """
        config = handle_config(config)
        self.log_dir = config.get('log_dir', "./tmp")
        self.comparer_name = config.get('comparer_name', "")
        self.compare_mode = config.get('compare_mode', 0)
        # compare mode:
        # 0 : compare directory
        # 1 : compare protobuf file

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

            self.compared_pb_file_1 = ""
            self.compared_pb_file_2 = ""

        elif self.compare_mode == 1:
            compare_options = config.get('compare_pb_file_options', {})
            self.compared_pb_file_1 = compare_options.get("compared_pb_file_1")
            self.compared_pb_file_2 = compare_options.get("compared_pb_file_2")

        self.verbose = compare_options.get('verbose', False)
        if self.verbose:
            print(config)

    # TODO: complete compare mode and compare precision way.

    def compare(self):
        if self.compare_mode == 0:
            self.compare_directory()
        elif self.compare_mode == 1:
            self.compare_pb_file()

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

                pb_filelist_1 = get_pb_file_list(step_path_1)
                pb_filelist_2 = get_pb_file_list(step_path_2)

                self.compare_pb_filelist(pb_filelist_1, pb_filelist_2)

    def compare_pb_filelist(self, pb_filelist_1, pb_filelist_2):

        both_pb_file_list_1, both_pb_file_list_2 = self._get_both_pb_filelist(
            pb_filelist_1, pb_filelist_2)
        for pb1, pb2 in tqdm(zip(both_pb_file_list_1, both_pb_file_list_2)):
            self.compare_pb_file(pb1, pb2)

    def compare_pb_file(self, pb_file_path_1, pb_file_path_2):
        if not pb_file_path_1 and not pb_file_path_2:
            pb_file_path_1 = self.compared_pb_file_1
            pb_file_path_2 = self.compared_pb_file_2

        pb_tensor_1 = self._parse_pb_path(pb_file_path_1)
        pb_tensor_2 = self._parse_pb_path(pb_file_path_2)
        self.compare_pb_tensor(pb_tensor_1, pb_tensor_2)

    def compare_pb_tensor(self, pb_tensor_1, pb_tensor_2):
        # TODO: compare way :
        pass

    def _check_path_exists(self, path):
        if not os.path.exists(path):
            raise path + "not exists!"

    def _get_both_pb_filelist(self, pb_filelist_1, pb_filelist_2):

        if not pb_filelist_1 or not pb_filelist_2:
            return [], []

        pbl1 = [os.path.basename(pb) for pb in pb_filelist_1]
        pbl2 = [os.path.basename(pb) for pb in pb_filelist_2]

        only_in_pb_file_list_1 = pbl1 - pbl2
        only_in_pb_file_list_2 = pbl2 - pbl1
        both_pb_file_list = pbl1 | pbl2

        if only_in_pb_file_list_1 != set():
            print("These files exist only in pb_file_list_1 : ",
                  only_in_pb_file_list_1)
        if only_in_pb_file_list_2 != set():
            print("These files exist only in pb_file_list_2 : ",
                  only_in_pb_file_list_2)

        both_pb_file_list_1 = [os.path.join(os.path.dirname(
            pb_filelist_1[0]), pb) for pb in both_pb_file_list]
        both_pb_file_list_2 = [os.path.join(os.path.dirname(
            pb_filelist_2[0]), pb) for pb in both_pb_file_list]

        return both_pb_file_list_1, both_pb_file_list_2

    def _parse_from_string(self, pb_file, proto):
        # Load the TensorProto
        with open(pb_file, 'rb') as f:
            proto.ParseFromString(f.read())
        return proto

    def _parse_pb_path(self, pb_file_path):
        pb = MetaData()
        proto_tensor = self._parse_from_string(pb_file_path, pb)
        return proto_tensor



