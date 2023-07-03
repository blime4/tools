from flask import Flask, request, render_template, send_file, jsonify
from hooktools import Comparer

import os
import datetime
import torch
import io
import contextlib

class DirectorysConfig(object):

    def __init__(self,
                 compared_directory_1,
                 compared_directory_2,
                 compare_folder_name,
                 compare_epochs,
                 compare_steps,
                 evaluation_metrics,
                 filter,
                 ) -> None:
        self.compare_directory_options = {
            'compared_directory_1':compared_directory_1,
            'compared_directory_2':compared_directory_2,
            'compare_folder_name':compare_folder_name,
            'compare_epochs':compare_epochs,
            'compare_steps':compare_steps,
            # 'compare_verbose':True,

        }
        self.evaluation_metrics=[evaluation_metrics]
        self.registersi_signal=False
        self.filter = {
            "global_filter":filter,
        }
        self.compare_by_order = True,

    def __call__(self):
        print("For debug : DirectorysConfig : ", vars(self))
        return vars(self)

class FilesConfig(object):
    def __init__(self,
                 compared_file_1,
                 compared_file_2,
                 evaluation_metrics,
                 filter,
                 ) -> None:
        self.compare_file_options = {
            'compared_file_1':compared_file_1,
            'compared_file_2':compared_file_2,
        }
        self.evaluation_metrics=[evaluation_metrics]
        self.registersi_signal=False
        self.filter = {
            "global_filter":filter,
        }
        self.compare_by_order=True
        self.compare_mode = 1
        self.auto_conclusion = False

    def __call__(self):
        print("For debug : FilesConfig : ", vars(self))
        return vars(self)

app = Flask(__name__, template_folder='templates')
output = ""  # 输出结果的全局变量
epochs = []
steps = []

global_directorys_comparer = None
global_files_comparer = None

@app.route('/', methods=['GET', 'POST'])
def compare_directorys():
    global output
    global epochs
    global steps
    global global_directorys_comparer
    if request.method == 'POST':
        compared_directory_1 = request.form['compared_directory_1']
        compared_directory_2 = request.form['compared_directory_2']
        compare_folder_name = request.form['compare_folder_name']
        compare_epochs = request.form['compare_epochs']
        compare_steps = request.form['compare_steps']
        filter = request.form['filter_number']
        # TODO: 临时方案：为了保存 log文件， 因为还不太熟悉flask
        epochs = compare_epochs
        steps = compare_steps
        # module_name = request.form['module_name']
        evaluation_metrics= request.form['evaluation_metrics']

        config = DirectorysConfig(
            compared_directory_1=compared_directory_1,
            compared_directory_2=compared_directory_2,
            compare_folder_name=compare_folder_name,
            compare_epochs=compare_epochs,
            compare_steps=compare_steps,
            evaluation_metrics=evaluation_metrics,
            filter=filter,
        )

        if not os.path.exists(compared_directory_1) or not os.path.exists(compared_directory_2):
            return render_template('error.html', message='指定的文件夹不存在')

        with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
            global_directorys_comparer = Comparer(config())
            global_directorys_comparer.compare()
            output = buffer.getvalue()

    return render_template('index.html', output=output)

@app.route('/export', methods=['POST'])
def export_log():
    global epochs
    global steps
    output = request.form['output']
    # epochs = request.form['epochs_from_first_form']
    # steps = request.form['steps_from_first_form']

    # 生成唯一的文件名，例如 "log_20230625_125430.txt"
    filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch{epochs}_step{steps}.txt"
    print("saving to ", filename)

    # 将输出结果保存到日志文件
    with open(filename, 'w') as file:
        file.write(output)

    return send_file(filename, as_attachment=True)

def get_conclusion_pk_file():
    if global_directorys_comparer is None:
        return find_latest_conclusion_file()
    return global_directorys_comparer.get_latest_conclusion_pk_filename()

def find_latest_conclusion_file():
    import glob
    current_path = os.getcwd()
    file_list = glob.glob(os.path.join(current_path, 'topk*pk'))
    if not file_list:
        return None
    latest_file = max(file_list, key=os.path.getmtime)
    return latest_file

@app.route('/get_conclusion', methods=['GET'])
def get_conclusion():
    if global_directorys_comparer is not None:
        with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
            global_directorys_comparer.conclusion(save_pk=False)
            return buffer.getvalue()
    else:
        return "Global comparer is None."

@app.route('/analyze_conclusion', methods=['POST'])
def analyze_conclusion():
    data = request.get_json()
    conclusion_file_path = data['conclusionFilePath']
    topk = data.get('topk', None)
    seq = data.get('seq', None)

    data = torch.load(conclusion_file_path)
    if topk is not None:
        data = data[:topk]
    if seq is not None:
        data = data[seq]

    import pandas as pd
    df = pd.DataFrame(data)
    excel_file_path = f'{os.path.splitext(os.path.basename(conclusion_file_path))[0]}.xlsx'

    df.to_excel(excel_file_path, index=False)
    print(df)
    return jsonify({'result': data.__repr__()})

def get_neighbor_file(file, offset):
    file_dir = os.path.dirname(file)
    file_name = os.path.basename(file)

    file_number = int(file_name.split('_')[-1].split('.')[0])
    neighbor_number = file_number + offset

    neighbor_file_name = file_name.replace(str(file_number).zfill(6), str(neighbor_number).zfill(6))

    neighbor_file_path = os.path.join(file_dir, neighbor_file_name)

    if os.path.exists(neighbor_file_path):
        return neighbor_file_path
    else:
        parent_path = os.path.abspath(os.path.dirname(os.path.dirname(file)))
        sub_dirs = [os.path.join(parent_path, path) for path in os.listdir(parent_path)]
        for sub_dir in sub_dirs:
            if os.path.isdir(sub_dir):
                neighbor_file_path = os.path.join(sub_dir, neighbor_file_name)

                if os.path.exists(neighbor_file_path):
                    return neighbor_file_path

    return None
def get_previous_file(file):
    return get_neighbor_file(file, -1)

def get_next_file(file):
    return get_neighbor_file(file, 1)

@app.route('/compare_files', methods=['GET', 'POST'])
def compare_files():
    global global_files_comparer
    if request.method == 'POST':
        compared_file_1 = request.form.get('compared_file_1')
        compared_file_2 = request.form.get('compared_file_2')
        file_evaluation_metrics = request.form.get('file_evaluation_metrics')
        file_filter_number = request.form.get('file_filter_number')

        if 'compare_file_button' in request.form:
            pass
        elif 'compare_previous_file_button' in request.form:
            compared_file_1 = get_previous_file(compared_file_1)
            compared_file_2 = get_previous_file(compared_file_2)
        elif 'compare_next_file_button' in request.form:
            compared_file_1 = get_next_file(compared_file_1)
            compared_file_2 = get_next_file(compared_file_2)

        config = FilesConfig(
            compared_file_1=compared_file_1,
            compared_file_2=compared_file_2,
            evaluation_metrics=file_evaluation_metrics,
            filter=file_filter_number,
        )
        global_files_comparer = Comparer(config())
        with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
            global_files_comparer.compare()
            file_output = buffer.getvalue()

        return render_template('index.html', file_output=file_output, compared_file_1=compared_file_1,
                               compared_file_2=compared_file_2, file_evaluation_metrics=file_evaluation_metrics,
                               file_filter_number=file_filter_number)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)