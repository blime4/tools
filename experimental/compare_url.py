from flask import Flask, request, render_template, send_file, jsonify
from hooktools import Comparer
# from hooktools.utils import ConfigFromFlask
import os
import datetime
import torch

class ConfigFromFlask(object):

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
            'compare_epochs':[compare_epochs],
            'compare_steps':[compare_steps],
        }
        self.evaluation_metrics=[evaluation_metrics]
        self.registersi_signal=False
        self.filter = {
            "global_filter":filter,
        }
        self.compare_by_order=True

    def __call__(self):
        print("For debug : ConfigFromFlask : ", vars(self))
        return vars(self)

app = Flask(__name__, template_folder='templates')
output = ""  # 输出结果的全局变量
epochs = []
steps = []

global_comparer = None

@app.route('/', methods=['GET', 'POST'])
def compare_files():
    global output
    global epochs
    global steps
    global global_comparer
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

        config = ConfigFromFlask(
            compared_directory_1=compared_directory_1,
            compared_directory_2=compared_directory_2,
            compare_folder_name=compare_folder_name,
            compare_epochs=compare_epochs,
            compare_steps=compare_steps,
            # module_name=module_name,
            evaluation_metrics=evaluation_metrics,
            filter=filter,
        )

        if not os.path.exists(compared_directory_1) or not os.path.exists(compared_directory_2):
            return render_template('error.html', message='指定的文件夹不存在')

        import io
        import contextlib
        with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
            global_comparer = Comparer(config())
            global_comparer.compare()
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

@app.route('/get_conclusion', methods=['GET'])
def get_conclusion():
    if global_comparer is None:
        return find_latest_conclusion_file()
    return global_comparer.evaluator.filter.get_latest_conclusion_pk_filename()

def find_latest_conclusion_file():
    import glob
    current_path = os.getcwd()
    file_list = glob.glob(os.path.join(current_path, 'topk*pk'))
    if not file_list:
        return None
    latest_file = max(file_list, key=os.path.getmtime)
    return latest_file

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

if __name__ == '__main__':
    app.run(debug=True)