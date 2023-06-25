from flask import Flask, request, render_template, send_file
from hooktools import Comparer
from hooktools.utils import ConfigFromFlask
import os
import datetime

app = Flask(__name__, template_folder='templates')
output = ""  # 输出结果的全局变量

@app.route('/', methods=['GET', 'POST'])
def compare_files():
    global output
    if request.method == 'POST':
        compared_directory_1 = request.form['compared_directory_1']
        compared_directory_2 = request.form['compared_directory_2']
        compare_folder_name = request.form['compare_folder_name']
        compare_epochs = request.form['compare_epochs']
        compare_steps = request.form['compare_steps']
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
        )

        if not os.path.exists(compared_directory_1) or not os.path.exists(compared_directory_2):
            return render_template('error.html', message='指定的文件夹不存在')

        import io
        import contextlib
        with io.StringIO() as buffer, contextlib.redirect_stdout(buffer):
            comparer = Comparer(config())
            comparer.compare()
            output = buffer.getvalue()

    return render_template('index.html', output=output)


def compare(actual, desired):

    def evaluate_l1_loss(actual, desired):
        # siyi's way
        try:
            actual = actual.cpu() if actual.is_cuda else actual
            desired = desired.cpu() if desired.is_cuda else desired
            l1_error = (actual - desired).float().abs().mean()
            rel_error = l1_error / (actual.abs().float().mean())
            if l1_error * rel_error > 10:
                print('\n###\n', 'should checked!', '\n###\n')
            return (l1_error.detach(), rel_error.detach())

        except Exception as e:
            print("ERROR : ", e)
            raise "failed."
    return evaluate_l1_loss(actual, desired)

@app.route('/export', methods=['POST'])
def export_log():
    output = request.form['output']
    epoch = request.form['epochs_from_first_form']
    step = request.form['steps_from_first_form']

    # 生成唯一的文件名，例如 "log_20230625_125430.txt"
    filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{epoch}_{step}.txt"
    print("saving to ", filename)

    # 将输出结果保存到日志文件
    with open(filename, 'w') as file:
        file.write(output)

    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)