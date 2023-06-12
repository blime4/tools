# Hooktools

# install
```bash
python setup.py bdist_wheel && pip install dist/hooktools-0.1-py3-none-any.whl --force-reinstall
```
# Usage
## Tracer
```python
import os
import hooktools
from hooktools import Tracer

model=TheNetYouTrace()                  # like model=Yolov3() or model=Bert()

config_path = os.path.join(hooktools.__path__[0], "config")
trace = Tracer(os.path.join(config_path, "tracer_demo.yaml"), model)
# or you can use the configuration file that you specify
# trace = Tracer(path_to_your_configuration_file)

trace.trace(epoch, step)
trace.untrace()
```

## Comparer
```python
import os
import hooktools
from hooktools import Comparer

config_path = os.path.join(hooktools.__path__[0], "config")
compare = Comparer(os.path.join(config_path, "comparer_demo.yaml"))
# or you can use the configuration file that you specify
# compare = Comparer(path_to_your_configuration_file)

compare.compare()
```

---
> some demo your can find in ./hooktools/model