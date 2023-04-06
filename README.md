# Hooktools

# pip install
python setup.py bdist_wheel && pip install dist/hooktools-0.1-py3-none-any.whl --force-reinstall

## Tracer
### DumpPbFileTracer
### CheckNanTracer

### demo

```
from hooktools import Tracer
trace = Tracer("./hooktools/config/tracer_demo.yaml")
trace.trace(epoch, step)
trace.untrace()
```

## Comparer