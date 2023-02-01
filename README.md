# Hooktools

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