def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break

from .convmf import ConvMF

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        from .rmg.RMG import RMG
    elif _backend == "pytorch":
        from .gcmc.GCMC import GCMC
        from .egcfv2.EGCFv2 import EGCFv2
