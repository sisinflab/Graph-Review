def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break

from .convmf import ConvMF
from .convmf import ConvMF
from .hrdr import HRDR

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        from .hrdr.HRDR import HRDR
        from .deepconn.DeepCoNN import DeepCoNN
        from .deepconnpp.DeepCoNNpp import DeepCoNNpp
        from .rmg.RMG import RMG
        from .narre.NARRE import NARRE
    elif _backend == "pytorch":
        from .gcmc.GCMC import GCMC
        from .egcf.EGCF import EGCF
        from .egcfv2.EGCFv2 import EGCFv2
