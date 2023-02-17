def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        from .rmg.RMG import RMG
    elif _backend == "pytorch":
        from .gcmc.GCMC import GCMC
        from .egcf.EGCF import EGCF
        from .uuii_mf.UUIIMF import UUIIMF
        from .uuii_ncf.UUIINCF import UUIINCF
        from .uuii_gcmc.UUIIGCMC import UUIIGCMC
        from .ncf.NCF import NCF
        from .mf.MF import MF
