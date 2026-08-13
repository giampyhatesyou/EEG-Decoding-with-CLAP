from .clip_loss import CLIP_Loss

__all__ = ["CLIP_Loss", "EEGContrastiveLearning", "SupervisedClassification"]

_LAZY = {"EEGContrastiveLearning": ".contrastive_learning",
         "SupervisedClassification": ".supervised_classification"}  # CHANGED(baseline): diagnostic unimodal supervised baselines (#1/#2)


# CHANGED(baseline): the two LightningModules are resolved on first ACCESS (PEP 562), not at
# package import. They pull pytorch_lightning, which the CPU dev box does not have, and that
# made `from modules.clip_loss import CLIP_Loss` -- the Chapter 1 canary itself, and the
# MAD-EEG arm -- unimportable there for a reason unrelated to either. Same objects, same
# ImportError if lightning is genuinely missing when Chapter 1 runs; only later.
# `contrastive_learning` does `from . import CLIP_Loss`, which still resolves: CLIP_Loss is
# bound above, before anything can trigger this hook.
def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
