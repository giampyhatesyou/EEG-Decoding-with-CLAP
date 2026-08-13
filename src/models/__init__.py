from .model import Model
from .sample_cnn2d_eeg import SampleCNN2DEEG
from .spectra_eeg import SpectraEEG  # CHANGED(baseline): new export, the SpectraCLIP EEG front-end

__all__ = ["Model", "SampleCNN2DEEG", "SpectraEEG", "CLAPEncoder"]


# CHANGED(baseline): `CLAPEncoder` is now resolved on first ACCESS (PEP 562) instead of at
# package import. Same object, same name, same failure -- only later: `clap_encoder` pulls
# in torchaudio, which the CPU dev box does not have, and that made `import models` fail
# there and with it every EEG-side check that has nothing to do with CLAP. This is not a
# silent fallback: touching `models.CLAPEncoder` without torchaudio still raises the very
# same ImportError it raised before, at the point of use.
def __getattr__(name):
    if name == "CLAPEncoder":
        from .clap_encoder import CLAPEncoder
        return CLAPEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")