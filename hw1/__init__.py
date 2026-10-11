"""CSE470 HW1 - COPD detection from lung sounds (inspired by Srivastava et al., PeerJ CS 2021)."""
import warnings

# Harmless library-internal warnings that would only clutter the console output.
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
