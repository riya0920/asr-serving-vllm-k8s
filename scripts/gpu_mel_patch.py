"""Force Whisper's log-mel feature extraction onto the GPU.

THE HYPOTHESIS
--------------
On an H100 NVL, serving whisper-large-v3-turbo through vLLM peaked at 5.93 req/s with the
GPU at 5% utilization and `VLLM::EngineCore` burning 23 CPU cores (see ADR-004). The engine
is not waiting on the accelerator — it is computing log-mel spectrograms on the CPU, one
30-second clip at a time, inside the process that is supposed to be scheduling GPU work.

Whisper's front end is an STFT plus a mel filterbank. It is pure tensor math and belongs on
the GPU. HuggingFace's WhisperFeatureExtractor already implements both paths:

    _np_extract_fbank_features(...)      # numpy, CPU
    _torch_extract_fbank_features(...)   # torch, honours a device argument

and `__call__` picks between them based on its `device` kwarg, which defaults to "cpu".
Nothing upstream of it ever passes anything else.

WHY A sitecustomize PATCH
-------------------------
The server is launched as a CLI (`vllm-omni serve ...`), so there is no import site we
control. Python imports `sitecustomize` automatically at interpreter startup if it is on the
path, which makes it the one reliable hook into a process we do not own. Install this file as
`sitecustomize.py` in site-packages and every Python process on the box — including the
engine core workers vLLM forks — gets the patch.

WHAT COULD GO WRONG, AND WHAT THIS DOES ABOUT IT
------------------------------------------------
The feature extractor will now return tensors on the GPU. Anything downstream that expects
numpy or CPU tensors will break loudly rather than silently, which is what we want. Set
ASR_MEL_RETURN_CPU=1 to copy the result back to CPU after computing on the GPU — that still
moves the expensive STFT off the CPU while keeping the output type identical to before, so
it is the safer variant to try if the aggressive one errors.

Verification is not optional here. A patch that changes how audio becomes features can
change transcripts. Run the WER gate after applying this; if WER moves at all, the patch is
wrong regardless of how fast it made things.
"""

import os
import sys


def _install() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return

    try:
        from transformers.models.whisper.feature_extraction_whisper import (
            WhisperFeatureExtractor,
        )
    except Exception:
        return

    if getattr(WhisperFeatureExtractor, "_asr_gpu_mel_patched", False):
        return

    device = os.environ.get("ASR_MEL_DEVICE", "cuda")
    return_cpu = os.environ.get("ASR_MEL_RETURN_CPU", "0") == "1"
    original = WhisperFeatureExtractor.__call__

    def patched_call(self, *args, **kwargs):
        # Only override when the caller did not ask for a specific device. If vLLM ever
        # starts passing one itself, defer to it rather than fighting.
        if kwargs.get("device") in (None, "cpu"):
            kwargs["device"] = device
        out = original(self, *args, **kwargs)

        if return_cpu:
            try:
                for key in list(out.keys()):
                    val = out[key]
                    if hasattr(val, "device") and str(val.device) != "cpu":
                        out[key] = val.cpu()
            except Exception:
                pass
        return out

    WhisperFeatureExtractor.__call__ = patched_call
    WhisperFeatureExtractor._asr_gpu_mel_patched = True

    if os.environ.get("ASR_MEL_VERBOSE", "1") == "1":
        print(
            f"[gpu_mel_patch] WhisperFeatureExtractor -> device={device} "
            f"return_cpu={return_cpu}",
            file=sys.stderr,
        )


try:
    _install()
except Exception as exc:  # never break interpreter startup
    print(f"[gpu_mel_patch] not installed: {type(exc).__name__}: {exc}", file=sys.stderr)
