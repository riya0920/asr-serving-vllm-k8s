"""Make vLLM's draft-model speculative decoding tolerate AUDIO multimodal targets.

vllm/v1/spec_decode/llm_base_proposer.py assumes every multimodal target is a VISION model:
it reads target_model.config.image_token_index unconditionally in the else branch, then calls
get_language_model() on it. Whisper is multimodal by audio - there is no image placeholder
token and no separate 'language model' submodule - so both assumptions raise.

This is a wrong-assumption bug, not a missing feature. The patch makes both steps optional and
leaves every vision path untouched.
"""
import re, sys, shutil, site, pathlib

sp = pathlib.Path(site.getsitepackages()[0])
f = sp / "vllm/v1/spec_decode/llm_base_proposer.py"
src = f.read_text()
shutil.copy(f, str(f) + ".orig")

# 1. image_token_index: only set it if the target actually has one.
old1 = """            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )"""
new1 = """            else:
                # PATCH: audio multimodal targets (Whisper) have no image placeholder
                # token. Only propagate the index when the target actually defines one.
                _img_idx = getattr(target_model.config, "image_token_index", None)
                if _img_idx is not None:
                    self.model.config.image_token_index = _img_idx"""
if old1 not in src:
    sys.exit("FAIL: image_token_index block not found - vllm version differs")
src = src.replace(old1, new1, 1)

# 2. get_language_model(): encoder-decoder audio models do not expose one.
old2 = """            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()"""
new2 = """            # PATCH: encoder-decoder audio models have no separate language submodule.
            try:
                target_language_model = cast(
                    SupportsMultiModal, target_model
                ).get_language_model()
            except (NotImplementedError, AttributeError):
                target_language_model = target_model"""
if old2 not in src:
    sys.exit("FAIL: get_language_model block not found")
src = src.replace(old2, new2, 1)

f.write_text(src)
import ast
ast.parse(src)
print("PATCHED and parses:", f)
print("backup at:", str(f) + ".orig")
