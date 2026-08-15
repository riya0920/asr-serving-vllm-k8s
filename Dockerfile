# vLLM-Omni serving image for Whisper-Large-v3.
#
# Weights are baked into the image rather than downloaded at startup. That trade is
# deliberate: it makes the image large (~4GB for large-v3 in fp16), but it removes a network
# download from the pod cold-start path, and cold start is the thing standing between the
# autoscaler and the SLO during a spike. It also makes the image content-addressed end to
# end — the digest identifies the code AND the weights, so "what is running in prod" has one
# answer instead of two.

ARG VLLM_OMNI_VERSION=latest
FROM vllm/vllm-omni:${VLLM_OMNI_VERSION}

ARG MODEL_ID=openai/whisper-large-v3
# Pinned commit SHA, never 'main'. A floating revision makes the build non-reproducible and
# quietly invalidates the WER baseline that CI gates against. ci/verify_weights.py enforces
# this at build time.
ARG MODEL_REVISION

ENV HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    VLLM_LOGGING_LEVEL=INFO

RUN test -n "$MODEL_REVISION" || (echo "MODEL_REVISION must be a pinned commit SHA" && exit 1)

# Fetch weights in their own layer so code changes do not re-download 4GB.
RUN --mount=type=secret,id=hf_token,required=false \
    HF_HUB_OFFLINE=0 python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', revision='${MODEL_REVISION}', \
    allow_patterns=['*.safetensors','*.json','*.txt','*.model'])"

COPY ci/verify_weights.py /opt/verify_weights.py
COPY model-manifest.json /opt/model-manifest.json

# Verify at BUILD time, so a corrupted download fails the build rather than shipping an
# image that starts fine and transcribes badly.
RUN python /opt/verify_weights.py --manifest /opt/model-manifest.json --cache /opt/hf/hub

EXPOSE 8000

# max-num-seqs is the batch slot count and must stay in sync with the KEDA headroom target
# in infra/keda/scaledobject.yaml (threshold 22 ~= 70% of 32). Changing one without the
# other silently breaks the warm-headroom policy that absorbs spikes.
ENTRYPOINT ["vllm-omni", "serve"]
CMD ["--model", "openai/whisper-large-v3", \
     "--task", "transcription", \
     "--port", "8000", \
     "--max-num-seqs", "32", \
     "--gpu-memory-utilization", "0.90"]
