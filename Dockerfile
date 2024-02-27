FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Update system
RUN apt update && apt install -y python3-pip && apt install -y python3-venv

WORKDIR /openbb-chat/

# Add dependencies
ADD pdm.lock /openbb-chat/pdm.lock
ADD pyproject.toml /openbb-chat/pyproject.toml
ADD openbb_chat /openbb-chat/openbb_chat
ADD README.md /openbb-chat/README.md
RUN pip install --no-cache-dir setuptools==68.2.2 wheel==0.41.3 pdm==2.12.3 && \
    pdm install --prod

# Add project root
ADD .project-root /openbb-chat/.project-root

# Add scripts, data and configs
ADD scripts /openbb-chat/scripts/
ADD data /openbb-chat/data/
ADD configs /openbb-chat/configs/

# Add mkdocs documentation
ADD docs /openbb-chat/docs
ADD mkdocs.yml /openbb-chat/mkdocs.yml

# Tests
ADD tests /openbb-chat/tests/

ENTRYPOINT ["/bin/bash"]

# IMPORTANT: To reduce start up time, download HF models and mount Docker with /root/.cache/huggingface/hub/ mapped to the local downloads
