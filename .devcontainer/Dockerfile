# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.231.6/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3
# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT="3.9"

# [Option] Install Node.js
ARG INSTALL_NODE="true"
ARG NODE_VERSION="lts/*"
RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN conda install mamba -n base -c conda-forge
# install package dependencies
RUN conda install --file /tmp/pip-tmp/requirements.txt -c conda-forge && rm -rf /tmp/py-tmp
#RUN conda install --file /tmp/pip-tmp/requirements.txt -c conda-forge && rm -rf /tmp/py-tmp
# install dev dependencies
RUN mamba install pytest pytest-cov tox mock commitizen pre-commit pip-tools jupyter -c conda-forge

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
