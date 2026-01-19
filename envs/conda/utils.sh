#!/bin/bash
#
# Helper functions for building a conda environment
#
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDA_DIR}/conda_env_info.sh"
source "${COLOR_MAP_PATH}"
source "${EXIT_CODE_PATH}"


initialize_conda() {
  local CONDA_BASE=$(conda info --base)
  local CONDA_DIRS=(
    "$CONDA_BASE"
    "/opt/conda"
    "/opt/miniconda"
    "/opt/miniconda2"
    "/opt/miniconda3"
  )

  local IS_CONDA_FOUND=false
  for dir in "${CONDA_DIRS[@]}"; do
    if [ -f "$dir/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="$dir"
      IS_CONDA_FOUND=true
      break
    fi
  done

  if ! $IS_CONDA_FOUND; then
    echo -e "${FG_RED}No Conda environment found matching${FG_RESET}"  >&2
    return ${ERROR_EXITCODE}
  fi

  echo -e "${FG_YELLOW}Intializing conda${FG_RESET}"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
}


find_conda_env_path() {
  # Will return `CONDA_ENV_DIR`
  local ENV_NAME=$1

  initialize_conda
  IFS=' ' read -r -a CONDA_INFO <<<"$(conda env list | grep "${ENV_NAME}")"

  if [ ${#CONDA_INFO[@]} -eq 0 ]; then
    if [ -n "${FG_RED}" ]; then
      echo -e "${FG_RED}No Conda environment found matching '${ENV_NAME}'${FG_RESET}"
    else
      echo "No Conda environment found matching '${ENV_NAME}'"
    fi
    return ${ERROR_EXITCODE:-1}
  fi

  AVAILABLE_ENV_NAME="${CONDA_INFO[0]}"

  if [[ "x${AVAILABLE_ENV_NAME}x" != "x${ENV_NAME}x" ]]; then
    if [ -n "${FG_RED}" ]; then
      echo -e "${FG_RED}Conda Env '${ENV_NAME}' is not available${FG_RESET}"
    else
      echo "Conda Env '${ENV_NAME}' is not available"
    fi
    return ${ERROR_EXITCODE:-1}
  fi

  if [ "x${CONDA_INFO[1]}x" == "x*x" ]; then
    CONDA_ENV_DIR="${CONDA_INFO[2]}"
  else
    CONDA_ENV_DIR="${CONDA_INFO[1]}"
  fi
  # DO NOT REMOVE
  echo "${CONDA_ENV_DIR}"
}


initialize_conda_env() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  conda create -c conda-forge -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  source activate "${CONDA_ENV}"
  pip install --no-cache-dir "poetry==${POETRY_VERSION}"
  conda deactivate
}

retry_to_find_conda_env_path() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  if [ "x${PYTHON_VERSION}x" == "xx" ]; then
    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
  fi

  initialize_conda_env "${CONDA_ENV}" "${PYTHON_VERSION}"

  find_conda_env_path "${CONDA_ENV}"
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    echo -e "${FG_RED}Unknown exception occurs from the side of Conda infra${FG_RESET}"
  fi

}

install_python_package() {
  local TARGET_PROJECT_DIR=$1

  pushd "${TARGET_PROJECT_DIR}" || exit

  if [ -d "${PWD}"/dist/ ]; then
    FILE_COUNT=$(ls "${PWD}/dist/*" 2>/dev/null | wc -l)
    if [ "x${FILE_COUNT//[[:space:]]/}x" != "x0x" ]; then
      echo -e "${FG_YELLOW}Removing ${PWD}/dist/* files${FG_RESET}"
      rm "${PWD}/dist/*"
    fi
  fi

  echo -e "${FG_YELLOW}Installing python package${FG_RESET}"
  poetry lock --no-update
  poetry install --no-root

  # Check if README.md exists
  if [ ! -f README.md ]; then
    echo "Temporary README.md" > README.md
    TEMP_README=true
  else
    TEMP_README=false
  fi

  # Build the project using poetry
  poetry build

  # Install the built package if the build was successful
  if [ -d "${PWD}/dist/" ]; then
    pip install dist/*.tar.gz
    rm -r dist
  else
    echo -e "Failed to install python package"
  fi

  # Remove the temporary README.md if it was created
  if [ "$TEMP_README" = true ]; then
    rm README.md
  fi

  popd || exit
}


activate_conda_environment() {
  local CONDA_ENV=$1
  initialize_conda
  if [ "$(command -v conda)" ]; then
    echo -e "${FG_YELLOW}Activating Conda Env: ${CONDA_ENV}${FG_RESET}"
    conda activate ${CONDA_ENV}
  else
    echo -e "${FG_RED}Activation Failed. Conda is not installed.${FG_RESET}"
  fi
}

update_conda_environment() {
  local PACKAGE_BASE_PATH=$1
  local CONDA_ENV=$2

  if [ "$(command -v conda)" ]; then
    echo -e "${FG_YELLOW}Updating Conda environment - ${CONDA_ENV}${FG_RESET}"
    bash "${CONDA_DIR}/build_conda_env.sh" --conda_env ${CONDA_ENV}
  else
    echo -e "${FG_RED}Update Failed. Conda is not installed.${FG_RESET}"
  fi
}
