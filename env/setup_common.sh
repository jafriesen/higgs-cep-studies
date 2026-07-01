#!/usr/bin/env bash

_higgs_cep_prepend_path() {
  local var_name="$1"
  local path_to_add="$2"
  local current_value="${!var_name:-}"

  [[ -n "$path_to_add" ]] || return 0
  case ":${current_value}:" in
    *":${path_to_add}:"*) return 0 ;;
  esac

  if [[ -n "$current_value" ]]; then
    export "${var_name}=${path_to_add}:${current_value}"
  else
    export "${var_name}=${path_to_add}"
  fi
}

_higgs_cep_common_main() {
  local env_dir
  env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  export HIGGS_CEP_STUDIES_DIR="$(cd "$env_dir/.." && pwd)"
  export REPO_ROOT_DIR="$(cd "$HIGGS_CEP_STUDIES_DIR/.." && pwd)"
}

_higgs_cep_common_main
