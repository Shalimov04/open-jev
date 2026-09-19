#!/usr/bin/env bash
# Label-only chain: keeps the teacher busy. Writes runs/<task>/label.log and a .labeled marker.
cd "$(dirname "$0")/.." && export PATH=$PWD/.venv/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
for t in "${@:-agnews kinopoisk georeview toxic headlines banking77}"; do
  for tt in $t; do
    mkdir -p runs/$tt
    openjev label tasks/$tt.yaml >> runs/$tt/label.log 2>&1 && touch runs/$tt/.labeled
  done
done
