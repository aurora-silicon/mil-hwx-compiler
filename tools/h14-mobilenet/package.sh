#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
case "$*" in *--frame*) ;; *) set -- "$@" --frame mirrored ;; esac
exec "${PYTHON:-python3}" "$root/tools/h13-mobilenet/mobilenet.py" "$@" --target H14
