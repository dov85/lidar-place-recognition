#!/usr/bin/env bash
# Clone the official repositories at the commits this project was written against.
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party
clone () {  # url dir commit
  if [ ! -d "third_party/$2" ]; then
    git clone -q "$1" "third_party/$2"
    git -C "third_party/$2" checkout -q "$3"
  fi
  echo "third_party/$2 @ $(git -C third_party/$2 rev-parse --short HEAD)"
}
clone https://github.com/dachengxiaocheng/NDT-Transformer.git NDT-Transformer 51f90dfa4cffc207bdac23d9c669046d8deb7ea6
clone https://github.com/zjuluolun/BEVPlace2.git BEVPlace2 2d0636d59fac6520bead4faf3f933be592bcd635
