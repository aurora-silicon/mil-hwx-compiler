#!/usr/bin/env bash
set -eo pipefail

root=${GNUSTEP_BUILD_ROOT:-${TMPDIR:-/tmp}/mil-hwx-gnustep}
prefix=${GNUSTEP_PREFIX:-$HOME/.local/mil-hwx-gnustep}
sources="$root/src"
build="$root/build"
jobs=${JOBS:-$(nproc)}
cc=${CC:-}
cxx=${CXX:-}
if [[ -z "$cc" || -z "$cxx" ]]; then
    for version in 18 17 16 15 14; do
        if command -v "clang-$version" >/dev/null &&
           command -v "clang++-$version" >/dev/null; then
            cc=$(command -v "clang-$version")
            cxx=$(command -v "clang++-$version")
            break
        fi
    done
fi
if [[ -z "$cc" || -z "$cxx" ]]; then
    cc=$(command -v clang || true)
    cxx=$(command -v clang++ || true)
fi
linker=$(command -v ld.lld-18 || command -v ld.lld-17 ||
    command -v ld.lld-16 || command -v ld.lld-15 ||
    command -v ld.lld-14 || command -v ld.lld || true)
linker_flavor=${linker##*/}
linker_flavor=${linker_flavor#ld.}

objc_commit=a1faad66cee79f29c4ca268b004111edc33a2ae2
make_commit=d0349cc83a23367acdc53df364f73504c281b26c
base_commit=3d7013e144b153abcd6c088dc8d6c64ebd43bc4b

for command in "$cc" "$cxx" "$linker" cmake ninja git make pkg-config; do
    command -v "$command" >/dev/null || {
        echo "missing build tool: $command" >&2
        exit 1
    }
done
for package in libffi libxml-2.0 icu-uc openssl zlib; do
    pkg-config --exists "$package" || {
        echo "missing development package: $package" >&2
        exit 1
    }
done

mkdir -p "$sources" "$build" "$prefix"
checkout() {
    local name=$1 url=$2 commit=$3
    if [[ ! -d "$sources/$name/.git" ]]; then
        git clone --filter=blob:none "$url" "$sources/$name"
    fi
    git -C "$sources/$name" fetch --depth=1 origin "$commit"
    git -C "$sources/$name" checkout --detach "$commit"
}

checkout libobjc2 https://github.com/gnustep/libobjc2.git "$objc_commit"
checkout tools-make https://github.com/gnustep/tools-make.git "$make_commit"
checkout libs-base https://github.com/gnustep/libs-base.git "$base_commit"

cmake -S "$sources/libobjc2" -B "$build/libobjc2" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_C_COMPILER="$cc" \
    -DCMAKE_CXX_COMPILER="$cxx" \
    -DTESTS=OFF
cmake --build "$build/libobjc2" -j "$jobs"
cmake --install "$build/libobjc2"

export PATH="$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export CC="$cc" CXX="$cxx" OBJC="$cc" OBJCXX="$cxx"
export CPPFLAGS="-I$prefix/include"
export LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib -fuse-ld=$linker_flavor"

(
    cd "$sources/tools-make"
    ./configure --prefix="$prefix" --with-layout=fhs \
        --with-library-combo=ng-gnu-gnu --with-runtime-abi=gnustep-2.0 \
        --enable-objc-arc --enable-native-objc-exceptions
    make -j "$jobs"
    make install
)

export GNUSTEP_CONFIG_FILE="$prefix/etc/GNUstep/GNUstep.conf"
. "$prefix/share/GNUstep/Makefiles/GNUstep.sh"
(
    cd "$sources/libs-base"
    ./configure --prefix="$prefix" --disable-tls --disable-xslt \
        --disable-zeroconf --disable-libdispatch --disable-nsurlsession
    make -j "$jobs"
    make install
)

printf 'GNUSTEP_PREFIX=%s\n' "$prefix"
printf 'libobjc2=%s\ngnustep-make=%s\ngnustep-base=%s\n' \
    "$objc_commit" "$make_commit" "$base_commit"
