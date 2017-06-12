#!/bin/bash
SCRIPTPATH="${BASH_SOURCE[0]}"
MAINDIR=`dirname $SCRIPTPATH`
MAINDIR=`readlink -f $MAINDIR`
INSTALLDIR="$MAINDIR/deps/install"
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$INSTALLDIR/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# optional:
#export PKG_CONFIG_PATH=$INSTALLDIR/lib/pkgconfig
#export PATH="$PATH:$INSTALLDIR/bin"
export OMP_NUM_THREADS=1
