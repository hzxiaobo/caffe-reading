#!/bin/bash
export LD_LIBRARY_PATH=/home/xiaobo/work/caffe-cpu/build/lib:/usr/lib/x86_64-linux-gnu:/home/xiaobo/work/caffe-test/new-version/target/src:/usr/local/lib
{
        echo "********************* Starting process $2 at `date`! ******************"
        ./testcaffe
}



exit 0
