#!/bin/bash
# 设置本地包的 PYTHONPATH
export PYTHONPATH="/home/lizilin/code/hydrodatasource:/home/lizilin/code/hydroutils:/home/lizilin/code/hydrodataset:/home/lizilin/code/hydroevaluate:/home/lizilin/code/hydromodel:/home/lizilin/code/hydrotopo:/home/lizilin/code/hydroweacast:$PYTHONPATH"

echo "PYTHONPATH has been set to:"
echo $PYTHONPATH

# 运行传入的命令
if [ $# -gt 0 ]; then
    exec "$@"
fi
