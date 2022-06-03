#!/bin/bash
[ -f ./path.sh ] && . ./path.sh
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
echo "Number of GPUs you need: 4"
cd /home/work_nfs4_ssd/azhang/workspace/wenet/wenet-main/examples/librispeech/s0
CUDA_VISIBLE_DEVICES=`nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print NR-1,$0}' | sort -k 2 -n -r | cut -d ' ' -f 1 | head -4 | perl -pe 'chop if eof' | tr '\n' ','`
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash run.sh 
EOF
) >ori_librispeech
time1=`date +"%s"`
 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash run.sh  ) 2>>ori_librispeech >>ori_librispeech
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>ori_librispeech
echo '#' Finished at `date` with status $ret >>ori_librispeech
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.1122
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/ori_librispeech -l gpu=1 -q g.q -pe smp 4   /home/work_nfs4_ssd/azhang/workspace/wenet/wenet-main/examples/librispeech/s0/./q/ori_librispeech.sh >>./q/ori_librispeech 2>&1
