#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2023 Johan Rohdin (rohdin@fit.vutbr.cz)

. ./path.sh || exit 1

stage=3
stop_stage=3

data=data
data_type="shard"  # shard/raw

#### -Johan
config=conf/resnet.yaml
exp_dir=exp/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150
#gpus="[0,1]"
gpus="[1]"
num_avg=10
checkpoint=

#trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
#score_norm_method="asnorm"  # asnorm/snorm
#top_n=300


## setup for large margin fine-tuning
lm_config=conf/resnet_lm.yaml
#### -Johan

. tools/parse_options.sh || exit 1


                 
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Prepare datasets ..."

    # Directory where the voxceleb data has been prepared with
    # ../../voxceleb/v2/local/prepare_data.sh. If left empty, the
    # preparation of VoxCeleb will be done here.
    vox_dir="/mnt/matylda6/rohdin/expts/wespeaker/wespeaker/examples/voxceleb/v2/data/"
    if [ "A" == "B" ];then
	
	if [ $vox_dir = "" ];then
	    echo "Preparing Voxceleb, rirs and Musan"
	    ../../voxceleb/v2/prepare_data.sh --stage 1 --stop_stage 4 --data ${data}
	    mv $data/vox1 $data/vox1_16k
	    mv $data/vox2_dev $data/vox2_dev_16k
	    mv $data/rirs $data/rirs_16k
	    mv $data/musan $data/musan_16k
	elif [ -d $vox_dir/vox1 ] && [ -d $vox_dir/vox2_dev ] && [ -d $vox_dir/rirs ] && [ -d $vox_dir/musan ];then
	    echo "Copying vox1, vox2_dev, rirs and musan from ${vox_dir}"
	    tools/copy_data_dir.sh $vox_dir/vox1 $data/vox1_16k                # Copy_data_dir should be renamed because it copies only some files.
	    tools/copy_data_dir.sh $vox_dir/vox2_dev $data/vox2_dev_16k
	    tools/copy_data_dir.sh $vox_dir/rirs $data/rirs_16k
	    tools/copy_data_dir.sh $vox_dir/musan $data/musan_16k
	else
	    echo "ERROR: Incorrectly specified Voxceleb data dirs."
	    exit 1
	fi
	# Note: "remove_prefix_wav" will be removed from the new wav filenames
	tools/downsample_audio.sh --src_dir $data/vox1_16k --dest_dir $data/vox1 --rate 8k --wav_dir `pwd`/wav/vox1 --remove_prefix_wav $vox_dir
	tools/downsample_audio.sh --src_dir $data/rirs_16k --dest_dir $data/rirs --rate 8k --wav_dir `pwd`/wav/rirs --remove_prefix_wav $vox_dir
	tools/downsample_audio.sh --src_dir $data/musan_16k --dest_dir $data/musan --rate 8k --wav_dir `pwd`/wav/musan --remove_prefix_wav $vox_dir
	tools/downsample_audio.sh --src_dir $data/vox2_dev_16k --dest_dir $data/vox2_dev --rate 8k --wav_dir `pwd`/wav/vox2_dev --remove_prefix_wav $vox_dir	
	#tools/combine_data_dir.sh $data/voxceleb $data/vox1 $data/vox2_dev

	tools/apply_gsm.sh --src_dir $data/vox1 --dest_dir $data/vox1_gsmfr --wav_dir `pwd`/wav/vox1_gsmfr --remove_prefix_wav `pwd`/wav/vox1 #--rate 8k
	tools/apply_gsm.sh --src_dir $data/vox2_dev --dest_dir $data/vox2_dev_gsmfr --wav_dir `pwd`/wav/vox2_dev_gsmfr --remove_prefix_wav `pwd`/wav/vox2_dev #--rate 8k
    fi

    
    if [ "A" == "B" ];then
	# Path to the directory with the SRE CTS Superset. 
	cts_superset_dir=/mnt/matylda2/data/LDC/LDC2021E08_SRE-CTS-Superset/
	local/prepare_cts_superset.sh --cts_superset_dir $cts_superset_dir --data_cts $data/cts --wav_dir `pwd`/wav/cts
	#local/prepare_cts_superset.sh --cts_superset_dir $cts_superset_dir --data_cts $data/cts --wav_dir /mnt/scratch/tmp/rohdin/wespeaker/wav/cts/
    fi

    if [ "A" == "B" ];then
	sre21_testset_dir=/mnt/matylda2/data/LDC/LDC2021E09_sre21_dev_set/
	sre21_devset_dir=/mnt/matylda2/data/LDC/LDC2021E10_sre21_eval_set/
	
	sre18_eval_set=/mnt/matylda2/data/LDC/LDC2018E51_2018_NIST_Speaker_Recognition_Evaluation_Test_Set/
	sre18_dev_set=/mnt/matylda2/data/NIST/sre18/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set
	
	sre16_eval_set=/mnt/matylda2/data/NIST/sre16/R149_0_1/
	sre16_dev_set=/mnt/matylda2/data/NIST/sre16/LDC2016E46_SRE16_Call_My_Net_Training_Data/
	
	janus_set_dir=/mnt/matylda2/data/LDC/LDC2019E55_Janus_Multimedia_Dataset
    fi

    #mkdir lists
    #cut  -f1 -d" " data/vox1_gsmfr/spk2utt | shuf | head -n100 > lists/vox1_100spk
    #tools/copy_data_dir.sh $data/vox1_gsmfr $data/vox1_gsmfr_100spk --spk_list lists/vox1_100spk

    if [ "A" == "B" ];then
	tools/combine_data_dir.sh $data/train $data/vox1_gsmfr $data/vox2_dev_gsmfr $data/cts 
    fi

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ "A" == "B" ];then 
	echo "Covert train and test data to ${data_type}..."
	for dset in vox1_gsmfr_100spk; do
	    if [ $data_type == "shard" ]; then
		python tools/make_shard_list.py --num_utts_per_shard 1000 \
		       --num_threads 10 \
		       --prefix shards \
		       --shuffle \
		       ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
		       ${data}/$dset/shards ${data}/$dset/shard.list
	    else
		python tools/make_raw_list.py ${data}/$dset/wav.scp \
		       ${data}/$dset/utt2spk ${data}/$dset/raw.list
	    fi
	done
    fi
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

#if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#  echo "Covert train and test data to ${data_type}..."
#  for dset in vox2_dev vox1; do
#    if [ $data_type == "shard" ]; then
#      python tools/make_shard_list.py --num_utts_per_shard 1000 \
#          --num_threads 16 \
#          --prefix shards \
#          --shuffle \
#          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
#          ${data}/$dset/shards ${data}/$dset/shard.list
#    else
#      python tools/make_raw_list.py ${data}/$dset/wav.scp \
#          ${data}/$dset/utt2spk ${data}/$dset/raw.list
#    fi
#  done
#  # Convert all musan data to LMDB
#  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
#  # Convert all rirs data to LMDB
#  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
#fi

#python wespeaker/bin/prep_archives.py --config $config \
#    --exp_dir ${exp_dir} \
#    --data_type "${data_type}" \
#    --train_data ${data}/vox1_gsmfr_100spk/${data_type}.list \
#    --train_label ${data}/vox1_gsmfr_100spk/utt2spk \
#    --reverb_data ${data}/rirs/lmdb \
#    --noise_data ${data}/musan/lmdb \
#    ${checkpoint:+--checkpoint $checkpoint}


if [ "A" == "B" ];then
    python tools/make_raw_list.py ${data}/train/wav.scp \
	${data}/train/utt2spk ${data}/train/raw.list
fi


if [ "A" == "A" ];then
python wespeaker/bin/prep_archives.py --config $config \
    --exp_dir ${exp_dir} \
    --data_type raw \
    --train_data ${data}/train/raw.list \
    --train_label ${data}/train/utt2spk \
    --reverb_data ${data}/rirs/lmdb \
    --noise_data ${data}/musan/lmdb \
    --num_epochs 37 
fi

if [ "A" == "B" ];then
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --whole_utt "True" \
      --train_data ${data}/train/shard.list \
      --train_label ${data}/train/utt2spk \
      ${checkpoint:+--checkpoint $checkpoint}
      #--reverb_data ${data}/rirs/lmdb \ # Augmentations are skipped if these files not provided
      #--noise_data ${data}/musan/lmdb \
fi
fi
exit 
#if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#  echo "Start training ..."
#  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
#  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
#    wespeaker/bin/train.py --config $config \
#      --exp_dir ${exp_dir} \
#      --gpus $gpus \
#      --num_avg ${num_avg} \
#      --data_type "${data_type}" \
#      --train_data ${data}/vox1_gsmfr_100spk/${data_type}.list \
#      --train_label ${data}/vox1_gsmfr_100spk/utt2spk \
#      --reverb_data ${data}/rirs/lmdb \
#      --noise_data ${data}/musan/lmdb \
#      ${checkpoint:+--checkpoint $checkpoint}
#fi


#if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#  echo "Start training ..."
#  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
#  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
#    wespeaker/bin/train.py --config $config \
#      --exp_dir ${exp_dir} \
#      --gpus $gpus \
#      --num_avg ${num_avg} \
#      --data_type "${data_type}" \
#      --train_data ${data}/vox2_dev/${data_type}.list \
#      --train_label ${data}/vox2_dev/utt2spk \
#      --reverb_data ${data}/rirs/lmdb \
#      --noise_data ${data}/musan/lmdb \
#      ${checkpoint:+--checkpoint $checkpoint}
#fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model
  if [[ $config == *repvgg*.yaml ]]; then
    echo "convert repvgg model ..."
    python wespeaker/models/convert_repvgg.py \
      --config $exp_dir/config.yaml \
      --load $avg_model \
      --save $exp_dir/models/convert_model.pt
    model_path=$exp_dir/models/convert_model.pt
  fi

  echo "Extract embeddings ..."
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Large margin fine-tuning ..."
  lm_exp_dir=${exp_dir}-LM
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run.sh --stage 3 --stop_stage 7 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
