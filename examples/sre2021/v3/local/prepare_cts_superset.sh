#!/bin/bash

set -o pipefail

export LC_ALL=C


data_cts=data/cts/
cts_superset_dir=""
wav_dir=wav/cts/

. tools/parse_options.sh || exit 1

echo $cts_superset_dir

if [ ! -f $cts_superset_dir/docs/cts_superset_segment_key.tsv ];then
    echo "ERROR: $cts_superset_dir/docs/cts_superset_segment_key.tsv does not exist."
    exit 1
fi

mkdir -p $data_cts
mkdir -p $wav_dir

# Make the directory for each subject
if [ "A" == "B" ];then
    for x in $(tail -n +2 $cts_superset_dir/docs/cts_superset_segment_key.tsv | cut -f 3 | uniq | sort -u );do
	mkdir -p $wav_dir/$x
    done
fi

# Convert sph to wav
if [ "A" == "B" ];then
    echo -n "" > ${data_cts}/make_wav.sh
    for x in $(tail -n +2 $cts_superset_dir/docs/cts_superset_segment_key.tsv | cut -f 1 | sed "s:\.sph::" );do
	echo "sph2pipe -f wav ${cts_superset_dir}/data/${x}.sph > ${wav_dir}/${x}.wav" >> ${data_cts}/make_wav.sh
    done
    bash ${data_cts}/make_wav.sh > logs/make_wav.sh.log 
fi

if [ "A" == "B" ];then
    tail -n +2 $cts_superset_dir/docs/cts_superset_segment_key.tsv | cut -f 1,3 --output-delimiter=" " | sed "s:\.sph:\.wav:" | sort > ${data_cts}/utt2spk
    tools/utt2spk_to_spk2utt.pl ${data_cts}/utt2spk > ${data_cts}/spk2utt
fi

echo -n "" > ${data_cts}/wav.scp
for x in $(tail -n +2 $cts_superset_dir/docs/cts_superset_segment_key.tsv | cut -f 1 | sed "s:\.sph::" );do
    echo "${x}.wav ${wav_dir}/${x}.wav" >> ${data_cts}/wav.scp
done
