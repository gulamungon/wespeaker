#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#           2014  David Snyder
#           2023  Brno Univeristy of Techology (Author: Johan Rohdin)
# Apache 2.0

# Combines utt2spk, wav.scp and utt2spk. Very reduced compared to the Kaldi script by Daniel Povey and David Snyder.



set -o pipefail

echo "$0 $@"  # Print the command line for logging

export LC_ALL=C

dest=$1;
shift;

mkdir $dest


# wav.scp must exist in all directpries. If not, abort.
for file in wav.scp; do
    ( for f in $*; do cat $f/$file; done ) | sort -k1 > $dest/$file || exit 1;
done


# Merge utt2spk if it exists in all directories
has_utt2spk=true
for in_dir in $*; do
  if [ ! -f $in_dir/utt2spk ]; then
    has_utt2spk=false
    break
  fi
done

if $has_utt2spk; then
    ( for f in $*; do cat $f/utt2spk; done ) | sort -k1 > $dest/utt2spk || exit 1;
else
    echo "WARNING: utt2spk does not exist in all directories. It will not be produced in the destination directory."
fi



# Merge spk2utt if it exists in all directories
has_spk2utt=true
for in_dir in $*; do
  if [ ! -f $in_dir/spk2utt ]; then
    has_spk2utt=false
    break
  fi
done

# Note that a concatenation of sevearl spk2utt could be invalided because if a speaker
# is present in more than one of the spk2utt to concatenate, there will be more than one 
# line with this speaker in the resulting spk2utt. For this reason, spk2utt_to_utt2spk.pl 
# is applied insided the loop utt2spk are obtained before concatenation. However in this 
# particular case, this doesn't  matter since the subsequent utt2spk_to_spk2utt.pl 
# processes line by line.
#  
if $has_spk2utt; then
    ( for f in $*; do cat $f/spk2utt | tools/spk2utt_to_utt2spk.pl; done)  | sort -k1 | tools/utt2spk_to_spk2utt.pl > $dest/spk2utt  || exit 1;
else
    echo "WARNING: spk2utt does not exist in all directories. It will not be produced in the destination directory."
fi




