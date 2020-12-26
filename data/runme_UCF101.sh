#!/usr/bin/env bash

# 1. TODO: download ucf-101 and unpack
#wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
#unzip it

#1.5 Make sure you have the env activated! Run this to split the data up
split_folders "$(pwd)/UCF-101" --output "$(pwd)/fusion_data/UCF-101" --ratio .7 .2 .1

# 2. Training data
# shellcheck disable=SC2164
cd "$(pwd)/fusion_data/UCF-101/train"
extension=".avi"

for i in *
do
  for j in "$i"/*"$extension"
  do
    mkdir -p "$i"/audio
    mkdir -p "$i"/video
    mkdir -p "$i"/frames

    # shellcheck disable=SC2001
#    outfile=$(echo "$j" | sed -e "s/$i\/\(.*\)$extension/\1/")
    outfile=$(basename "$j" $extension)
    mkdir -p "$i"/frames/"$outfile"

    # extract audio, downmix to mono and resample to 16k
    ffmpeg -y -i "$j" -vn -acodec pcm_s16le -ac 1 -ar 16000 "$i"/audio/"$outfile".wav -hide_banner -loglevel panic

    #convert video to be 16fps # TODO: 'scale=320:240,' ?
    ffmpeg -y -i "$j" -vf "scale=-1:240,fps=16" "$i"/frames/"$outfile"/%04d.jpeg -loglevel panic

  done
  echo "finished $i"
done

# 3. Test data
# shellcheck disable=SC2164
cd "../test"
extension=".avi"

for i in *
do
  for j in "$i"/*"$extension"
  do
    mkdir -p "$i"/audio
    mkdir -p "$i"/video
    mkdir -p "$i"/frames

    # shellcheck disable=SC2001
#    outfile=$(echo "$j" | sed -e "s/$i\/\(.*\)$extension/\1/")
    outfile=$(basename "$j" $extension)
    mkdir -p "$i"/frames/"$outfile"

    # extract audio, downmix to mono and resample to 16k
    ffmpeg -y -i "$j" -vn -acodec pcm_s16le -ac 1 -ar 16000 "$i"/audio/"$outfile".wav -hide_banner -loglevel panic

    #convert video to be 16fps # TODO: 'scale=320:240,' ?
    ffmpeg -y -i "$j" -vf "scale=-1:240,fps=16" "$i"/frames/"$outfile"/%04d.jpeg -loglevel panic

  done
  echo "finished $i"
done

# 4. Validation data
# shellcheck disable=SC2164
cd "../val"
extension=".avi"

for i in *
do
  for j in "$i"/*"$extension"
  do
    mkdir -p "$i"/audio
    mkdir -p "$i"/video
    mkdir -p "$i"/frames

    # shellcheck disable=SC2001
#    outfile=$(echo "$j" | sed -e "s/$i\/\(.*\)$extension/\1/")
    outfile=$(basename "$j" $extension)
    mkdir -p "$i"/frames/"$outfile"

    # extract audio, downmix to mono and resample to 16k
    ffmpeg -y -i "$j" -vn -acodec pcm_s16le -ac 1 -ar 16000 "$i"/audio/"$outfile".wav -hide_banner -loglevel panic

    #convert video to be 16fps # TODO: 'scale=320:240,' ?
    ffmpeg -y -i "$j" -vf "scale=-1:240,fps=16" "$i"/frames/"$outfile"/%04d.jpeg -loglevel panic

  done
  echo "finished $i"
done
