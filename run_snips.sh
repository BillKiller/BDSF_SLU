dataset=snips
for x in {1..10};
do
python train.py -wed 32 -ehd 256 -aod 128  --index $x -dd data/$dataset -rs $x -sd save_bmeso/$dataset-$x  -tr  --bio_schame bmeso
done
w