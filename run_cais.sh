dataset=cais
for x in {1..10};
do
python train.py -wed 128 -ehd 512 -aod 128  --index $x -dd data/$dataset  -rs $x -sd save_bmeso/$dataset-$x  -tr  --bio_schame bmeso  --dropout_rate 0.3
done
