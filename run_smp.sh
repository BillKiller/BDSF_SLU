for x in {1..10};
do
python train.py -wed 128 -ehd 256 -aod 128  --index $x -dd data/smp -rs $x -sd save_bmeso/smp_$x  -tr  --bio_schame bmeso
done
