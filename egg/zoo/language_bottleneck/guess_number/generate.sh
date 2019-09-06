#for path in 2019_08_27_04_12_35/ 2019_09_01_07_20_33/ 2019_09_03_05_06_19; do
#for path in 2019_09_04_03_06_28; do
for path in 2019_09_04_10_37_05; do
  python dump_languages.py --input_path=/private/home/kharitonov/nest/guess_number/$path --output_path=./factorized_discr
done
