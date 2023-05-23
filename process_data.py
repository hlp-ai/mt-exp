import pathlib

from sklearn.model_selection import train_test_split
from tensorflow import keras

text_file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"

with open(text_file, encoding="utf-8") as f:
    lines = f.read().split("\n")[:-1]

bitext = []
for line in lines:
    eng, spa = line.split("\t")
    bitext.append((eng.strip(), spa.strip()))

print("# of bitexts:", len(bitext))

test_size = 1500

train_bitext, test_bitext = train_test_split(bitext, test_size=test_size, random_state=31)

print("# of train data:", len(train_bitext))
print("# of test data:", len(test_bitext))


with open("train-eng-spa.tsv", "w", encoding="utf-8") as f:
    for p in train_bitext:
        f.write(p[0] + "\t" + p[1] + "\n")

with open("test-eng-spa.tsv", "w", encoding="utf-8") as f:
    for p in test_bitext:
        f.write(p[0] + "\t" + p[1] + "\n")
