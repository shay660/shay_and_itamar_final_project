import sys
import glob
sys.path.insert(0, '..')
from os import chdir

import joblib
from main import find_significant_kmers

def significant_kmers_from_model() -> None:
    model_path = glob.glob("*.joblib")[0]
    print(model_path)
    model = joblib.load(model_path)
    find_significant_kmers(model)

if __name__ == '__main__':
    dir_path = sys.argv[1]
    chdir(dir_path)
    significant_kmers_from_model()
