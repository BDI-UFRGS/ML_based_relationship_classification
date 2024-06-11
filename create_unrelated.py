
# Imports
from scipy.sparse import load_npz
import nltk
from nltk.corpus import wordnet as wn
import random
import pandas as pd
from tqdm import tqdm

# Donwload wordnet
nltk.download('wordnet')

# Load the saved matrices
hypernym_matrix = load_npz("hypernym_matrix.npz")
holonym_matrix = load_npz("holonym_matrix.npz")

# Convert the sparce matrices into full matrices
hypernym_matrix_full = hypernym_matrix.toarray()
holonym_matrix_full = holonym_matrix.toarray()
# Create an array of synsets
synsets = list(wn.all_synsets())
size = len(synsets)

# Create an empty DataFrame
df = pd.DataFrame(columns=['Definição_Synset', 'ID_Synset', 'Definição_Relacionada', 'ID_Relacionada', 'Relacao'])

# Choose the number of unrelated instances 
total_elements = 10000

# Initialize tqdm to track progress
with tqdm(total=total_elements, desc="Processing", unit="element") as pbar:
    while(len(df)<total_elements):
      i = random.randint(0, size-1)
      j = random.randint(0, size-1)
      if holonym_matrix_full[i,j] == 0 and hypernym_matrix_full[i,j]==0:
        df.loc[len(df)] = [synsets[i].definition(), synsets[i].name(), synsets[j].definition(), synsets[j].name(), 'Unrelated']
        pbar.update(1)  # Update progress bar
