import nltk
from nltk.corpus import wordnet as wn
import random
import pandas as pd
import tqdm

# Baixar o recurso do WordNet (caso ainda não tenha sido baixado)
#nltk.download('wordnet')

# Obter todas as synsets em WordNet
all_synsets = list(wn.all_synsets())

# Definir o número de iterações desejadas
num_iterations = len(all_synsets)

# Criar uma lista para armazenar os dataframes
dataframes = []

# Iniciar a iteração
for iteration in tqdm.tqdm(range(num_iterations), desc='Progress'):
    # Obter uma synset aleatória
    synset = all_synsets[iteration]

    # Obter todos os hiperônimos da synset
    hypernyms = synset.hypernyms()
    hypernyms_data = {
        'Definição_Synset': [synset.definition() for _ in range(len(hypernyms))],
        'ID_Synset': [synset.name() for _ in range(len(hypernyms))],
        'Definição_Relacionada': [h.definition() for h in hypernyms],
        'ID_Relacionada': [h.name() for h in hypernyms],
        'Relacao': ['Hypernyms' for _ in range(len(hypernyms))]
    }
    hypernyms_df = pd.DataFrame(hypernyms_data)
    dataframes.append(hypernyms_df)

    # Obter todos os hipônimos da synset
    hyponyms = synset.hyponyms()
    hyponyms_data = {
        'Definição_Synset': [synset.definition() for _ in range(len(hyponyms))],
        'ID_Synset': [synset.name() for _ in range(len(hyponyms))],
        'Definição_Relacionada': [h.definition() for h in hyponyms],
        'ID_Relacionada': [h.name() for h in hyponyms],
        'Relacao': ['Hyponyms' for _ in range(len(hyponyms))]
    }
    hyponyms_df = pd.DataFrame(hyponyms_data)
    dataframes.append(hyponyms_df)
    
    # Obter todos os holônimos da synset
    holonyms = synset.part_holonyms() + synset.substance_holonyms() + synset.member_holonyms()
    holonyms_data = {
        'Definição_Synset': [synset.definition() for _ in range(len(holonyms))],
        'ID_Synset': [synset.name() for _ in range(len(holonyms))],
        'Definição_Relacionada': [h.definition() for h in holonyms],
        'ID_Relacionada': [h.name() for h in holonyms],
        'Relacao': ['Holonyms' for _ in range(len(holonyms))]
    }
    holonyms_df = pd.DataFrame(holonyms_data)
    dataframes.append(holonyms_df)

# Concatenar todos os dataframes em um único dataframe
combined_df = pd.concat(dataframes)

# Salvar o dataframe em um arquivo CSV
combined_df.to_csv('all_synsets.csv', index=False)
