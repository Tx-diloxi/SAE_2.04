import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Import et préparation des données
chemin_fichier = "vue_voyelles_m2105.csv"

# Import des données
vue_df = pd.read_csv(chemin_fichier, sep=';')

# %% Fonction pour extraire les voyelles
def compter_voyelles(prenom):
    """Compte les voyelles dans un prénom"""
    voyelles = 'aeiouAEIOU'
    return ''.join(c for c in prenom if c in voyelles).lower()

# %% Analyse des voyelles dans les prénoms
# Créer une colonne avec les voyelles de chaque prénom
vue_df['voyelles'] = vue_df['prenom'].apply(compter_voyelles)

# Compter le nombre de prénoms pour chaque combinaison de voyelles
count_voyelles = vue_df['voyelles'].value_counts()
print("\nDistribution des combinaisons de voyelles:")
print(count_voyelles)

# %% Calcul des moyennes par combinaison de voyelles
moyennes_voyelles = vue_df.groupby('voyelles')['m2105'].mean()
print("\nMoyennes par combinaison de voyelles:")
print(moyennes_voyelles)

# %% Visualisation
plt.figure(figsize=(15, 6))

# Distribution des voyelles
plt.subplot(121)
count_voyelles.head(10).plot(kind='bar')
plt.title('Top 10 des combinaisons de voyelles')
plt.xlabel('Combinaisons de voyelles')
plt.ylabel('Nombre de prénoms')
plt.xticks(rotation=45)

# Moyennes par voyelles
plt.subplot(122)
moyennes_voyelles.head(10).plot(kind='bar')
plt.title('Moyennes m2105 par combinaison de voyelles')
plt.xlabel('Combinaisons de voyelles')
plt.ylabel('Moyenne')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %% Analyse statistique
# Créer un DataFrame pour l'analyse
analyse_df = pd.DataFrame({
    'count': count_voyelles,
    'moyenne': moyennes_voyelles
})

# Calculer la corrélation
correlation = np.corrcoef(analyse_df['count'], analyse_df['moyenne'])[0,1]
print(f"\nCorrélation entre le nombre de prénoms et la moyenne: {correlation:.3f}")

# %% Visualisation de la corrélation
plt.figure(figsize=(10, 6))
plt.scatter(analyse_df['count'], analyse_df['moyenne'])
plt.xlabel('Nombre de prénoms')
plt.ylabel('Moyenne m2105')
plt.title('Corrélation entre fréquence des voyelles et moyenne')

# Ajouter la ligne de régression
z = np.polyfit(analyse_df['count'], analyse_df['moyenne'], 1)
p = np.poly1d(z)
plt.plot(analyse_df['count'], p(analyse_df['count']), "r--", alpha=0.8)

plt.show()