import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Import et préparation des données
chemin_fichier = "vue_voyelles_m2105.csv"
matieres = ['m2105', 'm1101', 'm1102', 'm1201', 'm2101']

# Import des données
vue_df = pd.read_csv(chemin_fichier, sep=';')

# %% Fonction pour compter le nombre de voyelles
def compter_voyelles(prenom):
    """Compte le nombre de voyelles dans un prénom"""
    voyelles = 'aeiouAEIOU'
    return sum(1 for c in prenom if c in voyelles)

# Ajouter la colonne du nombre de voyelles
vue_df['nb_voyelles'] = vue_df['prenom'].apply(compter_voyelles)

# %% Analyse pour chaque matière
def analyser_matiere(df, matiere):
    """Analyse la relation entre nombre de voyelles et notes pour une matière"""
    # Calculer moyennes par nombre de voyelles
    moyennes = df.groupby('nb_voyelles')[matiere].mean()
    counts = df.groupby('nb_voyelles')[matiere].count()
    
    return moyennes, counts

# %% Visualisation pour toutes les matières
plt.figure(figsize=(20, 15))

for idx, matiere in enumerate(matieres):
    moyennes, counts = analyser_matiere(vue_df, matiere)
    
    # Créer un DataFrame pour l'analyse
    analyse = pd.DataFrame({
        'moyenne': moyennes,
        'count': counts
    }).reset_index()
    
    # Calculer la corrélation
    correlation = np.corrcoef(analyse['nb_voyelles'], analyse['moyenne'])[0,1]
    
    # Créer le subplot
    plt.subplot(3, 2, idx+1)
    
    # Scatter plot avec taille des points proportionnelle au nombre d'étudiants
    plt.scatter(analyse['nb_voyelles'], analyse['moyenne'], 
               s=analyse['count']*50,  # Taille des points
               alpha=0.6)
    
    # Ligne de régression
    z = np.polyfit(analyse['nb_voyelles'], analyse['moyenne'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(analyse['nb_voyelles'].min(), analyse['nb_voyelles'].max(), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8)
    
    plt.xlabel('Nombre de voyelles dans le prénom')
    plt.ylabel('Moyenne')
    plt.title(f'{matiere}\nCorrélation: {correlation:.3f}')
    
    # Ajouter le nombre d'étudiants à côté de chaque point
    for _, row in analyse.iterrows():
        plt.annotate(f'n={int(row["count"])}', 
                    (row['nb_voyelles'], row['moyenne']),
                    xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

# %% Tableau récapitulatif
resultats = []
for matiere in matieres:
    moyennes, counts = analyser_matiere(vue_df, matiere)
    analyse = pd.DataFrame({
        'moyenne': moyennes,
        'count': counts
    }).reset_index()
    
    correlation = np.corrcoef(analyse['nb_voyelles'], analyse['moyenne'])[0,1]
    
    # Trouver les meilleurs et pires nombres de voyelles
    meilleur = analyse.nlargest(1, 'moyenne')
    pire = analyse.nsmallest(1, 'moyenne')
    
    resultats.append({
        'matiere': matiere,
        'correlation': correlation,
        'meilleur_nb_voyelles': meilleur['nb_voyelles'].iloc[0],
        'meilleure_moyenne': meilleur['moyenne'].iloc[0],
        'pire_nb_voyelles': pire['nb_voyelles'].iloc[0],
        'pire_moyenne': pire['moyenne'].iloc[0]
    })

# Afficher le tableau récapitulatif
resultats_df = pd.DataFrame(resultats)
print("\nTableau récapitulatif:")
print(resultats_df.to_string(index=False))

# %% Statistiques descriptives par nombre de voyelles
print("\nStatistiques par nombre de voyelles:")
stats_voyelles = vue_df.groupby('nb_voyelles').agg({
    'prenom': 'count',  # Nombre d'étudiants
    **{matiere: ['mean', 'std'] for matiere in matieres}  # Moyenne et écart-type pour chaque matière
}).round(2)

print(stats_voyelles)

# %% Sauvegarder les résultats
resultats_df.to_csv('resultats_nb_voyelles.csv', index=False)
stats_voyelles.to_csv('stats_nb_voyelles.csv')