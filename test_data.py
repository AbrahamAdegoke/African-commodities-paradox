import pandas as pd

# Charge les données
df = pd.read_csv('data/raw/worldbank_wdi.csv')

# Aperçu
print("Shape:", df.shape)
print("\nPremières lignes:")
print(df.head(10))

# Vérifie le CDI
print("\n📊 CDI Statistics:")
print(df['cdi_raw'].describe())

# Vérifie les pays
print(f"\nPays uniques: {df['country'].nunique()}")
print(df['country'].unique())

# Missing values
print("\n❌ Missing values:")
print(df.isnull().sum())

# Affiche quelques exemples de CDI élevé
print("\n🔥 Top 10 CDI (pays les plus dépendants):")
top_cdi = df.groupby('country')['cdi_raw'].mean().sort_values(ascending=False).head(10)
print(top_cdi)