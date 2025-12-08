import pandas as pd

df = pd.read_csv('data/processed/features_ready.csv')

print("=" * 70)
print("📊 DATASET FINAL POUR MACHINE LEARNING")
print("=" * 70)
print(f"\n✅ Total observations: {len(df)}")
print(f"✅ Pays uniques: {df['country'].nunique()}")
print(f"✅ Période: {df['year'].min()}-{df['year'].max()}")
print(f"\n📋 Colonnes disponibles:")
print(df.columns.tolist())

print("\n🎯 Features pour les modèles ML:")
features = ['cdi_smooth_lag1', 'inflation_lag1', 'trade_openness_lag1', 'investment_lag1']
print(df[features].describe())

print("\n🎯 Target variable (log_gdp_volatility):")
print(df['log_gdp_volatility'].describe())

print(f"\n❌ Missing values dans les features:")
print(df[features + ['log_gdp_volatility']].isnull().sum())