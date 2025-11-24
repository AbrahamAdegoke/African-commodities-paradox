import yaml
from pathlib import Path
from data_io.worldbank import fetch_wdi

if __name__ == "__main__":
    # Charger la liste des pays africains depuis configs/countries.yaml
    config_path = Path("configs/countries.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    countries = config["countries"]

    indicators = {
        "FP.CPI.TOTL.ZG": "inflation",        # Inflation (%)
        "NE.TRD.GNFS.ZS": "openness",         # Trade (% of GDP)
        "NE.GDI.TOTL.ZS": "investment",       # Gross capital formation
        "NY.GDP.MKTP.KD.ZG": "gdp_growth",    # Real GDP growth
    }

    df = fetch_wdi(countries, indicators, start_year=2000, end_year=2023)

    print(df.head())
    print("\nShape:", df.shape)
    print("\nMissing values per column:")
    print(df.isna().sum())

    # Sauvegarder une version brute pour vérifier plus tard
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/wdi_africa_2000_2023.csv", index=False)
    print("\n✅ Saved to data/processed/wdi_africa_2000_2023.csv")
