import yaml
from pathlib import Path
from data_io.worldbank import fetch_wdi

if __name__ == "__main__":
    config_path = Path("configs/countries.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    countries = config["countries"]

    indicators = {
        "FP.CPI.TOTL.ZG": "inflation",
        "NE.TRD.GNFS.ZS": "openness",
        "NE.GDI.TOTL.ZS": "investment",
        "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    }

    df = fetch_wdi(countries, indicators, start_year=2000, end_year=2023)

    print(df.head())
    print("\nShape:", df.shape)
    print("\nMissing values per column:")
    print(df.isna().sum())

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/wdi_africa_2000_2023.csv", index=False)
    print("\n✅ Saved to data/processed/wdi_africa_2000_2023.csv")
