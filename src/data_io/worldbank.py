"""
World Bank (WDI) data loader.

This module downloads macroeconomic indicators (inflation, openness, investment, GDP growth)
for a list of countries and years using the World Bank API.
"""

from typing import Dict, List
import pandas as pd
import requests
import time
from requests.exceptions import ReadTimeout, ConnectionError

def safe_request(url, retries=5, timeout=120):
    """
    Perform a GET request with automatic retries and long timeout.

    Parameters
    ----------
    url : str
    retries : int
    timeout : int

    Returns
    -------
    Response or None if failed.
    """
    for attempt in range(1, retries + 1):
        try:
            return requests.get(url, timeout=timeout)
        except (ReadTimeout, ConnectionError):
            print(f"[WARN] Timeout for {url} (attempt {attempt}/5). Retrying in 2 sec...")
            time.sleep(2)

    print(f"[ERROR] Request failed after {retries} attempts → {url}")
    return None


BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=20000"


def fetch_wdi(
    countries: List[str],
    indicators: Dict[str, str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Download selected WDI indicators for a list of countries.

    Parameters
    ----------
    countries : list of ISO3 country codes (e.g. ["NGA", "GHA"])
    indicators : dict mapping WDI code -> column name
        e.g. {
            "FP.CPI.TOTL.ZG": "inflation",
            "NE.TRD.GNFS.ZS": "openness",
            "NE.GDI.TOTL.ZS": "investment",
            "NY.GDP.MKTP.KD.ZG": "gdp_growth"
        }
    start_year : int
    end_year : int

    Returns
    -------
    DataFrame with columns: country, year, and one column per indicator.
    """
    rows = []

    for iso in countries:
        iso_lower = iso.lower()
        for code, name in indicators.items():
            url = BASE_URL.format(country=iso_lower, indicator=code)
            resp = safe_request(url, timeout=120)
            resp = safe_request(url, timeout=120)
            if resp is None:
                continue


            if resp.status_code != 200:
                print(f"[WARN] Failed for {iso} - {code} (status {resp.status_code})")
                continue

            data = resp.json()
            # World Bank API returns [metadata, data]
            if not isinstance(data, list) or len(data) < 2 or data[1] is None:
                continue

            for rec in data[1]:
                year = rec.get("date")
                value = rec.get("value")

                if year is None or value is None:
                    continue

                y = int(year)
                if start_year <= y <= end_year:
                    rows.append(
                        {
                            "country": iso,
                            "year": y,
                            name: float(value),
                        }
                    )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Group by (country, year) and keep first non-null per indicator
    df = df.groupby(["country", "year"]).first().reset_index()
    return df
