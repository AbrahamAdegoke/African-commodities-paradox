from setuptools import setup, find_packages

setup(
    name="african-commodities-paradox",
    version="1.0.0",
    author="Abraham Adegoke",
    description="Data-driven analysis of commodity dependence and economic volatility in Africa",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'requests>=2.31.0',
        'pyyaml>=6.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'joblib>=1.3.0',
    ],
)