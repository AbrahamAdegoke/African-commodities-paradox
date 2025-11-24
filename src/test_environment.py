import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

print("✔️ pandas version:", pd.__version__)
print("✔️ numpy version:", np.__version__)

# Test simple d'un modèle ML
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = Ridge()
model.fit(X, y)

print("✔️ Ridge prediction for 6:", model.predict([[6]])[0])

# Test simple de plot (ne s'affiche pas, mais ne doit pas planter)
plt.plot([1,2,3], [1,4,9])
plt.close()

print("🎉 Tout fonctionne : environnement Python OK")
