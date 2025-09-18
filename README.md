# IdeoMap · Geometría de la Ideología

Modelado y visualización **multidimensional** de ideologías políticas a partir de
un espacio de dimensiones explícitas (ontología) y dos rutas de reducción:
**PCA** (lineal) y **Autoencoders** (no lineal, 1D y 3D) con visualizaciones interactivas.

> Proyecto en español, pensado para ser *reproducible* de punta a punta.

---

## 🗂 Estructura del repositorio

```
data/
  ae1d_coords.csv
  ae3d_coords.csv
  dimensions.csv
  ideologies.csv
  texts.csv
figures/
  ae1d_enhanced.html
  ae3d_colored.html
  ae3d_colored.png
  projection.png
models/
  ae1d/ae3d_model.pt        # AE 1D (archivo del modelo)
  ae1d/ae3d_stats.json
  ae3d/ae3d_model.pt        # AE 3D (archivo del modelo)
  ae3d/ae3d_stats.json
  pca.joblib
  text_regressor.joblib
nn/
  nn_reducer.py
  train_ae1d.py
  train_ae3d.py
embedder.py
requirements.txt
run.ipynb
```

---

## 📦 Instalación rápida

```bash
git clone https://github.com/diegolavallade/IdeoMap
cd IdeoMap
python -m venv .venv && source .venv/bin/activate  # (en Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

---

## 📚 Datos de entrada (CSV)

* `data/dimensions.csv`: `dimension_id,name,neg_label,pos_label,description,scale_min,scale_max,default_weight`
* `data/ideologies.csv`: `ideology_id,name,<columnas por dimensión>`
* `data/texts.csv` *(opcional)*: `ideology_id,text`

> **Rango recomendado de valores:** cada puntuación numérica debe estar dentro de `[scale_min, scale_max]` (típicamente `[-1, 1]`).

---

## 🧭 Dos rutas de trabajo

### 1) Ruta clásica (PCA)

```python
from embedder import EmbeddingSpace

space = EmbeddingSpace(base_path=".")
space.load()
space.fit_projection(n_components=2)
coords = space.project()
space.plot_projection(save_path="figures/projection.png")
```

* Guarda el modelo PCA en `models/pca.joblib` y un scatter 2D en `figures/projection.png`.
* También puedes consultar vecindades con `space.nearest(vector, top_k=5)`.

### 2) Ruta no lineal (Autoencoders)

#### 2.1 AE **1D** con visualización semántica interactiva

```bash
python nn/train_ae1d.py --device cuda --epochs 2500 --patience 40 --open
# Cambia a --device cpu si no tienes GPU
```

* Exporta `data/ae1d_coords.csv`.
* Genera `figures/ae1d_enhanced.html` con un panel de detalles (composición por dimensión, “lectura general”, etc.).
* El modelo y el historial de entrenamiento quedan en `models/ae1d/`.

#### 2.2 AE **3D** con coloración procedimental

```bash
python nn/train_ae3d.py --device cuda --epochs 10000 --patience 20 --open
```

* Exporta `data/ae3d_coords.csv`.
* Genera `figures/ae3d_colored.html` (interactivo) y `figures/ae3d_colored.png` (estático).
* El modelo y el historial de entrenamiento quedan en `models/ae3d/`.

> **Tips**
>
> * Usa `--no-train` para reutilizar `data/ae1d_coords.csv` o `data/ae3d_coords.csv` si ya existen.
> * `--open` abre automáticamente el HTML resultante en tu navegador.

---

## 🧠 ¿Qué hace cada módulo?

* **`embedder.py`**
  Carga/valida CSV, entrena PCA, exporta proyecciones y vecindades; además, expone un `TextRegressor` (TF-IDF → Regressor) para mapear texto → vector ideológico y persistirlo en `models/text_regressor.joblib`.

* **`nn/nn_reducer.py`**
  Define el **Autoencoder MLP** (encoder/decoder con BatchNorm y Dropout), el ciclo de entrenamiento con **MSE**, **estandarización por dimensión**, partición **train/val** con semilla fija y **early stopping**. Persiste `ae3d_model.pt` y `ae3d_stats.json` con metadatos del entrenamiento.

* **`nn/train_ae1d.py`**
  Configura el **cuello de botella en 1 dimensión**, entrena si es necesario, guarda `ae1d_coords.csv` y construye un **HTML enriquecido** que muestra la composición ideológica por dimensión al hacer clic en cada punto.

* **`nn/train_ae3d.py`**
  Entrena el AE 3D, colorea los puntos **por posición (Z1→R, Z2→G, Z3→B)** para resaltar clústeres y genera **HTML interactivo + PNG**.

---

## 🔬 Reproducibilidad

* Estandarización por dimensión (media/sigma) antes de entrenar.
* Semilla fija para el *shuffle* del *split* (85%/15%).
* Early stopping por pérdida de validación y guardado del **mejor** modelo.
* CPU/GPU conmutables vía `--device` (usa CPU si no hay CUDA disponible).

---

## 📄 Paper / contexto

Pre-print por publicarse

---

## 🤝 Contribuir

1. Crea un branch desde `main`.
2. Sigue la convención de nombres para *scripts* y salidas.
3. Abre un PR con descripción clara y *before/after* de figuras si aplica.

---

## 📜 Licencia

```
MIT License

Copyright (c) 2025 diegolavallade

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ✉️ Contacto

Diego Lavallade — issues y PRs bienvenidos.
