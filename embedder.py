
"""
Political Ideology Embedding: dimensions, vectors, training and inference.

Files expected (CSV):
- data/dimensions.csv: dimension_id,name,neg_label,pos_label,description,scale_min,scale_max,default_weight
- data/ideologies.csv: ideology_id,name,<dimension_id columns...>
- data/texts.csv (optional): ideology_id,text

Outputs:
- models/pca.joblib           -> PCA projection
- models/text_regressor.joblib-> TF-IDF + Ridge MultiOutputRegressor (optional)
- figures/projection.png      -> 2D scatter

Usage (inside Python):
    from embedder import EmbeddingSpace, TextRegressor
    space = EmbeddingSpace(base_path="/mnt/data/political_embedding")
    space.load()
    space.fit_projection(n_components=2)
    coords = space.project()
    space.plot_projection(save_path="/mnt/data/political_embedding/figures/projection.png")

    # Optional: train text->vector
    tr = TextRegressor(base_path="/mnt/data/political_embedding")
    tr.load(space)
    tr.train()
    v = tr.predict("texto político aquí")
    neighbors = space.nearest(v, top_k=3)

All numeric scores must be in range [scale_min, scale_max], typically [-1, 1].
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# scikit-learn pieces (optional)
try:
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import LinearRegression
    import joblib
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


@dataclass
class SpaceConfig:
    base_path: Path
    dims_file: Path
    id_file: Path
    texts_file: Path
    models_dir: Path
    figures_dir: Path


class EmbeddingSpace:
    def __init__(self, base_path: str):
        self.cfg = SpaceConfig(
            base_path=Path(base_path),
            dims_file=Path(base_path) / "data" / "dimensions.csv",
            id_file=Path(base_path) / "data" / "ideologies.csv",
            texts_file=Path(base_path) / "data" / "texts.csv",
            models_dir=Path(base_path) / "models",
            figures_dir=Path(base_path) / "figures",
        )
        self.dimensions: pd.DataFrame = pd.DataFrame()
        self.ideologies: pd.DataFrame = pd.DataFrame()
        self.X: pd.DataFrame = pd.DataFrame()  # numeric matrix aligned to dimensions
        self.pca = None
        self._dim_ids: List[str] = []

    # ----------------- I/O -----------------
    def load(self):
        self.dimensions = pd.read_csv(self.cfg.dims_file)
        self._dim_ids = self.dimensions["dimension_id"].tolist()
        df = pd.read_csv(self.cfg.id_file)

        # Ensure all dimension columns exist; fill missing with 0
        for d in self._dim_ids:
            if d not in df.columns:
                df[d] = 0.0
        # Keep only known columns
        keep_cols = ["ideology_id", "name"] + self._dim_ids
        self.ideologies = df[keep_cols].copy()

        # Numeric matrix
        self.X = self.ideologies[self._dim_ids].astype(float)

        # Clip to allowed ranges per dimension
        for i, d in enumerate(self._dim_ids):
            lo = float(self.dimensions.loc[self.dimensions.dimension_id == d, "scale_min"].iloc[0])
            hi = float(self.dimensions.loc[self.dimensions.dimension_id == d, "scale_max"].iloc[0])
            self.X[d] = self.X[d].clip(lo, hi)

    def save_ideologies(self):
        self.ideologies.to_csv(self.cfg.id_file, index=False)

    def add_dimension(self, dimension_id: str, name: str, neg_label: str, pos_label: str,
                      description: str = "", scale_min: float = -1.0, scale_max: float = 1.0,
                      default_weight: float = 1.0):
        # Add to dimensions table
        new_row = {
            "dimension_id": dimension_id, "name": name, "neg_label": neg_label, "pos_label": pos_label,
            "description": description, "scale_min": scale_min, "scale_max": scale_max,
            "default_weight": default_weight
        }
        self.dimensions = pd.concat([self.dimensions, pd.DataFrame([new_row])], ignore_index=True)
        # Add column to ideologies with default 0
        self.ideologies[dimension_id] = 0.0
        self._dim_ids.append(dimension_id)
        self.X = self.ideologies[self._dim_ids].astype(float)
        self.dimensions.to_csv(self.cfg.dims_file, index=False)
        self.save_ideologies()

    def add_ideology(self, ideology_id: str, name: str, scores: Dict[str, float]):
        # Ensure all dims exist
        for d in self._dim_ids:
            scores.setdefault(d, 0.0)
        row = {"ideology_id": ideology_id, "name": name}
        row.update({d: float(scores.get(d, 0.0)) for d in self._dim_ids})
        self.ideologies = pd.concat([self.ideologies, pd.DataFrame([row])], ignore_index=True)
        self.X = self.ideologies[self._dim_ids].astype(float)
        self.save_ideologies()

    # ----------------- Geometry -----------------
    def fit_projection(self, n_components: int = 2, random_state: int = 42):
        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn no disponible: no puedo ajustar PCA.")
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.pca.fit(self.X.values)
        # persist
        model_path = self.cfg.models_dir / "pca.joblib"
        try:
            import joblib
            joblib.dump(self.pca, model_path)
        except Exception:
            pass
        return self.pca

    def load_projection(self):
        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn no disponible: no puedo cargar PCA.")
        model_path = self.cfg.models_dir / "pca.joblib"
        if model_path.exists():
            import joblib
            self.pca = joblib.load(model_path)
        else:
            raise FileNotFoundError("No hay modelo PCA guardado. Ejecuta fit_projection primero.")

    def project(self) -> pd.DataFrame:
        if self.pca is None:
            raise RuntimeError("PCA no entrenado. Llama fit_projection() o load_projection().")
        coords = self.pca.transform(self.X.values)
        cols = [f"PC{i+1}" for i in range(coords.shape[1])]
        df = pd.DataFrame(coords, columns=cols)
        df.insert(0, "name", self.ideologies["name"].values)
        df.insert(0, "ideology_id", self.ideologies["ideology_id"].values)
        return df

    def nearest(self, vector: np.ndarray, top_k: int = 5) -> pd.DataFrame:
        """Return nearest ideologies by cosine distance to a provided vector of size = #dims."""
        V = np.atleast_2d(vector.astype(float))
        M = self.X.values
        D = cosine_distances(V, M)[0]  # shape: (n,)
        idx = np.argsort(D)[:top_k]
        return pd.DataFrame({
            "ideology_id": self.ideologies.iloc[idx]["ideology_id"].values,
            "name": self.ideologies.iloc[idx]["name"].values,
            "distance": D[idx]
        })

    def export_long_format(self) -> pd.DataFrame:
        """Return long-format table: ideology_id, name, dimension_id, score"""
        long_df = self.ideologies.melt(id_vars=["ideology_id", "name"],
                                       value_vars=self._dim_ids,
                                       var_name="dimension_id",
                                       value_name="score")
        return long_df

    # ----------------- Plotting -----------------
    def plot_projection(self, save_path: Optional[str] = None):
        """Creates a simple 2D scatter plot of the PCA projection (PC1 vs PC2)."""
        import matplotlib.pyplot as plt

        proj = self.project()
        x, y = proj["PC1"].values, proj["PC2"].values
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y)
        for i, label in enumerate(proj["name"].values):
            ax.annotate(label, (x[i], y[i]), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Proyección PCA del espacio ideológico")
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return proj


class TextRegressor:
    """TF-IDF -> MultiOutput Ridge to map text -> ideology vector over current dimensions."""
    def __init__(self, base_path: str):
        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn no disponible: no puedo entrenar el modelo de texto.")
        self.cfg = SpaceConfig(
            base_path=Path(base_path),
            dims_file=Path(base_path) / "data" / "dimensions.csv",
            id_file=Path(base_path) / "data" / "ideologies.csv",
            texts_file=Path(base_path) / "data" / "texts.csv",
            models_dir=Path(base_path) / "models",
            figures_dir=Path(base_path) / "figures",
        )
        self.space: Optional[EmbeddingSpace] = None
        self.pipeline: Optional[Pipeline] = None
        self.dim_ids: List[str] = []

    def load(self, space: Optional[EmbeddingSpace] = None):
        if space is None:
            space = EmbeddingSpace(str(self.cfg.base_path))
            space.load()
        self.space = space
        self.dim_ids = self.space.dimensions["dimension_id"].tolist()

    def train(self, alpha: float = 1.0, max_features: int = 20000, ngram_range=(1,2)):
        assert self.space is not None, "Llama load() primero."
        texts = pd.read_csv(self.cfg.texts_file)
        # Join targets by ideology_id
        Y = texts.merge(self.space.ideologies[["ideology_id"] + self.dim_ids], on="ideology_id", how="left")
        if Y[self.dim_ids].isna().any().any():
            raise ValueError("Hay textos con ideology_id no presente en ideologies.csv")
        X_text = texts["text"].astype(str).values
        Y_vec = Y[self.dim_ids].values.astype(float)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)),
            ("reg", MultiOutputRegressor(LinearRegression()))
        ])
        self.pipeline.fit(X_text, Y_vec)

        # Persist
        try:
            import joblib
            joblib.dump(self.pipeline, self.cfg.models_dir / "text_regressor.joblib")
        except Exception:
            pass

    def load_trained(self):
        import joblib
        path = self.cfg.models_dir / "text_regressor.joblib"
        if not path.exists():
            raise FileNotFoundError("No existe text_regressor.joblib. Entrena con train().")
        self.pipeline = joblib.load(path)

    def predict(self, text: str) -> np.ndarray:
        assert self.pipeline is not None, "Modelo no cargado. Usa load_trained() o train()."
        vec = self.pipeline.predict([text])[0]  # shape = (n_dims,)
        # Clip to allowed ranges
        for j, d in enumerate(self.dim_ids):
            lo = float(self.space.dimensions.loc[self.space.dimensions.dimension_id == d, "scale_min"].iloc[0])
            hi = float(self.space.dimensions.loc[self.space.dimensions.dimension_id == d, "scale_max"].iloc[0])
            vec[j] = float(np.clip(vec[j], lo, hi))
        return vec
