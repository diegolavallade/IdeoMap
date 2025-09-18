# train_ae1d_enhanced.py
# Entrena un Autoencoder 1D y genera una visualización interactiva con detalles semánticos.

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- Rutas: Asegurarse de que la raíz del proyecto esté en el path de Python ---
# Esto permite importar 'embedder' y 'nn_reducer' correctamente.
# Asume que este script está en un subdirectorio como 'scripts/' o 'nn/'.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedder import EmbeddingSpace
from nn_reducer import train_autoencoder, AEConfig

# Configuración del AE para que el bottleneck (capa de compresión) sea 1D
@dataclass
class AEConfig1D(AEConfig):
    bottleneck: int = 1

def _json_default(o):
    """Convierte tipos de numpy a tipos nativos de Python para la serialización JSON."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_details(space: EmbeddingSpace, X_df: pd.DataFrame, coords: pd.DataFrame, meta: dict | None, score_threshold: float = 0.2) -> list[dict]:
    """
    Crea un arreglo de dicts con detalles enriquecidos para cada ideología.
    Incluye una lectura general y la composición detallada usando los datos de dimensions.csv.
    """
    # Crear un mapa de dimension_id -> info para búsquedas rápidas
    dim_info_map = space.dimensions.set_index('dimension_id').to_dict('index')
    
    # Mapear ideology_id -> índice para empatar datos aunque vengan desordenados
    id_series = space.ideologies["ideology_id"]
    id_to_idx = {str(val): i for i, val in enumerate(id_series.values)}

    # Extraer resumen del entrenamiento si existe
    training_summary = {}
    if isinstance(meta, dict):
        keys = ("best_val_loss", "epochs_run", "patience", "input_dim", "device_used")
        training_summary = {k: meta.get(k) for k in keys if k in meta}

    details = []
    coords = coords.reset_index(drop=True)

    for i, row in coords.iterrows():
        ideology_id = str(row["ideology_id"])
        idx = id_to_idx.get(ideology_id, i)
        
        # Vector de puntuaciones para esta ideología
        vec = X_df.iloc[idx]
        
        # --- Composición Ideológica y Lectura General ---
        composition = []
        for dim_id, score in vec.items():
            # Ignorar dimensiones con puntuación muy baja para no saturar
            if abs(score) < score_threshold:
                continue
            
            dim_info = dim_info_map.get(dim_id, {})
            position_label = dim_info.get('pos_label', 'Positivo') if score > 0 else dim_info.get('neg_label', 'Negativo')
            
            composition.append({
                "dim_name": dim_info.get('name', dim_id),
                "dim_description": dim_info.get('description', ''),
                "score": float(score),
                "position_label": position_label
            })
        
        # Ordenar por la magnitud de la puntuación para mostrar lo más relevante primero
        composition.sort(key=lambda x: abs(x['score']), reverse=True)
        
        # Crear una "lectura general" con las 5 características más definitorias
        summary_points = [item['position_label'] for item in composition[:5]]

        item = {
            "index": int(i),
            "ideology_id": ideology_id,
            "name": str(row["name"]),
            "coords": [float(row["Z1"])],
            "summary_points": summary_points,
            "composition": composition,
            "training": training_summary,
        }
        details.append(item)
    return details


def make_interactive_1d(coords: pd.DataFrame, details: list[dict], out_dir: Path, auto_open: bool = False) -> Path | None:
    """Genera un HTML interactivo con Plotly para la visualización 1D."""
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        print("⚠️  'plotly' no está instalado. Instálalo con:  pip install plotly")
        return None

    fig = go.Figure(data=[go.Scatter(
        x=coords["Z1"], y=np.zeros(len(coords)),
        mode="markers", marker=dict(size=8, opacity=0.8),
        text=coords["name"], customdata=np.stack([coords["ideology_id"].values], axis=-1),
        hovertemplate="<b>%{text}</b><br>Z1=%{x:.4f}<extra></extra>",
    )])
    fig.update_layout(
        title="Autoencoder 1D del Espacio Ideológico (Interactivo)",
        xaxis_title="Dimensión Latente (Z1)",
        yaxis=dict(showticklabels=False, zeroline=True, showgrid=False),
        margin=dict(l=20, r=20, b=20, t=50), height=300,
    )

    plot_html = to_html(fig, include_plotlyjs="cdn", full_html=False, div_id="ae1d_plot")
    details_json = json.dumps(details, ensure_ascii=False, default=_json_default)

    html_template = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
<title>AE1D – Espacio Ideológico</title>
<style>
  body {{ margin: 0; font-family: system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; }}
  .container {{ display: flex; flex-direction: column; height: 100%; }}
  #plot_container {{ padding: 1rem; border-bottom: 1px solid #ddd; }}
  #info_container {{ flex-grow: 1; padding: 1rem; overflow-y: auto; background: #f9f9f9; }}
  .placeholder {{ color: #6c757d; }}
  h3 {{ margin-top: 0; }} h4 {{ margin-bottom: 0.5rem; }}
  ul {{ margin-top: 0.5rem; padding-left: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 600px; margin-top: 0.5rem; }}
  th, td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: left; font-size: 0.9em; }}
  th {{ background-color: #f2f2f2; }}
  td:last-child {{ font-variant-numeric: tabular-nums; text-align: right; }}
</style>
</head>
<body>
<div class="container">
  <div id="plot_container">{plot_html}</div>
  <div id="info_container"><p class="placeholder">Haz clic en un punto para ver sus detalles.</p></div>
</div>
<script>
    const DETAILS = {details_json};
    const panel = document.getElementById('info_container');

    function renderDetails(d) {{
      // 1. Resumen o Lectura General
      const summaryHtml = d.summary_points && d.summary_points.length
        ? `<h4>Lectura General</h4><ul>` + d.summary_points.map(s => `<li>${{s}}</li>`).join('') + `</ul>`
        : '';
      
      // 2. Tabla de Composición Ideológica
      const compositionHtml = d.composition && d.composition.length
        ? `<h4>Composición Ideológica</h4><table>
            <thead><tr><th>Dimensión</th><th>Posición</th><th>Valor</th></tr></thead>
            <tbody>` +
            d.composition.map(c => `
              <tr>
                <td>${{c.dim_name}}</td>
                <td>${{c.position_label}}</td>
                <td>${{c.score.toFixed(3)}}</td>
              </tr>`).join('') +
            `</tbody></table>`
        : '';

      return `<h3>${{d.name}}</h3>
        <p><b>ID:</b> ${{d.ideology_id}} / <b>Coordenada Z1:</b> ${{d.coords[0].toFixed(4)}}</p>
        ${{summaryHtml}}
        ${{compositionHtml}}`;
    }}

    window.addEventListener('DOMContentLoaded', function() {{
      const plot = document.getElementById('ae1d_plot');
      if (!plot) return;
      plot.on('plotly_click', function(ev) {{
        if (!ev || !ev.points || !ev.points.length) return;
        const idx = ev.points[0].pointIndex;
        const detailsData = DETAILS[idx];
        panel.innerHTML = detailsData ? renderDetails(detailsData) : '<p><em>Sin detalles disponibles.</em></p>';
      }});
    }});
</script>
</body>
</html>"""

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "ae1d_enhanced.html"
    html_path.write_text(html_template, encoding="utf-8")

    if auto_open:
        import webbrowser
        webbrowser.open(html_path.as_uri())

    return html_path


def main():
    p = argparse.ArgumentParser(description="Entrena AE 1D y genera una visualización semántica interactiva.")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--device", type=str, default="cuda", help="cuda o cpu")
    p.add_argument("--no-train", action="store_true", help="Usa coords existentes de data/ae1d_coords.csv")
    p.add_argument("--open", action="store_true", help="Abre el HTML en el navegador al terminar.")
    args = p.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    out_models = base_path / "models" / "ae1d"
    out_models.mkdir(parents=True, exist_ok=True)
    fig_dir = base_path / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = base_path / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    coords_path = data_dir / "ae1d_coords.csv"

    print("Cargando espacio ideológico desde los CSVs...")
    space = EmbeddingSpace(str(base_path))
    space.load()
    X_df = space.X.astype(np.float32)

    meta = {}
    if not args.no_train or not coords_path.exists():
        print("🚀 Entrenando modelo Autoencoder 1D...")
        cfg = AEConfig1D(input_dim=X_df.shape[1], epochs=args.epochs, patience=args.patience, device=args.device)
        Z, meta = train_autoencoder(X_df.values, cfg, out_models)

        coords = pd.DataFrame(Z, columns=["Z1"])
        coords.insert(0, "name", space.ideologies["name"].values)
        coords.insert(0, "ideology_id", space.ideologies["ideology_id"].values)
        coords.to_csv(coords_path, index=False)
        print(f"✅ Coords 1D guardadas en {coords_path}")
    else:
        coords = pd.read_csv(coords_path)
        print(f"ℹ️ Usando coords 1D existentes: {coords_path}")

    print("🛠️ Construyendo detalles enriquecidos para la visualización...")
    details = build_details(space, X_df, coords, meta)
    
    print("🌐 Generando HTML interactivo...")
    html_path = make_interactive_1d(coords.reset_index(drop=True), details, fig_dir, auto_open=args.open)
    if html_path:
        print(f"✨ Visualización interactiva lista: {html_path.resolve()}")
        if not args.open:
            print("   Ejecuta con --open para abrirla automáticamente en tu navegador.")


if __name__ == "__main__":
    main()