# train_ae3d_colored.py
# Entrena un Autoencoder 3D y genera una visualización interactiva, robusta y
# con colores procedurales basados en la posición espacial para resaltar clústeres.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Rutas: Asegurarse de que la raíz del proyecto esté en el path de Python ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedder import EmbeddingSpace
from nn_reducer import AEConfig, train_autoencoder

def _json_default(o):
    """Convierte tipos de numpy a tipos nativos para la serialización JSON."""
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

def assign_procedural_colors(coords: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna un color a cada punto basado en su posición 3D (Z1->R, Z2->G, Z3->B).
    Normaliza cada eje a [0, 1] para crear un gradiente de color suave.
    """
    df = coords.copy()
    for i, axis in enumerate(["Z1", "Z2", "Z3"]):
        channel = ['R', 'G', 'B'][i]
        min_val = df[axis].min()
        max_val = df[axis].max()
        # Evitar división por cero si todos los puntos están en un plano
        if max_val - min_val > 1e-6:
            df[channel] = (df[axis] - min_val) / (max_val - min_val)
        else:
            df[channel] = 0.5  # Asignar un valor neutral si no hay rango
    
    # Crear la cadena de color 'rgb(r,g,b)' para Plotly
    df['color_str'] = df.apply(
        lambda row: f"rgb({int(row['R']*255)}, {int(row['G']*255)}, {int(row['B']*255)})",
        axis=1
    )
    return df

def build_details_map(space: EmbeddingSpace, coords: pd.DataFrame, meta: dict | None, score_threshold: float = 0.2) -> dict:
    """Crea un DICCIONARIO de detalles {ideology_id: details}."""
    dim_info_map = space.dimensions.set_index('dimension_id').to_dict('index')
    
    full_df = pd.merge(coords, space.ideologies, on=['ideology_id', 'name'])

    training_summary = {}
    if isinstance(meta, dict):
        keys = ("best_val_loss", "epochs_run", "device_used")
        training_summary = {k: meta.get(k) for k in keys if k in meta}

    details_map = {}
    for _, row in full_df.iterrows():
        ideology_id = str(row["ideology_id"])
        vec = row[space._dim_ids]
        
        composition = []
        for dim_id, score in vec.items():
            if abs(score) < score_threshold: continue
            dim_info = dim_info_map.get(dim_id, {})
            position_label = dim_info.get('pos_label', 'Positivo') if score > 0 else dim_info.get('neg_label', 'Negativo')
            composition.append({
                "dim_name": dim_info.get('name', dim_id),
                "score": float(score),
                "position_label": position_label
            })
        
        composition.sort(key=lambda x: abs(x['score']), reverse=True)
        summary_points = [item['position_label'] for item in composition[:5]]

        item = {
            "ideology_id": ideology_id, "name": str(row["name"]),
            "coords": [float(row["Z1"]), float(row["Z2"]), float(row["Z3"])],
            "color_str": str(row.get('color_str', 'rgb(128,128,128)')), # Añadir color a los detalles
            "summary_points": summary_points, "composition": composition, "training": training_summary,
        }
        details_map[ideology_id] = item
    return details_map

def make_interactive_colored(coords_with_colors: pd.DataFrame, details_map: dict, out_dir: Path, auto_open: bool = False) -> Path | None:
    """Genera un HTML interactivo robusto con puntos coloreados."""
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        print("⚠️  'plotly' no está instalado. Instálalo con:  pip install plotly")
        return None

    fig = go.Figure(data=[go.Scatter3d(
        x=coords_with_colors["Z1"], y=coords_with_colors["Z2"], z=coords_with_colors["Z3"],
        mode="markers",
        marker=dict(
            size=5, # Un poco más grandes para apreciar el color
            color=coords_with_colors['color_str'], # Aplicar los colores procedurales
            opacity=0.9
        ),
        text=coords_with_colors["name"],
        customdata=coords_with_colors["ideology_id"].astype(str).values,
        hovertemplate="<b>%{text}</b><br>ID: %{customdata}<br>" +
                      "Z1=%{x:.3f}<br>Z2=%{y:.3f}<br>Z3=%{z:.3f}<extra></extra>",
    )])
    fig.update_layout(
        title="Autoencoder 3D del espacio ideológico (Coloreado por Posición)",
        scene=dict(xaxis_title="Z1 (Rojo)", yaxis_title="Z2 (Verde)", zaxis_title="Z3 (Azul)", aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40), showlegend=False,
    )

    plot_html = to_html(fig, include_plotlyjs="cdn", full_html=False, div_id="ae3d_plot")
    details_json = json.dumps(details_map, ensure_ascii=False, default=_json_default)

    html = f"""<!doctype html><html lang="es"><head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
<title>AE3D Coloreado – Espacio Ideológico</title>
<style>
  body {{ margin: 0; font-family: system-ui, sans-serif; }}
  .wrap {{ display: grid; grid-template-columns: 2fr 1fr; gap: 12px; height: 100vh; }}
  #ae3d_plot {{ width: 100%; height: 100%; }}
  #info {{ padding: 12px 15px; overflow-y: auto; border-left: 1px solid #e5e7eb; background: #f9f9f9; }}
  #info h3 {{ margin-top: 0; padding-bottom: 5px; border-bottom: 2px solid; }}
  #info h4 {{ margin-bottom: 0.5rem; border-bottom: 1px solid #ddd; padding-bottom: 4px;}}
  #info table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
  .placeholder {{ color: #6b7280; font-style: italic; }}
</style></head><body>
  <div class="wrap">
    {plot_html}
    <aside id="info"><p class="placeholder">Haz clic en un punto para ver detalles.</p></aside>
  </div>
  <script>
    const DETAILS = {details_json};
    const panel = document.getElementById('info');
    function renderDetails(d) {{
      // Usa el color del punto para la cabecera
      panel.querySelector('h3').style.borderBottomColor = d.color_str || '#333';
      const summaryHtml = d.summary_points && d.summary_points.length ? `<h4>Lectura General</h4><ul>` + d.summary_points.map(s => `<li>${{s}}</li>`).join('') + `</ul>` : '';
      const compositionHtml = d.composition && d.composition.length ? `<h4>Composición Ideológica</h4>...` : ''; // Resto del HTML igual
      return `<h3>${{d.name}}</h3> ...`; // El resto de la función render es igual
    }}
    // El resto del JS (lógica de clic y renderizado) es idéntico al de la versión robusta.
    // Para brevedad, se omite aquí, pero está completo en el script.
  </script>
</body></html>"""
    # ... (El JS completo se inyectará en la plantilla real)
    # Re-insertamos el JS completo aquí para asegurar que el script funcione:
    full_js = f"""
    <script>
        const DETAILS = {details_json};
        const panel = document.getElementById('info');

        function renderDetails(d) {{
          const z = d.coords || [0,0,0];
          const coordsText = `Z1=${{z[0].toFixed(3)}}, Z2=${{z[1].toFixed(3)}}, Z3=${{z[2].toFixed(3)}}`;

          const summaryHtml = d.summary_points && d.summary_points.length ? `<h4>Lectura General</h4><ul>` + d.summary_points.map(s => `<li>${{s}}</li>`).join('') + `</ul>` : '';
          
          const compositionHtml = d.composition && d.composition.length ? `<h4>Composición Ideológica</h4><table style="width:100%; border-collapse:collapse;">
                <thead style="background-color:#f2f2f2;"><tr><th style="padding:6px; border:1px solid #ddd;">Dimensión</th><th style="padding:6px; border:1px solid #ddd;">Posición</th><th style="padding:6px; border:1px solid #ddd; text-align:right;">Puntaje</th></tr></thead>
                <tbody>` +
                d.composition.map(c => `
                  <tr>
                    <td style="padding:6px; border:1px solid #ddd;">${{c.dim_name}}</td>
                    <td style="padding:6px; border:1px solid #ddd;">${{c.position_label}}</td>
                    <td style="padding:6px; border:1px solid #ddd; text-align:right;">${{c.score.toFixed(3)}}</td>
                  </tr>`).join('') +
                `</tbody></table>`
            : '';
          
          const header_html = `<h3 style="border-bottom-color: ${{d.color_str}};">${{d.name}}</h3>`;

          return header_html + `<p><b>ID:</b> ${{d.ideology_id}}</br><b>Coords:</b> ${{coordsText}}</p>
            ${{summaryHtml}}
            ${{compositionHtml}}`;
        }}

        window.addEventListener('DOMContentLoaded', function() {{
          const plot = document.getElementById('ae3d_plot');
          if (!plot) return;
          plot.on('plotly_click', function(ev) {{
            if (!ev || !ev.points || !ev.points.length) return;
            const ideologyId = ev.points[0].customdata;
            const d = DETAILS[ideologyId];
            panel.innerHTML = d ? renderDetails(d) : '<p class="placeholder">Error: No se encontraron detalles para el ID: ' + ideologyId + '</p>';
          }});
        }});
      </script>"""
    
    # Inyectamos el plot y el script completo en la plantilla principal
    full_html = html.split('<body>')[0] + "<body>" + html.split('<body>')[1].split('<script>')[0] + full_js + "</body></html>"


    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "ae3d_colored.html"
    html_path.write_text(full_html, encoding="utf-8")

    if auto_open:
        import webbrowser
        webbrowser.open(html_path.as_uri())

    return html_path


def main():
    p = argparse.ArgumentParser(description="Entrena AE 3D y genera visualización coloreada y robusta.")
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-train", action="store_true", help="Usa coords existentes de data/ae3d_coords.csv.")
    p.add_argument("--open", action="store_true", help="Abre el HTML en el navegador al terminar.")
    args = p.parse_args()

    base = Path(__file__).resolve().parent.parent
    out_models = base / "models" / "ae3d"
    out_models.mkdir(parents=True, exist_ok=True)
    fig_dir = base / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir = base / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    coords_path = data_dir / "ae3d_coords.csv"

    print("Cargando espacio ideológico desde los CSVs...")
    space = EmbeddingSpace(str(base))
    space.load()

    meta = {}
    if not args.no_train or not coords_path.exists():
        print("🚀 Entrenando modelo Autoencoder 3D...")
        cfg = AEConfig(input_dim=space.X.shape[1], epochs=args.epochs, patience=args.patience, device=args.device)
        Z, meta = train_autoencoder(space.X.values.astype(np.float32), cfg, out_models)
        coords = pd.DataFrame(Z, columns=["Z1", "Z2", "Z3"])
        coords.insert(0, "name", space.ideologies["name"].values)
        coords.insert(0, "ideology_id", space.ideologies["ideology_id"].values)
        coords.to_csv(coords_path, index=False)
        print(f"✅ Coords 3D guardadas en {coords_path}")
    else:
        coords = pd.read_csv(coords_path)
        print(f"ℹ️ Usando coords 3D existentes: {coords_path}")

    print("🎨 Asignando colores procedurales a los puntos...")
    coords_with_colors = assign_procedural_colors(coords)

    print("🛠️ Construyendo mapa de detalles enriquecidos...")
    details_map = build_details_map(space, coords_with_colors, meta)
    
    # --- PNG estático con colores ---
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        # Matplotlib puede usar directamente los canales [0,1] como colores
        ax.scatter(
            coords_with_colors["Z1"], coords_with_colors["Z2"], coords_with_colors["Z3"],
            s=15, alpha=0.9, c=coords_with_colors[['R', 'G', 'B']].values
        )
        ax.set_xlabel("Z1 (Rojo)"); ax.set_ylabel("Z2 (Verde)"); ax.set_zlabel("Z3 (Azul)")
        ax.set_title("Autoencoder 3D (Coloreado por Posición)")
        png_path = fig_dir / "ae3d_colored.png"
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"🖼️ PNG coloreado guardado en {png_path}")
    except Exception as e:
        print(f"⚠️ No se pudo generar PNG: {e}")

    print("🌐 Generando HTML interactivo coloreado...")
    html_path = make_interactive_colored(coords_with_colors, details_map, fig_dir, auto_open=args.open)
    if html_path:
        print(f"✨ Visualización interactiva lista: {html_path.resolve()}")
        if not args.open:
            print("   Ejecuta con --open para abrirla automáticamente.")

if __name__ == "__main__":
    main()