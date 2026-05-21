# Frame Solver — Pórtico 2D por Método Matricial de la Rigidez

Aplicación Python con interfaz Streamlit para el análisis estructural de pórticos planos (2D frames) usando el Método Matricial de la Rigidez.

## Características

- **3 GDL por nodo:** desplazamiento horizontal (u), vertical (v), rotación (θ).
- **Propiedades independientes por barra:** módulo de elasticidad (E), área (A), momento de inercia (I).
- **Cargas nodales:** fuerzas horizontales, verticales y momentos puntuales.
- **Apoyos arbitrarios:** empotrado, articulado, rodillo, libre.
- **Visualización:** estructura original + deformada escalada con Matplotlib.
- **Tablas editables** con `st.data_editor` (agregar/eliminar barras dinámicamente).
- **Matrices intermedias visibles:** k', T, k_global, K ensamblada, partición de GDL.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Estructura

```
frame_solver/
├── core/        # Motor: Node, Element, Frame2DSolver
├── ui/          # Interfaz Streamlit + visualización Matplotlib
└── tests/       # pytest (11 tests, equilibrio, simetría, mecanismos)
```

## Caso de prueba

Pórtico simple de 3 barras (2 columnas + 1 viga), empotrado en ambas bases, con carga lateral y vertical. Verificado contra equilibrio global y simetría de matrices.
