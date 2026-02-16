# Portafolio IA · 001 + 002 + 003

Este directorio agrupa tres mini proyectos académicos conectados en una evolución técnica única:

1. `001-Entrenamiento de IA personalizada`
2. `002-IA Generativa`
3. `003-AgenteIA`

## Relación entre actividades

- **001** construye el entrenamiento semántico local (ingesta, embeddings, recuperación).
- **002** usa ese conocimiento para generación de contenido empresarial.
- **003** añade comportamiento de agente con políticas, canales y trazas.

## Dónde está el código principal

El código operativo unificado está en:

`002-IA Generativa/ensenyem_generative_studio`

Esto permite reutilizar contexto y no duplicar lógica entre actividades.

## Memorias de entrega

- `001-Entrenamiento de IA personalizada/Actividad_EntrenamientoIA_53945291X.md`
- `002-IA Generativa/Actividad_IAGenerativa_53945291X.md`
- `003-AgenteIA/Actividad_AgenteIA_53945291X.md`

## Ejecución del prototipo global

```bash
cd "002-IA Generativa/ensenyem_generative_studio"
pip install -r requirements.txt
python app.py
```

URL local: `http://127.0.0.1:5102`
