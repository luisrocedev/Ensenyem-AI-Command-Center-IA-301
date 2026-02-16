# ORM y base de datos en proyectos DAM

Un ORM (Object Relational Mapper) permite mapear entidades de código a tablas relacionales.

## Conceptos clave

- Entidad: clase que representa una tabla.
- Repositorio: capa para consultas y persistencia.
- Migraciones: versionado incremental del esquema.

## Ventajas

- Reduce SQL repetitivo.
- Mejora mantenibilidad.
- Facilita pruebas de integración.

## Riesgos

- Consultas N+1 por mala estrategia de carga.
- Sobrecoste en consultas complejas.

## Recomendaciones

- Monitorizar consultas lentas.
- Añadir índices en columnas de filtrado frecuente.
- Combinar ORM con SQL nativo para reportes complejos.
