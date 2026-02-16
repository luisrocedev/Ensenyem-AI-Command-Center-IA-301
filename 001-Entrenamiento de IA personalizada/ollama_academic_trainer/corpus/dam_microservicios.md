# Microservicios en DAM2

Los microservicios son un estilo de arquitectura donde una aplicación se divide en servicios pequeños, independientes y desplegables de forma autónoma.

## Ventajas principales

- Escalabilidad independiente por servicio.
- Despliegue continuo por módulos.
- Aislamiento de fallos: un servicio puede fallar sin tumbar todo el sistema.

## Inconvenientes

- Mayor complejidad operativa.
- Necesidad de observabilidad (logs, métricas, trazas).
- Gestión de comunicación entre servicios (REST, gRPC, colas).

## Buenas prácticas

- Diseñar contratos de API estables.
- Aplicar tolerancia a fallos con timeouts y reintentos.
- Usar circuit breaker para evitar cascadas de errores.
- Mantener base de datos por servicio cuando sea posible.

## Seguridad en microservicios

- Autenticación entre servicios con tokens.
- Autorización por scopes/roles.
- Cifrado TLS en tránsito.
- Validación estricta de entrada en cada endpoint.
