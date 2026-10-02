# concept-embeddings-RAG-Edge

Torneo de estrategias de recuperación para RAG con varias pruebas por pregunta: qué estrategia recupera todas las pruebas más a menudo con un coste dado, medida contra lo mejor de la literatura reproducido en el mismo banco, y si conserva la ventaja en un corpus que nadie miró al elegirla.
Sucede a `concept-embeddings-RAG` (cerrado el 2026-10-01); el punto de partida está en `docs/refs/`.

## Siempre
- Actúa como interlocutor único; para desarrollar carga `.agents/skills/sdd-lite/SKILL.md`.
- Declara suposiciones y dudas antes de programar; pregunta cuando cambien el resultado.
- Haz lo mínimo que cumpla el criterio: sin funciones, abstracciones ni configuración no pedidas.
- Cambios quirúrgicos: toca solo lo que exige la petición y respeta el estilo existente.
- Trabaja contra criterios observables; reproduce fallos en el recorrido real del usuario antes de corregirlos.
- Cada resultado lleva evidencia vigente; corrige defectos claros relacionados y registra los ajenos.
- Lee solo el contexto necesario; la exploración amplia va a subagentes que devuelven resúmenes.
- Guion simple; en Markdown largo, una oración por línea; salidas Python sin símbolos especiales.
- Una fuente por decisión; los aprendizajes reutilizables van al archivo más específico.
- Este archivo tiene un tope de 5.000 tokens y no se edita a mitad de sesión; una regla nueva exige evidencia y, si no cabe, otra sale.

## Investigación
- Todo lo que produce, elige o informa una cifra medida carga antes `.agents/skills/research-protocol/SKILL.md`.
- Nunca se elige mirando el examen; nunca se presenta una proyección o una lectura como medida.
- El control es siempre lo mejor y barato de la literatura, reproducido aquí; si el control de trabajo es más débil, dilo primero y claro.
- Documentos y código en inglés; con el usuario, en español llano, con cada término técnico explicado la primera vez.
  Al pedir una aprobación, resume en español lo que queda congelado; la traducción completa se hace solo si la pide, fuera del repo.
- Los artefactos del proyecto anterior se leen en su sitio, solo lectura, contra su digest; nunca se mueven ni se reescriben.

## Pregunta antes
- Decisiones pendientes del usuario, costes o dependencias nuevas y cambios a criterios congelados.
- Cualquier gasto: GPU alquilada, APIs de pago o procesado de un corpus con LLM, con su estimación antes de empezar.
- Publicar fuera de la entrega: la autorización vale para ese destino y ese contenido; respeta confidencialidad y permisos heredados.
  La entrega incluye push de la rama de trabajo y PR a `main` en el remoto `origin` del proyecto, que es público.

## Nunca
- Rebajar criterios u ocultar pruebas fallidas o no ejecutadas.
- Fusionar, aprobar tu propio trabajo o escribir en la rama por defecto; los hooks de `.claude/` lo bloquean.
- Editar archivos generados o CHANGELOG.md a mano, ni añadir al agente como coautor de commits.
- Subir al repo `.env`, credenciales, corpus en bruto ni artefactos de `data/`.

## Mapa
- Skills en `.agents/skills/`: flujo, enrutado y delegación en `sdd-lite`; revisión en `sdd-review`; entrega y No Mistakes en `sdd-delivery`;
  protocolo experimental en `research-protocol`; GPU alquilada en `remote-gpu`.
- Plan maestro y fases: `docs/plans/`; cambios sueltos: `docs/changes/`; plantillas: `docs/templates/`.
- Punto de partida: `docs/refs/successor_project_charter.md` (decisiones), `successor_project_handover.md` (estado, cifras y know-how), `research_summary.md`.
- Garantías: `.claude/settings.json` y `.claude/hooks/`; comprobaciones: `npm run check`.
- Entorno: Windows 11 ARM64 sin GPU, Node.js 22, Git y GitHub CLI; el trabajo pesado va a GPU alquilada.
