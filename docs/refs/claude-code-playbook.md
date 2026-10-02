# Claude Code Playbook - Construir sistemas que hacen fiable a la IA

> Resumen estructurado de un artículo de terceros.
> Autor: Jason Zhou (AI Jason), YouTube Creator y Product Designer, fundador de AI Builder Club.
> Fuente: https://offers.hubspot.com/view/claude-code-playbook
> Capturado: 2026-09-23.
> Los precios, versiones y flags proceden del artículo y conviene verificarlos contra la documentación oficial antes de aplicarlos.

---

## Idea central

El modelo adivina; el sistema no tiene por qué hacerlo.

El trabajo del modelo es adivinar bien.
El trabajo del sistema es garantizar que una mala suposición no rompa nada.
Claude Code no es una sola cosa que se promptea: es un stack de capas, y cada capa hace más difícil descarrilar al agente.

Los ingenieros que sacan 10x de Claude Code no escriben prompts mágicos.
Han construido el andamiaje que hace fiable al agente.

### El stack por capas

| Capa | Qué garantiza | Analogía humana |
|---|---|---|
| Config (`CLAUDE.md`) | El agente conoce las reglas y convenciones del proyecto | Documento de onboarding |
| Patrones de prompting | El agente recibe instrucciones específicas y conscientes del código | Brief claro |
| Hooks | Las reglas críticas se ejecutan siempre, sin excepción | Maquinaria de pipeline |
| Task state | El trabajo largo se mantiene ordenado y reanudable | Checklist de cirujano |
| Sub-agents / Teams / Worktrees | Trabajo paralelo sin inflar contexto ni pisar archivos | Coordinación de equipo |
| Cost controls | La factura se corresponde con el valor, no con el hype | Presupuesto |

**Regla de decisión:** si un fallo cuesta dinero real, datos reales o confianza real, no pertenece a un prompt.
Pertenece a la maquinaria (hooks).

---

## Parte 1 - Fundamentos

Claude Code es el agente CLI oficial de Anthropic para ingeniería de software.
Se distribuye como paquete npm y corre en cualquier terminal: macOS, Linux, WSL en Windows.

La diferencia frente a un chatbot es concreta: acceso al sistema de archivos, ejecución de shell e integración con git.
Lee el repositorio, escribe archivos, ejecuta `npm test` y hace commits.
No hay copiar y pegar entre una ventana de chat y el editor.

### Instalación

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

### Modelo de facturación

Primera decisión real, y condiciona el coste después.

- **Claude Pro/Max** (login de Claude).
- **API Key** (`ANTHROPIC_API_KEY`): aproximadamente 3 $/M tokens de entrada y 15 $/M de salida en Sonnet.
  Recomendado para equipos, uso intensivo, scripts y CI.
  Una feature típica cuesta entre 1 y 3 $.

Se alterna con `claude logout` / `claude login`.
Muchos builders van en híbrido: Pro para trabajo diario en el IDE, API key para trabajos por lotes y automatización.

### CLAUDE.md

Lo primero tras instalar.
Es un archivo plano en la raíz del repositorio que el agente carga en contexto cada sesión: un system prompt persistente para el código.
Sin él, cada sesión empieza de cero.

Plantilla mínima que se gana su sitio:

```markdown
# CLAUDE.md

## Stack
Next.js 14 App Router, TypeScript strict, Tailwind, Supabase, Stripe.

## Conventions (max ~10 rules)
- API routes: see app/api/stripe/checkout/route.ts as reference
- Use server components by default, 'use client' only when needed
- All prices in cents for Stripe

## Never do (max ~5)
- Don't install new dependencies without asking
- Don't modify middleware.ts without explicit approval
- Don't change Stripe webhook handling without confirmation

## Commands
- Build: npm run build   Test: npm run test   Lint: npm run lint
```

Mantenerlo entre 200 y 400 líneas.
Cada línea se carga en cada turno, así que un `CLAUDE.md` inflado es una fuga de coste silenciosa.

### El flujo de trabajo que funciona

Explore -> Plan -> Code -> Commit.

En la fase de exploración, pedir a Claude que lea los archivos relevantes y explique la arquitectura actual antes de tocar nada.

### Los cuatro principios de Karpathy para CLAUDE.md

Andrej Karpathy pasó de 80 % de código manual a 80 % dirigido por agentes en semanas.
Luego nombró los cuatro modos de fallo que se le repetían y escribió una regla para cada uno.

| Modo de fallo | Regla que lo corrige |
|---|---|
| El modelo hace suposiciones erróneas en silencio | **Think Before Coding**: declarar suposiciones y sacar a la luz la confusión primero |
| Sobrecomplica el código | **Simplicity First**: nada de features, abstracciones o configuración más allá de lo pedido |
| Toca código no relacionado | **Surgical Changes**: tocar solo lo que la petición exige, respetando el estilo existente |
| Ejecución débil, necesita que le lleven de la mano | **Goal-Driven Execution**: dar criterios de éxito, no instrucciones paso a paso, para que el modelo itere solo |

### Sugerencia frente a garantía

Las instrucciones de `CLAUDE.md` son sugerencias que el modelo normalmente sigue.
Es la herramienta correcta para "preferir comentarios en inglés".
Es la herramienta equivocada para "formatea cada archivo" (falla 1 de cada 10) o "nunca hagas force-push a main" (un fallo = desastre).
Para eso hacen falta hooks (Parte 3).

---

## Parte 2 - Diez patrones de prompting

La habilidad de mayor ROI y la más barata de aprender.

La mayoría escribe "add a login page" y se queja de que la salida es genérica.
El problema no es el modelo, es el prompt.

> Nota: en la captura de la página cada patrón aparecía colapsado, con el título visible pero sin el ejemplo "versión mala / versión corregida".
> Se conservan aquí los títulos y la explicación agregada que sí aparece en el artículo.

1. **Referenciar patrones existentes.**
2. **Describir el resultado, no los pasos.**
3. **El sándwich de restricciones** (resultado + restricciones + listón de calidad).
4. **Leer antes de escribir.**
5. **El marco "senior engineer".**
6. **Alcance incremental con checkpoints.**
7. **Dar el "por qué".**
8. **El patrón de migración** (origen + destino + verificación).
9. **Depurar con contexto, no con síntomas.**
10. **Refactor con red de seguridad.**

### Por qué funcionan, en una frase cada uno

- La especificidad elimina las conjeturas.
- Los resultados desbloquean implementaciones mejores que las que uno prescribiría.
- Las restricciones evitan el scope creep.
- Leer primero mata el código duplicado.
- El marco senior fija prioridades correctas (los tests definen la corrección).
- Los checkpoints acotan el radio de daño de un giro equivocado.
- El "por qué" alimenta el contexto de decisión.
- El paso de verificación compra confianza en que el cambio no rompió nada en silencio.

### El meta-patrón

Tratar a Claude Code como a un fichaje con talento en su primer día.
Sabe escribir buen código, pero no conoce el código base, ni las preferencias, ni esa cosa legacy rara que se rompe al tocar cierto archivo.

Contexto, restricciones y pasos de verificación son el onboarding.
Cuanto mejor el onboarding, mejor la salida.

> Los prompts específicos y anclados en referencias ganan siempre a la frase ingeniosa.
> El repositorio es el contexto. Úsalo.

---

## Parte 3 - Hooks: las reglas que la IA no puede ignorar

Un hook es un comando externo cableado al pipeline de ejecución de Claude Code que se dispara automáticamente en un momento concreto, y que el modelo no puede saltarse.

Es el salto de "normalmente" a "siempre".

Un builder mantuvo un bloqueador de force-push durante nueve meses.
Se disparó 8 veces.
Cada una fue Claude decidiendo que "limpiar el historial de git parece buena idea".
Ocho incidentes evitados con un script diminuto.

### Eventos del ciclo de vida

| Evento | Cuándo se dispara | Para qué sirve |
|---|---|---|
| `PreToolUse` | Antes de cualquier llamada a herramienta | El único evento que puede bloquear una acción |
| `PostToolUse` | Tras completarse una llamada a herramienta | Formatear, lint, tests |
| `Stop` | Cuando Claude intenta terminar | Puerta de calidad: no dejar que pare con tests fallando |
| `Notification` | Cuando Claude necesita atención | Reenviar a escritorio o Slack |
| `SessionStart` | Al empezar una sesión nueva | Inyectar variables de entorno, comprobar dependencias, arrancar logging |
| `UserPromptSubmit` | Al enviar un mensaje | Adjuntar contexto extra antes de que el modelo lo vea |

### Las tres decisiones

Cada hook devuelve una de tres:

- **Allow**: dejar que el pipeline continúe.
- **Block**: detener la acción (solo en `PreToolUse`).
- **Inject**: añadir información a la conversación, por ejemplo devolver los 3 avisos del linter para que Claude los arregle.

Tres verbos, compuestos entre eventos, cubren casi toda la automatización que se necesita.

### Matchers y tipos de handler

Los **matchers** acotan cuándo se dispara un hook, por nombre de herramienta (regex):

- `"Bash"` para shell solamente.
- `"Write|Edit"` para cambios en archivos.
- `""` para todo (logs de auditoría).

**Tipos de handler**:

- **Command**: un script de shell.
  Cerca del 90 % de los hooks, rápido y gratis.
  Empezar aquí.
- **HTTP**: envía a una API.
- **Prompt**: un modelo juzga zonas grises; consume tokens.
- **Agent**: comprobaciones pesadas multi-paso, poco frecuentes.

### Configuración

Vive en `settings.json` en tres niveles aditivos:

- `~/.claude/settings.json`: todos los proyectos (seguridad, auditoría, notificaciones).
- `.claude/settings.json`: este repositorio, versionado (reglas compartidas del equipo).
- `.claude/settings.local.json`: ignorado por git (preferencias personales).

### Los siete hooks del artículo

1. Auto-formatear tras editar.
2. Bloquear force-push a main.
3. Proteger archivos con secretos.
4. Puerta de calidad en `Stop`.
5. Notificación de escritorio.
6. Log de auditoría completo.
7. Guardia de migraciones.

> Nota: en la captura estos siete aparecían colapsados, con el título visible pero sin la implementación, salvo el auto-formateador.

El auto-formateador son cinco minutos de setup y la victoria fácil por la que empezar:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          { "type": "command", "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write --ignore-unknown" }
        ]
      }
    ]
  }
}
```

### Reglas prácticas de campo

- **Empezar con tres**: uno de seguridad, uno de formato, uno de notificación, y convivir con ellos una semana.
  Los setups de 95 hooks que circulan por X se acumularon durante meses de incidentes reales, no se diseñaron de golpe.
- **Mantener los hooks tontos.**
  El valor de un hook es la certeza; un hook "inteligente" que llama a un LLM devuelve esa certeza.
- **Vigilar la latencia.**
  Claude hace decenas de llamadas a herramientas por tarea: mantener los hooks síncronos por debajo de unos 200 ms y empujar el trabajo lento a asíncrono.
- **Probar de verdad los hooks que bloquean.**
  Un formateador que falla en silencio no cuesta nada; un bloqueador de force-push que falla en silencio cuesta la rama main.

### Permisos frente a hooks

Los permisos son una puerta (pasa o no pasa).
Los hooks son maquinaria programable (bloquear, transformar, inyectar, registrar).

Usar permisos para acceso grueso y hooks para cualquier cosa con lógica dentro.
Se apilan: los permisos filtran primero, los hooks refinan.

---

## Parte 4 - Task state: por qué Claude Code se salta el paso 5 de 18

Pedir a Claude Code un workflow de 18 pasos y verle saltarse el paso 5 en silencio.
Sin error. Simplemente ausente.

Es estructural: las tareas largas inflan la ventana de contexto, las instrucciones tempranas derivan hacia los bordes, la atención se diluye.
Los cirujanos usan checklists exactamente por esto.

### Los dos mecanismos

| | TodoWrite | Task |
|---|---|---|
| Almacenamiento | Contexto (RAM) | Disco (`~/.claude/tasks/`) |
| Sobrevive a reinicio | No | Sí |
| Dependencias | No | Sí |
| Multi-sesión | No | Sí (ID de lista compartido) |
| Mejor para | Trabajos de una pasada, menos de ~10 pasos | Trabajo multi-día, reanudable, paralelo |

**TodoWrite** es el post-it: una checklist plana que vive en la ventana de contexto, con ítems moviéndose de `pending` a `in_progress` a `completed`.
Al cerrar la terminal desaparece.

**Task** (disponible desde la v2.1.16) es el tablero de proyecto.
Cuatro herramientas (`TaskCreate`, `TaskGet`, `TaskUpdate`, `TaskList`) con cuatro mejoras:

- Persistencia en disco (`~/.claude/tasks/`), sobrevive a reinicios.
- Grafos de dependencias reales: al marcar la Task 1 como hecha, la Task 2 se desbloquea automáticamente.
- Colaboración multi-sesión: fijar `CLAUDE_CODE_TASK_LIST_ID` en dos terminales y comparten un tablero.
- Jerarquía: epics que se descomponen en fases y estas en tareas.

### La regla de decisión

Una sola pregunta: ¿este trabajo cruza el límite de una sesión?

- **No** -> TodoWrite (bug fixes, trabajos de una pasada con menos de ~10 pasos).
- **Sí** -> Task (features multi-día, refactors ordenados por fases, cualquier cosa que se abandone el viernes y se retome el lunes).

### El patrón de potencia: los dos a la vez

Task gestiona el nivel de proyecto (features, fases, dependencias).
Dentro de la ejecución de cada task, TodoWrite lleva el paso a paso.

La estructura grande persiste en disco, la pequeña se queda barata en contexto.
Task para el mapa, TodoWrite para el giro a giro.

### La victoria más barata posible

Un builder con un comando de brief diario de 18 pasos perdía dos pasos sistemáticamente.
El arreglo fue un Paso 0 obligatorio: "crea una checklist TodoWrite con los 18 pasos antes de empezar".
Pasos saltados tras el cambio: cero.

Para workflows recurrentes, integrar ese Paso 0 en la definición del comando o de la skill.
No confiar en que el modelo se acuerde de acordarse.

> Nada de esto hace al modelo más listo.
> Todo esto hace más difícil descarrilar al sistema que rodea al modelo.

---

## Parte 5 - Sub-agents: aislamiento de contexto, no juego de roles

Un sub-agent es una sesión separada de Claude Code lanzada con la herramienta Task, con su propia ventana de contexto limpia.
Hace trabajo enfocado en aislamiento y devuelve un único resumen.

Bien usados, recortan el trabajo multi-archivo entre un 50 % y un 70 %.
Mal usados, triplican la factura de tokens sin ganancia.

La razón real de usarlos no es "que los agentes actúen como un equipo".
Es el **aislamiento de contexto**.
Un sub-agent puede leer decenas de miles de tokens de archivos en su propia ventana limpia y devolver al padre un resumen de 1 a 2 K.
El contexto del padre se mantiene ligero.

### Tres patrones donde se pagan solos

- **Exploración paralela.**
  ¿Código base desconocido?
  En vez de un agente leyendo 12 archivos en secuencia, desplegar 4 sub-agents (auth, API, base de datos, tests), cada uno devolviendo un resumen de 200 palabras.
  El padre recibe un briefing de 4 párrafos en lugar de 12 volcados de archivo.
- **Aplicación paralela.**
  Misma plantilla, distintas entradas: añadir un test a cada uno de 6 módulos.
  El tiempo de reloj baja de unos 12 minutos a unos 3; el coste total en tokens es aproximadamente el mismo porque es el mismo trabajo.
- **Personas especializadas.**
  Un "revisor de código despiadado" o un "verificador de seguridad" con un prompt ajustado en contexto fresco produce salida más afilada que la misma instrucción atornillada al contexto ya cargado del padre.

### Tres antipatrones que solo queman tokens

- **Trabajo secuencial.**
  "Explorar, luego planificar, luego programar, luego testear": cada paso necesita el contexto del anterior, que los sub-agents no comparten.
  Usar un solo agente.
- **Tareas triviales.**
  Lanzar un sub-agent tiene entre 3 y 5 segundos de sobrecarga de arranque.
  Para una tarea de 30 segundos, hacerla en línea.
- **Trabajo con estado compartido.**
  Si el sub-agent B necesita la salida del sub-agent A a mitad de vuelo, no se puede: no se comunican durante la ejecución.
  Eso es trabajo para Agent Teams (Parte 6).

### Tres plantillas que merece la pena guardar

- **Codebase Explorer**: un sub-agent por directorio, sintetizados en un brief de arquitectura.
- **Test Generator**: uno por módulo, siguiendo los patrones de test existentes.
- **Solution Surveyor**: tres enfoques a un mismo problema (librería, a medida, servicio externo), comparados y sintetizados en una recomendación.

### Matemática de costes

| Workflow | Sub-agents | Coste (Sonnet 4.5) |
|---|---|---|
| Línea base, un solo agente | 0 | ~0,45 $ |
| Exploración de código base | 4 | ~1,58 $ |
| Generación de tests | 6 | ~2,25 $ |
| Triaje de bugs | 7 | ~2,85 $ |

Esto es de 3 a 6 veces una ejecución simple, a cambio de entre un 50 % y un 80 % menos de tiempo de reloj.

Punto óptimo: de 4 a 6 en paralelo.
Por encima de ~5 en API Tier 1 se encolan y la ganancia de paralelismo se evapora.
Poner los sub-agents en Haiku o Sonnet por defecto, nunca en Opus.

> El mayor error es usar sub-agents para trabajo secuencial.
> Ejecutar 5 agentes para "planificar, programar, testear, revisar, commitear" es peor que un agente haciendo los cinco: cada uno pierde el contexto que el siguiente necesita.
> Los sub-agents son la forma de cambiar dinero por tiempo de reloj.

---

## Parte 6 - Agent Teams: cuando los trabajadores necesitan hablar

Los sub-agents tienen un techo: no pueden hablar entre ellos.
Un sub-agent corre, devuelve un resumen y muere.
Si el agente de frontend cambia la forma de una API, el de backend se entera nunca.

Agent Teams resuelve eso: múltiples instancias completas de Claude Code, un tablero de tareas compartido y mensajería directa entre compañeros.

Una ejecución real: un equipo de 3 agentes entregó una feature en 19 minutos que en solitario lleva 45, incluyendo el tiempo que un agente pasó diciéndole a otro "el contrato de la API ha cambiado, esta es la nueva forma".

### El diferenciador es el buzón

Los sub-agents son topología en estrella: todo se enruta por el padre.
Los equipos son malla: el agente de backend manda mensaje directo al de frontend.
Sin relé humano.

### Las cuatro piezas

- **Team Lead**: la sesión con la que se habla; reparte trabajo, lanza compañeros, fusiona resultados.
- **Teammates**: instancias completas de Claude Code con su propio contexto, herramientas y terminal.
- **Shared Task List**: un tablero con estados y dependencias del que los compañeros reclaman trabajo.
- **Mailbox**: mensajes directos entre compañeros.

### Activación

En `~/.claude/settings.json`:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Luego se describe el equipo en lenguaje natural:

> "Build a team for the team-settings feature: one backend agent for the Supabase CRUD API, one frontend agent for the React settings page, one test agent for unit + E2E."

### Tres escenarios donde el equipo gana al agente solitario

- **Revisión de código multi-perspectiva**: seguridad, rendimiento y tests en contextos aislados para evitar el anclaje.
  Una vez que un agente encuentra un bug de seguridad, su atención se queda en territorio de seguridad.
- **Depuración por hipótesis en competencia**: cinco teorías, en modo adversarial, con instrucción de atacar las conclusiones de los demás.
- **Desarrollo de features entre capas**: backend, frontend y tests en paralelo.

### Reglas de gestión

El rol pasa de conversar con una IA a dirigir un equipo pequeño.

- **Briefear a los compañeros como a nuevos fichajes.**
  No heredan la conversación del lead, solo `CLAUDE.md`, la configuración MCP y la descripción que se les dé.
  "Ocúpate del backend" produce basura; una especificación acotada produce trabajo.
- **Trazar fronteras de archivos.**
  Dos compañeros editando un archivo es un conflicto.
  Repartir la propiedad por adelantado (frontend posee `src/pages/`, backend posee `src/api/`).
- **Limitar el equipo a entre 3 y 5.**
  Pasado de cinco, la sobrecarga de coordinación crece más rápido que la producción.
- **Revisar cada pocos minutos.**
  Un compañero desviándose de la especificación durante 20 minutos sin supervisión son tokens que no se recuperan.
- **Planificar primero en trabajo de alto riesgo.**
  Que los compañeros redacten un enfoque en modo solo lectura y el lead apruebe antes de escribir código.

### Límites actuales (es experimental)

- Sin reanudación de sesión.
- Un equipo a la vez.
- Sin anidamiento.
- El panel dividido requiere tmux o iTerm2.

> Cada compañero es una instancia completa.
> Un equipo de 3 agentes quema aproximadamente de 3 a 4 veces los tokens de una sesión en solitario, y Anthropic señala hasta 15x para equipos grandes.
> La única pregunta que decide equipo frente a sub-agent: ¿los trabajadores necesitan hablar entre ellos a mitad de tarea?
> "No" cubre cerca del 90 % del trabajo paralelo: usar sub-agents.

---

## Parte 7 - Worktrees: aislamiento de archivos para agentes en paralelo

Dos agentes, un `package.json`, cero supervivientes.

Con agentes en paralelo en el mismo directorio, uno pisará el archivo a medio escribir de otro, o un `npm install` del Agente A romperá la build del Agente B.

La mensajería no arregla esto. La separación física sí.
`git worktree` da a cada agente su propio directorio de trabajo completo respaldado por el mismo repositorio.
Trabajar en paralelo, fusionar al final, dejar que git arbitre.

### El modelo de aislamiento en tres capas

| Capa | Aísla | Pregunta que responde |
|---|---|---|
| Sub-Agent | Contexto | "¿Cómo mantengo el ruido de la exploración fuera de la conversación principal?" |
| Agent Teams | Roles | "¿Cómo se coordinan agentes especializados?" |
| Worktree | Archivos | "¿Cómo evito que agentes paralelos destrocen el trabajo del otro?" |

Se componen según lo que el trabajo requiera:

- Los agentes solo leen código -> sub-agents a secas.
- Los agentes escriben con propiedad de archivos limpia -> la mensajería de Agent Teams basta.
- Los agentes escriben con propiedad solapada -> añadir worktrees y dejar de preocuparse.

### El detalle que pilla a todo el mundo

Los worktrees solo copian archivos versionados en git.
El `.env`, `node_modules` y la configuración local no están por defecto: el agente arranca, no encuentra las variables de entorno y falla de forma confusa.

Listar lo que debe viajar en un `.worktreeinclude`:

```
# .worktreeinclude
.env
.env.local
supabase/.temp
```

Saltarse `node_modules` y dejar que el agente ejecute la instalación.

Fijar `baseRef` a:

- `"fresh"`: última main remota, para tareas independientes.
- `"head"`: el commit actual, para construir sobre trabajo en curso.

Cablear el hook `WorktreeCreate` para lanzar automáticamente `npm install` o sembrar una base de datos local al arrancar.

### Lo que los worktrees no resuelven

- **Conflictos de merge.**
  El aislamiento los aplaza, no los elimina.
  Pero un conflicto de git revisable es mejor que una pérdida silenciosa de datos.
- **Conflictos semánticos.**
  El Agente A cambia un tipo de retorno y el Agente B escribe llamadores contra el antiguo.
  Solo la mensajería entre agentes detecta esto.
- **Disco.**
  Cada worktree es una copia casi completa.
  Cinco worktrees de un repositorio de 2 GB son más de 10 GB.

> Sin worktrees se raciona el paralelismo por miedo a las colisiones.
> Con ellos, "no toquéis los archivos del otro" deja de ser tu problema y pasa a ser de git.

---

## Parte 8 - Dynamic Workflows: orquestar 1.000 agentes

Al lanzar diez sub-agents a mano, el agente lead se ahoga.
Cada resultado aterriza en su contexto: diez resúmenes, diez volcados de herramienta.
Para el agente seis, el orquestador razona dentro de un pantano creado por él mismo.

Los Dynamic Workflows (desde la v2.1.154) lo arreglan cambiando quién sostiene el bucle.
Claude escribe un script de JavaScript que ejecuta los agentes, sostiene el bucle y guarda los resultados intermedios.
El contexto del lead solo llega a ver la respuesta final.

Mismos sub-agents, sin el impuesto de contexto, y el techo salta a 1.000 agentes por ejecución (16 concurrentes).

| | Quién decide el siguiente paso | Dónde aterrizan los resultados | Mejor para |
|---|---|---|---|
| Sub-agents | Claude, turno a turno | El contexto de Claude | Un puñado de tareas paralelas |
| Skills | Claude (sigue `SKILL.md`) | El contexto de Claude | Conocimiento reutilizable |
| Workflows | El script | El runtime del script | De decenas a cientos de agentes, repetible |

### Cómo se activa y se dispara

Activar la fila "Dynamic workflows" en `/config`.
Luego, dos formas de disparar:

- Meter la palabra clave `ultracode` en un prompt: Claude escribe un script de orquestación para esa tarea concreta.
- Ejecutar `/effort ultracode` una vez, para emparejar razonamiento máximo con orquestación automática de workflows durante toda la sesión.

La rampa de entrada sin setup es el workflow incluido `/deep-research`: despliega búsquedas web por ángulos, contrasta fuentes, vota cada afirmación y devuelve un único informe con citas.

### La recompensa es la reutilización

Cuando una ejecución hace lo que se quería, abrir `/workflows` y pulsar `s` para guardarla en:

- `.claude/workflows/`: compartido con el repositorio.
- `~/.claude/workflows/`: solo para ti.

Aparece en el autocompletado de `/` como un comando integrado, y recibe entrada a través de un global `args` para parametrizar en la llamada.

### Los límites que duelen

- Sin entrada de usuario a mitad de ejecución.
  Si hace falta aprobación entre etapas, dividir en workflows separados.
- El script no puede tocar archivos ni shell directamente.
  Los agentes sí lo hacen, y esto además es una propiedad de seguridad.
- Cada agente usa el modelo de la sesión por defecto.
  Hay que indicar a Claude que enrute las etapas baratas a un modelo menor, o se pagará a tarifa Opus trabajo de nivel grep.

### La pregunta que lo zanja

¿Esto se despliega en ancho y ejecuta la misma forma muchas veces?

- **Sí** -> workflow (una migración de 200 archivos, un benchmark de 80 combinaciones).
- **No** -> un sub-agent o un equipo.

> El token más barato es el que nunca se carga.
> Mover el bucle al código mantiene limpia la ventana del lead incluso a escala masiva.

---

## Parte 9 - Skills: el punto de extensión más flexible

Una Skill es una carpeta, no un archivo Markdown.
Esa distinción lo es todo.

Contiene un punto de entrada `SKILL.md` más scripts, documentación de referencia y plantillas.
La carpeta entera es una superficie de ingeniería de contexto.

El agente descubre las Skills al arrancar la sesión, escanea sus descripciones para decidir relevancia, y lee el `SKILL.md` completo solo cuando una Skill encaja con la tarea.

Anthropic mantiene varios cientos internamente y las catalogó en 9 tipos.
Una buena Skill encaja en exactamente uno: las Skills que intentan hacer de todo confunden al agente.

### Los nueve tipos

1. Referencia de librería o API.
2. Verificación de producto.
3. Obtención y análisis de datos.
4. Automatización de procesos de negocio.
5. Scaffolding y plantillas de código.
6. Calidad y revisión de código.
7. CI/CD y despliegue.
8. Runbooks.
9. Operaciones de infraestructura.

### Los patrones que hacen que funcionen

- **La descripción es una señal de disparo, no un resumen.**
  Cargarla con las frases exactas que los usuarios escriben.
  "Use when the user wants to..." es la frase más importante.
  Una descripción vaga de tres palabras ("Creates SEO pages") nunca se dispara; nueve frases de disparo explícitas sí.
- **Divulgación progresiva a través del sistema de archivos.**
  `SKILL.md` siempre se carga; los archivos de referencia en subcarpetas se cargan bajo demanda.
  Poner las decisiones en `SKILL.md` y el detalle en `references/`.
  Es el patrón más infrautilizado de todos.
- **La lista de gotchas es la sección de mayor valor.**
  Capturar lo que el modelo hace mal en tu dominio ("la tabla `subscriptions` es append-only: coge la versión más alta, no el `created_at` más reciente").
  Omitir todo lo que el modelo ya sabe.
- **No sobre-restringir.**
  Escribir reglas y principios, no guiones rígidos paso a paso.
  Una Skill de 15 líneas que dice "resuelve conflictos de merge preservando la intención de ambas ramas" gana a otra que guioniza cada caso límite.

### Instalación

El ecosistema ya tiene más de 1.400 Skills curadas listas para instalar, para ventas (HubSpot publica skills oficiales de CRM), ciencia de datos, marketing, DevOps y procesamiento de documentos.

Tres vías de instalación:

- Copia manual en `~/.claude/skills/` (o `~/.cursor/skills/`).
- La CLI `skills.sh`: `npx skills add owner/repo`.
- `asm`, un gestor universal para 18 agentes.

> No le digas al modelo lo que ya sabe.
> Cada línea que repite comportamiento por defecto es contexto desperdiciado y señal diluida.

---

## Parte 10 - Recortar la factura de API un 73 % sin perder calidad

El uso intensivo de Claude Code va de 250 a 500 $ al mes en API.
El usuario interno más pesado del autor llegó a 498 $/mes antes de optimizar.
Siete estrategias lo bajaron a 128 $, un 73 % de reducción, sin caída de calidad y con 0 quejas internas de calidad.

### Anatomía del coste

Cada turno tiene tres componentes:

- **Tokens de entrada**: prompt + system prompt + `CLAUDE.md` + definiciones de herramientas + turnos previos + resultados de herramientas.
  3 $/M sin caché, 0,30 $/M con caché.
- **Tokens de salida**: 15 $/M, sin descuento por caché.
- **Ejecución de herramientas**: gratis, pero produce más tokens de entrada en el turno siguiente.

Una sesión pesada antes de optimizar: ~200 K de entrada / 15 K de salida = ~0,83 $.
Después: ~0,21 $.
A 600 sesiones al mes, esa cuenta se compone con fuerza.

### Las siete estrategias, ordenadas por ROI

1. **Prompt caching.**
   Ahorra entre un 60 % y un 85 % de entrada.
   Descuento del 90 % en los tokens del prefijo cacheado, aplicado automáticamente al system prompt, las definiciones de herramientas y `CLAUDE.md`.
   Mantener el prefijo estable: editar `CLAUDE.md` a mitad de sesión revienta la caché.
   Preferir una sesión larga a muchas cortas.
2. **Enrutado de modelo.**
   Elegir por sesión con el flag `--model` (`claude --model haiku` para una pasada de leer y resumir), o cambiar a mitad de sesión con `/model`.
   Fijar el modelo diario por defecto en `.claude/settings.json` (`"model": "sonnet"`), reservar Haiku para exploración, y escalar a Opus solo cuando Sonnet se atasca visiblemente.
3. **CLAUDE.md ligero.**
4. **Poda de contexto.**
5. **Disciplina de sub-agents.**
6. **Modo batch.**
7. **Híbrido Pro + API.**

> Nota: en la captura las estrategias 3 a 7 aparecían colapsadas, con el título visible pero sin el detalle.

### El antes y el después medido en cuatro semanas

| Métrica | Antes | Después | Cambio |
|---|---|---|---|
| Tokens de entrada medios por sesión | 200 K | 80 K (50 K cacheados) | -60 % |
| Coste medio por sesión | 0,83 $ | 0,21 $ | -75 % |
| Factura mensual | 498 $ | 128 $ | -73 % |
| Quejas de calidad | línea base | sin cambios | 0 |

### Lo que NO ahorra dinero

- Pasarlo todo a Haiku: la calidad baja y se gastan 30 minutos arreglando lo que Sonnet hacía bien.
- Prompts agresivamente cortos: el modelo necesita contexto.
- Ejecutar modelos open source en local: unas 3 veces más lento, uso de herramientas más débil, pérdida neta de productividad.

> El coste es un problema de ingeniería, no un problema de precios.
> El mayor error individual es dejar que el contexto se infle: tras dos horas una sesión puede llegar a más de 150 K tokens y cada turno nuevo los paga todos.

---

## La ruta de aprendizaje

Los diez sistemas de esta guía no son paralelos: tienen dependencias.
Aprenderlos en este orden:

1. **Fundamentos + CLAUDE.md**: el agente conoce el proyecto.
2. **Patrones de prompting**: se obtiene salida específica y consciente del código.
3. **Hooks**: lo que debe pasar, pasa el 100 % de las veces.
4. **Task state**: el trabajo largo se mantiene ordenado y reanudable.
5. **Sub-agents**: trabajo paralelo sin inflar contexto.
6. **Agent Teams**: coordinación cuando los trabajadores deben hablar.
7. **Worktrees**: seguridad de archivos para escrituras en paralelo.
8. **Dynamic Workflows**: orquestación a escala.
9. **Skills**: conocimiento reutilizable que el agente carga solo.
10. **Cost controls**: la factura se corresponde con el valor.

No se puede apreciar por qué importan los sub-agents hasta haber visto una ventana de contexto llenarse y la salida degradarse.
No se pueden apreciar los worktrees hasta que dos agentes han pisado el mismo archivo.

Saltarse pasos lleva a memorizar patrones.
Seguir la ruta lleva a entender la física.

---

## Pro tips de más de 20 lecciones de Claude Code en producción

- **La mayor parte del trabajo paralelo son sub-agents, no equipos.**
  Los equipos cubren el ~10 % donde los trabajadores deben coordinarse a mitad de tarea.
  Ir a equipos por defecto es quemar 4x los tokens por una ganancia de 2 minutos.
- **Briefear a los compañeros como a nuevos fichajes.**
  No heredan la conversación del lead, solo `CLAUDE.md`, la configuración MCP y tu descripción.
  "Ocúpate del backend" produce basura. Una especificación acotada produce trabajo.
- **El agente mejora drásticamente cuando puede verificar su propio trabajo.**
  Cablear una puerta de calidad en `Stop` para que no entregue código roto.
  Este único cambio mueve el estado por defecto de "parece hecho" a "verificado hecho".
- **El trabajo en riesgo es "traducir especificación a código".**
  El trabajo que crece es "decidir qué construir y verificar que la IA lo construyó bien".
  Todo lo de esta guía es entrenamiento para el segundo.

---

## Huecos en esta captura

Los siguientes bloques aparecían colapsados en la página de origen y no se pudieron recuperar:

- El ejemplo "versión mala / versión corregida" de cada uno de los 10 patrones de prompting (Parte 2).
- La implementación de los hooks 2 a 7 (Parte 3).
- El detalle de las estrategias de coste 3 a 7 (Parte 10).
- El detalle de la opción de facturación Claude Pro/Max (Parte 1).
- Los vídeos enlazados sobre Agent Teams, Dynamic Workflows y Skills.
