---
name: architecture-best-practices
description: >
  Review, design, and refactor software architecture using pragmatic engineering
  principles: high cohesion, low coupling, GRASP, DRY, SOLID (especially
  Open-Closed and Dependency Inversion), information hiding, separation of
  concerns, explicit boundaries, composition over inheritance, stable
  dependencies, and testable designs. Use when reviewing module/class/service
  responsibilities, deciding where behavior belongs, reducing architectural
  complexity, designing extension points, refactoring duplicated logic, or
  evaluating whether an abstraction improves maintainability. Prefer concrete
  architectural reasoning over pattern names. Avoid over-engineering,
  speculative abstractions, and cargo-cult design patterns.
---

# Architecture best practices

Design and review software so that responsibilities are cohesive, dependencies
are explicit and minimal, changes remain local, and abstractions correspond to
real variation.

Use principles as reasoning tools, not commandments.

The goal is not maximum abstraction. Prefer:

```text
clear responsibilities
+ local reasoning
+ controlled dependencies
+ easy change
+ testability
+ operational clarity
```

Choose the simplest design that keeps likely changes localized.

## Review priorities

Consider, in order:

1. correctness and domain invariants;
2. responsibility placement;
3. cohesion;
4. coupling and dependency direction;
5. change locality;
6. clarity of boundaries and interfaces;
7. testability;
8. extensibility where real variation exists;
9. duplication and consistency;
10. performance and operational constraints.

Do not optimize for hypothetical extensibility before understanding actual
change patterns.

# High cohesion

A module, class, service, or component should have responsibilities that belong
together.

Ask:
- Does this component represent one coherent concept?
- Do its fields and methods mostly participate in that concept?
- Would a change to one responsibility usually imply changes to the others?
- Is this the natural place a maintainer would look for this behavior?

Signs of low cohesion:

```text
UserManager
  - authenticate user
  - generate invoice PDF
  - upload files
  - calculate tax
  - send email
```

Prefer responsibility groups such as:

```text
Authenticator
InvoiceRenderer
FileStorage
TaxCalculator
NotificationService
```

Do not split coherent behavior into many tiny one-method types merely to reduce
class/module size. Cohesion is conceptual, not a line-count metric.

# Low coupling

Minimize both the number and the strength of dependencies.

Prefer dependencies on:
- small, stable interfaces;
- domain concepts;
- immutable values;
- explicit inputs.

Avoid unnecessary dependencies on:
- concrete infrastructure;
- global mutable state;
- implementation details of neighboring modules;
- deep object graphs;
- incidental knowledge.

Ask:

```text
If component A changes internally, how many unrelated components must change?
```

Also consider coupling beyond source imports:
- runtime coupling;
- schema/data coupling;
- temporal coupling;
- deployment coupling;
- protocol coupling;
- shared-state coupling.

# Change locality

Good architecture keeps a conceptual change local.

Trace a representative change:

```text
requirement
  -> domain rule
  -> responsible component
  -> boundary/interface changes
  -> infrastructure changes
  -> tests
```

If a small feature requires touching many unrelated modules, inspect whether
responsibilities or abstractions are misplaced.

# GRASP

Use GRASP as a set of responsibility-assignment heuristics. Explain the reason
for a responsibility placement rather than merely naming a pattern.

## Information Expert

Assign behavior to the component that already has the information needed to do
it.

Prefer:

```text
Order knows its line items
-> Order computes subtotal
```

rather than an external service repeatedly reaching into Order internals.

Do not force behavior into an entity if doing so introduces unrelated
infrastructure dependencies. Balance Information Expert with cohesion and
dependency direction.

## Creator

A component is a good creator when it:
- aggregates or contains the created object;
- records it;
- closely uses it;
- has the initialization data;
- owns its lifecycle.

Use factories when creation itself is complex, variable, or requires dependency
injection. Do not introduce factories for trivial construction.

## Controller

Place system-operation coordination in a component representing the use case or
application boundary.

A controller should orchestrate rather than absorb all domain logic.

Prefer:

```text
OrderController
  -> PlaceOrder use case
      -> domain objects/policies
      -> repository
      -> notification port
```

rather than a controller containing business rules, SQL, rendering, and email
logic together.

## Low Coupling / High Cohesion

Use these together when assigning responsibilities. Improving one while badly
hurting the other can produce a worse design.

## Polymorphism

When behavior varies by strategy/type, place the variation behind a stable
interface when that variation is meaningful and recurring.

Example:

```text
CompressionStrategy
  - Gzip
  - Zstd
  - Lz4
```

Do not replace a simple local `match` with a hierarchy merely because
polymorphism exists.

## Pure Fabrication

Introduce a non-domain service when it improves cohesion or reduces coupling.

Examples:
- repository;
- serializer;
- mapper;
- gateway;
- scheduler.

Do not use Pure Fabrication to justify generic `Manager`, `Utils`, or `Helper`
dumping grounds.

## Indirection

Introduce an intermediate abstraction only when it buys meaningful decoupling.

Examples:
- repository interface between domain and database;
- adapter between internal model and third-party API;
- message bus between genuinely independent participants.

Every indirection has costs: more types, navigation, configuration, runtime
paths, and debugging complexity.

## Protected Variations

Identify real volatility and protect stable code behind an appropriate boundary.

Examples:
- database backend;
- execution backend;
- serialization format;
- pricing policy;
- external provider.

Protect known or likely variation, not imaginary futures.

# DRY

DRY primarily means avoiding duplicated knowledge, not eliminating every pair of
similar lines.

Strong duplication:

```text
two modules independently encode the same business rule
```

Weak duplication:

```text
two code fragments happen to look similar
```

Before extracting common code, ask:
- Is this truly the same concept?
- Must these implementations always change together?
- Is there one source of truth?
- Will the abstraction accumulate flags and exceptions?

Avoid "wrong abstraction" APIs such as:

```text
process(kind, special_mode, flag_a, flag_b, optional_x)
```

created only to remove a few duplicated lines.

Prefer modest duplication over a misleading abstraction when concepts evolve
independently.

# Open-Closed Principle

Keep stable core behavior open to meaningful extension without requiring
repeated modification of central logic.

OCP is especially useful when:
- algorithms have multiple strategies;
- variants are expected to grow;
- third-party/custom implementations are required;
- central dispatch code is risky to modify repeatedly.

Example:

```rust
trait CostPolicy {
    fn calculate(&self, order: &Order) -> Money;
}
```

Do not interpret OCP as "never modify existing code." Directly changing simple,
stable code is often better than inventing a speculative extension framework.

Protect known axes of variation.

# SOLID

Use SOLID as heuristics, not compliance rules.

## Single Responsibility Principle

A component should have a coherent reason to change.

Do not interpret SRP as "each type/function must do exactly one tiny thing."
Separate concerns driven by independent concepts or stakeholders.

## Liskov Substitution Principle

Implementations of an abstraction must preserve the behavioral contract.

Check:
- preconditions are not unexpectedly stronger;
- postconditions are not weaker;
- invariants remain valid;
- error semantics remain compatible;
- implementations do not expose "unsupported" behavior hidden by a misleading
  interface.

## Interface Segregation Principle

Clients should depend only on capabilities they need.

Prefer focused capabilities over one broad interface whose consumers and
implementations use only subsets.

Do not fragment an API into dozens of tiny interfaces unless that separation is
actually useful.

## Dependency Inversion Principle

High-level policy should not depend directly on volatile low-level mechanisms.

Prefer:

```text
domain/use case
    -> StoragePort
    <- PostgresStorage
```

rather than domain policy importing PostgreSQL-specific details.

Keep the abstraction close to the policy that needs it when practical.

Do not wrap every stable library solely to claim DIP compliance.

# Dependency direction

Prefer dependencies pointing toward stable domain/application policy.

Typical shape:

```text
UI / CLI / HTTP
      |
      v
Application / Use cases
      |
      v
Domain
      ^
      |
Infrastructure adapters
```

Avoid leaking HTTP, SQL, cloud SDK, GUI, or serialization framework types deep
into domain logic unless those concepts genuinely belong there.

# Stable dependencies

Stable components should not unnecessarily depend on volatile components.

Wrap an external dependency when:
- its API would otherwise spread throughout the system;
- domain translation is needed;
- replacement is plausible;
- operational policy belongs at the boundary;
- testing benefits materially from isolation.

Do not wrap libraries mechanically.

# Information hiding and encapsulation

Expose what callers need, not internal representation.

Prefer:
- private fields;
- narrow public APIs;
- validated constructors;
- invariant-preserving operations;
- opaque implementation details.

Avoid APIs requiring callers to remember internal update sequences such as:

```text
set A
set B
then remember to recompute C
```

Prefer one semantic operation that preserves the invariant.

# Tell, don't ask

Where useful, tell the responsible object/component to perform a domain action
rather than extracting state and reimplementing its logic externally.

Suspicious:

```rust
if order.status() == Status::Pending && !order.items().is_empty() {
    order.set_status(Status::Confirmed);
}
```

Prefer something like:

```rust
order.confirm()?;
```

Do not hide legitimate read/query operations just to follow this heuristic.

# Law of Demeter

Deep navigation such as:

```text
a.get_b().get_c().get_d().do_x()
```

can indicate excessive knowledge of internal topology.

Reduce unnecessary structural knowledge, but do not add forwarding methods
mechanically.

# Composition over inheritance

Prefer composition when independent behaviors can be assembled.

Use subtyping only where substitutability is real.

In Rust, prefer traits, enums, generics, and composition rather than emulating
inheritance-heavy object models.

# Explicit dependencies

Avoid hidden dependencies:
- service locators;
- mutable globals;
- singleton registries;
- ambient thread-local state;
- side-effectful global initialization.

Prefer constructor/function parameters or focused context objects.

Do not replace many explicit dependencies with one enormous `Context` object
passed everywhere; that merely hides coupling.

# Dependency injection

Use dependency injection to make required collaborators explicit.

Prefer constructor injection for required dependencies.

Use runtime trait objects only when runtime substitution is needed. Prefer
static dispatch/generics when implementations are known at compile time and the
type complexity remains reasonable.

Do not introduce a DI container unless project complexity warrants it.

# Ports and adapters

Use ports/adapters when external systems are volatile or domain logic should
remain infrastructure-independent.

Ports may represent:
- repository;
- clock;
- filesystem;
- model executor;
- message publisher;
- payment gateway.

Adapters may implement those ports with PostgreSQL, a filesystem, CUDA, Kafka,
etc.

Translate external representations at the boundary instead of spreading
third-party DTOs throughout the core.

# Separation of concerns

Separate concerns that evolve for different reasons:
- business rules;
- persistence;
- transport;
- presentation;
- authorization;
- observability;
- retry policy;
- caching;
- serialization.

Do not add layers merely for architectural symmetry. A five-line feature does
not need seven layers.

# Domain logic vs orchestration

Keep domain decisions separate from orchestration where practical.

Application/use-case layer:

```text
load entity
invoke domain behavior
persist result
publish event
```

Domain layer:

```text
validate invariant
calculate decision
perform state transition
```

Infrastructure layer:

```text
SQL
HTTP
filesystem
GPU/backend calls
```

# Command/query separation

Where useful, separate operations that mutate state from operations that only
return information.

Do not turn this simple principle into full CQRS unless the system benefits from
that architecture.

# Make illegal states difficult to represent

Use types and enums to encode important invariants.

Prefer:

```rust
enum JobState {
    Pending,
    Running { worker: WorkerId },
    Finished { result: ResultId },
}
```

over unrelated booleans such as `is_running`, `is_finished`, and `has_result`
that permit contradictory combinations.

Use semantic newtypes when they materially prevent mistakes or centralize
validation. Do not wrap every primitive by default.

# Abstraction quality

A good abstraction:
- has a clear purpose;
- hides volatile details;
- makes common usage straightforward;
- prevents or discourages invalid usage;
- has few surprising exceptions.

Warning signs:
- many mode flags;
- generic names;
- frequent downcasts;
- callers checking concrete implementation types;
- "except when" rules;
- an interface much larger than most users need.

If callers routinely bypass an abstraction, reconsider it.

# Avoid speculative generality

Do not introduce plugin systems, factories, strategy layers, generic
repositories, event buses, or configuration-driven behavior without a real
need.

The rule of three is a useful heuristic: wait for enough concrete cases to
understand the true abstraction.

Earlier abstraction is justified when extensibility, compatibility,
multi-backend support, or third-party implementations are already requirements.

# God objects and vague containers

Names such as these deserve scrutiny:

```text
Manager
Helper
Utils
Common
Base
CoreService
Context
```

Do not reject them by name alone. Ask:
- What concept does this type represent?
- Can its responsibility be described in one sentence?
- Are unrelated change reasons accumulating here?

# Side effects

Keep side effects at explicit boundaries when practical.

Prefer pure domain calculation where it improves reasoning and testing, with I/O
and infrastructure orchestration outside.

Do not contort inherently stateful workflows into artificial functional designs
if clarity suffers.

# State ownership

For every important mutable state, ask:
- Who owns it?
- Who may mutate it?
- Who may observe it?
- How are updates synchronized?
- What is the source of truth?

Multiple competing sources of truth are a major architectural risk.

# Caching

Treat cache as derived state unless explicitly authoritative.

Make clear:
- source of truth;
- cache key;
- invalidation policy;
- consistency model;
- lifecycle;
- ownership.

Do not leak cache mechanics into domain APIs unnecessarily.

# Events

Use events to decouple genuinely independent reactions.

Good example:

```text
OrderPlaced
  -> send confirmation
  -> analytics
  -> fulfillment scheduling
```

Avoid event chains that make control flow impossible to trace. Prefer a direct
call when the relationship is simple, synchronous, and conceptually direct.

# Error boundaries

Errors should reflect their abstraction boundary.

A low-level SQL uniqueness failure may need translation to a domain/application
error such as `UserAlreadyExists` when callers care about that meaning.

Preserve useful source information for diagnostics while avoiding leakage of
irrelevant infrastructure details through every layer.

# Testability

Good architecture usually improves testability naturally.

Prefer tests against:
- domain behavior;
- public module interfaces;
- meaningful external boundaries.

Do not make private helpers public solely for tests.

Do not introduce an interface merely to mock a trivial deterministic function.

# Observability

Important orchestration and infrastructure boundaries should make it possible
to understand:
- what operation is running;
- what dependency is being called;
- latency;
- failure;
- correlation/request identity where relevant.

Avoid scattering logging into pure domain logic unless the domain event itself
is important.

# Concurrency and architecture

Shared-state concurrency increases coupling.

Prefer clear ownership, message passing, immutable snapshots, and explicit
synchronization boundaries where appropriate.

If many unrelated modules must coordinate locks over the same state, the
architecture may be exposing internal ownership too broadly.

# Rust-specific guidance

Use Rust modules/crates to enforce architectural boundaries.

Prefer:
- small public surfaces;
- `pub(crate)` when external visibility is unnecessary;
- private implementation modules;
- domain types separated from adapters;
- traits at meaningful boundaries.

Avoid a giant prelude that exports everything.

Do not create crate boundaries solely to create more folders; use them when
independent compilation, versioning, reuse, or architectural isolation has real
value.

## Trait design

Prefer small semantic traits.

Create a trait when there is:
- real polymorphism;
- a meaningful architectural boundary;
- multiple implementations;
- third-party extensibility;
- useful test substitution.

Do not create a trait for every concrete struct.

## Enums vs traits

Prefer enums when:
- the set of variants is closed;
- exhaustive handling is useful;
- variants are controlled by the core application.

Prefer traits/trait objects when:
- implementations are open-ended;
- third parties may add implementations;
- runtime substitution is required.

This is a practical Open-Closed tradeoff.

# YAGNI

Do not build infrastructure for imagined requirements.

Prefer:

```text
current requirements
+ credible near-term variation
+ low-cost extension path
```

But do not use YAGNI to ignore known future constraints.

# KISS

Prefer the simplest design that satisfies requirements and preserves important
invariants.

"Simplest" does not mean fewest types or shortest code. It means lowest total
conceptual complexity.

# Principle conflicts

When principles conflict, reason from concrete consequences.

Examples:

DRY vs cohesion:
- sharing code can couple unrelated domains;
- duplication may be preferable.

OCP vs KISS:
- an extension point may be speculative;
- modifying simple code directly may be better.

DIP vs simplicity:
- an abstraction around a stable dependency may add no value.

SRP vs discoverability:
- splitting too aggressively can scatter one coherent operation.

Never treat a principle as automatically overriding the others.

# Architecture review procedure

When reviewing a codebase or design, follow this sequence.

## 1. Identify responsibilities

For each important component, describe its responsibility in one sentence.
Repeated unrelated "and" clauses often indicate low cohesion.

## 2. Map dependencies

Sketch the dependency graph and identify:
- high fan-out;
- high fan-in;
- cycles;
- volatile dependencies;
- globals/hidden dependencies.

## 3. Identify sources of truth

For important state, determine its authoritative owner and flag duplicated
mutable state that must remain synchronized.

## 4. Identify change axes

Ask what actually varies:
- algorithm;
- storage;
- transport;
- backend;
- policy;
- data format.

Check whether stable code is unnecessarily exposed to these changes.

## 5. Evaluate cohesion and coupling

Look for mixed concerns, implementation leakage, shared mutable state, broad
interfaces, deep navigation, and temporal coupling.

## 6. Evaluate duplication

Distinguish duplicated knowledge from coincidentally similar code.

## 7. Evaluate abstractions

For each abstraction ask:
- What variation does it protect?
- Who benefits?
- What complexity does it add?
- Is there evidence the variation exists?

## 8. Evaluate extension mechanisms

Check whether polymorphism/OCP improves change locality or merely complicates
simple modifications.

## 9. Evaluate operational behavior

Architecture should also support clear:
- error propagation;
- retries;
- cancellation;
- shutdown;
- observability;
- resource limits.

# Suspicious patterns

Actively inspect:

```text
FooManager / FooUtils / GlobalContext / ServiceLocator
```

```text
A -> B -> C -> D -> internal field
```

```text
same large switch/match repeated in multiple modules
```

```text
boolean flags selecting fundamentally different behaviors
```

```text
one service importing DB + HTTP + UI + domain layers
```

```text
domain code containing framework/database-specific objects
```

```text
every module depending on one giant context object
```

```text
two independent copies of the same business rule
```

```text
interface with many methods most implementations cannot support
```

```text
generic abstraction filled with special cases
```

```text
event-driven flow where a direct call would be clearer
```

```text
shared mutable state where one owner could suffice
```

# Refactoring guidance

Prefer incremental architectural changes:

```text
identify responsibility
-> isolate behavior
-> clarify boundary
-> redirect callers
-> remove obsolete coupling
```

Avoid full rewrites unless explicitly justified.

When introducing an abstraction:
1. identify the real variation;
2. define the smallest useful interface;
3. move volatile details behind it;
4. keep policy outside;
5. verify callers become simpler.

When removing an abstraction:
1. identify indirection with no real benefit;
2. inline or merge responsibility;
3. preserve meaningful boundaries;
4. remove obsolete factory/configuration machinery.

# Pattern usage

Patterns are vocabulary, not goals.

Do not say:

```text
Use Strategy because Strategy is a good pattern.
```

Instead explain:

```text
The pricing algorithm varies independently from Order. A pricing-policy
interface keeps Order cohesive and prevents repeated conditionals in callers.
```

Reason first; name the pattern second.

# Reporting findings

For each significant architectural issue report:

1. **Issue**
2. **Evidence**
3. **Principle involved**
4. **Practical consequence**
5. **Recommended change**
6. **Tradeoff**

Example:

```text
Issue:
JobManager owns scheduling, CUDA execution, retry policy, persistence, and
metrics.

Evidence:
Changes to retry policy and GPU execution modify the same type, and tests must
mock unrelated persistence behavior.

Principles:
Low cohesion, high coupling, SRP.

Consequence:
Independent changes interact and the component is difficult to test.

Recommendation:
Keep JobCoordinator as orchestration. Extract execution policy and persistence
behind focused boundaries.

Tradeoff:
Adds two explicit dependencies and slightly more wiring, but isolates two
independent change axes.
```

Do not merely list violated principles. Explain the actual maintenance,
correctness, testing, or operational consequence.

# Full review output

When asked for a full architecture review, organize the response as:

```text
1. Current architecture
2. Main responsibility boundaries
3. Coupling/cohesion findings
4. Change-locality problems
5. Abstraction/DRY findings
6. Extension/OCP findings
7. Dependency-direction findings
8. Recommended incremental refactoring
9. Tradeoffs and things not worth changing
```

Prioritize high-impact findings and avoid flooding the review with minor style
comments.

# Final rule

Optimize architecture for humans changing the system.

A good design should make it reasonably obvious:
- where behavior belongs;
- which component owns state;
- which dependency performs an external effect;
- what must change for a new requirement;
- what does not need to change.

Prefer understandable systems over theoretically perfect ones.
