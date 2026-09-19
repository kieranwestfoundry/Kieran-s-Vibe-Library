


# Kieran's Vibe Library
<p align="center">
  <img src="https://media1.tenor.com/m/O6Nt1oLMtHYAAAAC/chill-chill-out.gif" width="320" alt="chill out">
</p>

An experimental Griptape Nodes library for AI-driven VFX workflows — roughly 35 nodes
spanning diffusion, Nuke integration, USD/pipeline tooling, agents, RAG and utilities.

This is a sandbox. Nodes here are built fast to answer a specific question, and the
ones that survive contact with real work graduate to `vfx_pipeline_nodes` once they're
hardened, documented and tested. Promotion is one-directional — fixes don't get
back-ported here.

---

## Highlights

### Nuke as a compute backend (`nuke/`)
`BaseNukeNode` generates a Nuke Python script, executes it against a headless Nuke
(`-t`) via subprocess, and returns the result as a `VideoUrlArtifact`. Subclasses
declare a `nuke_node_class` and an input-pipe map — a new Nuke node costs about ten lines.

Shipping: `NukeKronos` (motion-estimated retime), `NukeBokeh` (defocus), `NukeKeyer`.

- Async execution via `AsyncResult`, so the graph stays responsive
- Full Write-node knob set exposed in a collapsed parameter group
- Accepts images, videos and sequence patterns; remote media downloaded without a
  PIL re-encode for video containers
- Unresolved macro templates produce a human-readable connection error rather than a stack trace
- Read `on_error` set to nearest frame
- Optional emit of the generated `.py` and assembled `.nk` for debugging — when a
  Nuke-side process fails, you get a script you can open and inspect

**Licensing note:** each node execution spawns a Nuke process and holds a licence for
its lifetime. Several of these in one graph, or one graph on a farm, is a real
contention model. Batching operations into a single session is the obvious next step.


### `StringMultiplexer` (variable switch)

Routes one of five data inputs to the output by matching an incoming string against
per-branch keys. Branch selection happens in `initialize_spotlight` / `_branch_spotlight`,
so **only the winning branch is ever evaluated** — the other four never execute. In a
graph where each branch is a diffusion pass or a Nuke process, that's the difference
between a switch and a real conditional. (sort of devised to work with GSVs)

> [!NOTE]
> Exact string comparison, first match wins. No match outputs `None` silently — check
> your keys if a downstream node receives nothing. 

### `CryptomatteCompiler`
Compiles binary mask sequences (e.g. SAM2 output) into a Cryptomatte EXR sequence:
MurmurHash3 per-item IDs, uint32 bit-cast to float32, JSON manifest plus
`cryptomatte/0/*` metadata written via the OpenEXR 3.3+ API, ID in R and coverage in G.

The point: generative segmentation masks are a toy until they arrive in comp as
something a compositor can pick. This is the bridge.

*Known gaps — see Limitations.*

### AOV-driven multi-pass diffusion

> [!IMPORTANT]
> **Use Foundry's Modular Diffusion library instead.** This predates it and is kept
> here as a reference implementation only — no fixes, no support. If you want
> AOV-driven generation in a real graph made by the biggest brain devs in our industry, go there.

Composable config nodes rather than one monolith:
`BaseModelConfig`, `LoraConfig`, `ControlNetConfig`, `MultiPassRenderNode`, plus
`ConstructControlNet`, `SDControlNetBuilder` and the `InjectSDControlNet` middleware node.

`ControlNetConfig` binds a ControlNet model to a **specific AOV pass**, so generation is
driven by render data — depth, normals, position — instead of prompt alone.

### `PythonScriptRunnerNode`

> [!WARNING]
> **Proof of concept, not production ready.** Dynamically generated parameters
> periodically disappear from the node UI — the underlying execution works, the
> parameter rebuild on the canvas doesn't hold. Built to prove the approach is viable;
> reload the workflow if inputs vanish.
> Also this can be a security nightmare, allowing of execution of arbitrary code. Again good vibes only. 

Parses a script's `argparse` definitions and generates node parameters from them
dynamically. Any existing CLI tool becomes a Griptape node with no wrapper work.

### `UsdModelInspectorToolNode` / `UsdNukeInspector`

> [!NOTE]
> **Requires a local Nuke install.** Point `NUKE_BIN_PATH` at your Nuke binary via
> **Settings → Secrets** in Griptape Nodes. Falls back to `Nuke` on `PATH` if unset.

Gives an agent the ability to render and visually inspect a USD model through Nuke's
headless USD renderer, returning model information and metadata in response to a query.

---

## Node index

| Category | Nodes |
|---|---|
| Nuke | `NukeKronos`, `NukeBokeh`, `NukeKeyer` (via `BaseNukeNode`) |
| Diffusion | `BaseModelConfig`, `LoraConfig`, `ControlNetConfig`, `MultiPassRenderNode`, `ConstructControlNet`, `SDControlNetBuilder`, `InjectSDControlNet` |
| TD Pipeline Tools | `CryptomatteCompiler`, `PythonScriptRunnerNode`, `UsdModelInspectorToolNode`, `UsdNukeInspector` |
| Plate Prep | `NormalCrafter` (normal estimation), EXR→PNG converter with 2K rescale |
| Video | Frame extraction, frame grid |
| Text / RAG | Griptape Cloud Vector Store KB tool, local ChromaDB RAG, text cleaner (PDF/DOCX/TXT/MD), LangChain recursive splitter, token estimator, token budget |
| Agents | Ollama prompt/agent, local Whisper transcriber, file manipulator tool, archetype/identity preset |
| Utility | Variable switch, multi-output, UUID generator, JSON builders |

---

## Requirements

- Griptape Nodes engine `>= X.XX.X` (library schema `0.6.0`)
- Python 3.12
- Nuke 17.0+ (for the `Nuke` and USD-inspector categories only) — the rest of the library
  works without it
  > [!NOTE]
> **Tested on macOS and Rocky Linux 9 only.** Windows is untested — subprocess
> invocation and path handling in the Nuke nodes are the most likely places it breaks.
> Reports welcome.

Heavier dependencies are declared in the library manifest and installed by the library
manager: `diffusers`, `torch`, `transformers`, `usd-core`, `OpenEXR`, `opencv`,
`chromadb`, `static-ffmpeg`, `PyMuPDF`, `langchain-text-splitters`.

Diffusion nodes want a CUDA GPU. Assume 16 GB VRAM as a floor for the multi-pass work.

## Install

1. Clone this repo.
2. In Griptape Nodes, **Settings → Libraries → Add Library** and point it at
   `griptape-nodes-library.json`.
3. Set the secrets below, then restart the engine.

## Configuration

| Secret | Required for | Notes |
|---|---|---|
| `NUKE_BIN_PATH` | Nuke nodes | Absolute path to the Nuke binary. Falls back to `Nuke` on `PATH` if unset. Set this to your own install — there is no sensible default. |
| `OPENAI_API_KEY` | Agent nodes | Optional if you're running Ollama locally |
| `GT_CLOUD_API_KEY` | Griptape Cloud KB tool | |

---

## Limitations

Honest list, because these will bite in production rather than in testing:

- **`CryptomatteCompiler`** — no NaN/Inf guard on the hash bit-cast (roughly 1 in 256
  names), `cryptomatte/0/conversion` metadata not yet written, metadata key uses `0`
  rather than the hex digest of the layer name, overlapping masks are last-write-wins
  (fine for disjoint masks, silently lossy otherwise), and `total_frames` isn't
  validated against source resolution.
- **Nuke nodes** — output is hardcoded to MOV with an index-selected codec and 24 fps.
  Lossy for motion-estimated results; an EXR sequence branch is the fix.
- **Script generation** — parameter values are interpolated into generated Nuke source.
  Fine for trusted graphs, not for untrusted input.
- **Categories** — several node `category` values aren't declared in the manifest's
  `categories` block, and `json`/`JSON` both appear.

## Conventions

- One node per file, named after the node.
- Long-running work returns `AsyncResult`.
- Nodes that can fail meaningfully extend `SuccessFailureNode` and expose status params.
- Every node gets a real description. "json make good" is not a description.

  > [!NOTE]
> Built for specific workflows as they came up. No stability guarantees, no support,
> no promise that anything stays where it is between commits.
