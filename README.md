# Kieran's Vibe Library

<p align="center">
<img src="https://media1.tenor.com/m/O6Nt1oLMtHYAAAAC/chill-chill-out.gif" width="320" alt="chill out">
</p>

Around 35 Griptape Nodes for AI-driven VFX work. Diffusion, Nuke integration, USD and
pipeline tools, agents, RAG, and a pile of utilities.

Most of these were built to answer one specific question in one specific workflow, then
kept because they turned out useful. Nodes that survive real work will eventually be hardened and moved
to `vfx_pipeline_nodes`.

No support, no stability guarantees!

---

## Highlights

### Nuke as a compute backend (`nuke/`)

`BaseNukeNode` writes a Nuke Python script, runs it against a headless Nuke (`-t`) via
subprocess, and hands the result back as a `VideoUrlArtifact`. Subclasses just declare a
`nuke_node_class` and an input-pipe map, so a new Nuke node is about ten lines.

Currently: `NukeKronos` (retime), `NukeBokeh` (defocus), `NukeKeyer`.

- Runs async via `AsyncResult`, so the graph doesn't lock up
- Full Write knob set in a collapsed parameter group
- Takes images, videos and sequence patterns. Remote media is downloaded without a PIL
  re-encode for video containers
- Unresolved macro templates give you a readable connection error instead of a traceback,
  ie it tells you that you've wired the file config parameter in rather than the artifact
- Read `on_error` set to nearest frame
- Can emit the generated `.py` and the assembled `.nk`. If the Nuke side falls over you
  get a script you can open and debug yourself

**Licensing:** every execution spawns a Nuke process and holds a licence for as long as
it runs. A few of these in one graph, or anything on a farm, and that's a real cost.
Batching operations into one session is the obvious fix, not done yet.

### `StringMultiplexer` (variable switch)

Routes 1 of 5 inputs to the output by matching an incoming string against per-branch keys.
The node I use most. Sort of devised to work with GSVs.

Branching happens in `initialize_spotlight` / `_branch_spotlight`, so only the winning
branch is ever evaluated. The other four don't run at all. Worth knowing if your branches
are diffusion passes or Nuke processes.

> [!NOTE]
> Exact string comparison, first match wins, no strip or casefold. No match outputs `None`
> silently, so if a downstream node gets nothing, check your keys. LLM output with a
> trailing space will not match.

### `CryptomatteCompiler`

Turns binary mask sequences (SAM2/3/3.1 output, for example) into a Cryptomatte EXR sequence.
MurmurHash3 per-item IDs, uint32 bit-cast to float32 via `struct`, JSON manifest and
`cryptomatte/0/*` metadata written into the header with the OpenEXR 3.3+ API, ID in R and
coverage in G.

Segmentation masks out of a model aren't much use on their own. This is the bit that gets
them into comp as something a compositor can actually pick.

Known gaps below, read them before you trust it on a real shot.

### AOV-driven multi-pass diffusion

> [!IMPORTANT]
> 🚨 **USE FOUNDRY'S OFFICIAL MODULAR DIFFUSION LIBRARY.** 🚨 Not this. This was me
> working the idea out before there was a supported one. Kept for reference, unmaintained.

Config nodes rather than one big one: `BaseModelConfig`, `LoraConfig`, `ControlNetConfig`,
`MultiPassRenderNode`, plus `ConstructControlNet`, `SDControlNetBuilder` and the
`InjectSDControlNet` middleware node.

`ControlNetConfig` ties a ControlNet model to a specific AOV pass, so generation is driven
by render data (depth, normals, position) rather than prompt alone.

### `PythonScriptRunnerNode`

> [!WARNING]
> **Proof of concept more than anything.** Dynamically generated parameters periodically
> disappear from the node UI. Execution works, the parameter rebuild on the canvas doesn't
> hold. Reload the workflow if your inputs vanish.

Reads a script's `argparse` definitions and builds node parameters from them. Every
facility has a drawer full of argparse scripts, this makes them nodes without writing a
wrapper for each one.

### `UsdModelInspectorToolNode` / `UsdNukeInspector`

Lets an agent render and visually inspect a USD model through Nuke's headless USD
renderer, and return model info and metadata off the back of a query.

> [!NOTE]
> Needs a local Nuke install. See [Configuration](#configuration).

---

## Node index

| Category | Nodes |
|---|---|
| Nuke | `NukeKronos`, `NukeBokeh`, `NukeKeyer` (via `BaseNukeNode`) |
| Diffusion | `BaseModelConfig`, `LoraConfig`, `ControlNetConfig`, `MultiPassRenderNode`, `ConstructControlNet`, `SDControlNetBuilder`, `InjectSDControlNet` |
| TD Pipeline Tools | `CryptomatteCompiler`, `PythonScriptRunnerNode`, `UsdModelInspectorToolNode`, `UsdNukeInspector` |
| Plate Prep | `NormalCrafter` (normal estimation), EXR to PNG with 2K rescale |
| Video | Frame extraction, frame grid |
| Text / RAG | Griptape Cloud Vector Store KB tool, local ChromaDB RAG, text cleaner (PDF/DOCX/TXT/MD), LangChain recursive splitter, token estimator, token budget |
| Agents | Ollama prompt/agent, local Whisper transcriber, file manipulator tool, archetype/identity preset |
| Utility | `StringMultiplexer`, multi-output, UUID generator, JSON builders |

---

## Requirements

> [!NOTE]
> Tested on macOS and Rocky Linux 9. Windows is untested. Subprocess invocation and path
> handling in the Nuke nodes are where I'd expect it to break first.

- Griptape Nodes engine `>= X.XX.X`, library schema `0.6.0`
- Python 3.12
- Nuke, for the Nuke and USD inspector nodes only. Everything else runs without it
- A CUDA GPU for the diffusion nodes. 16GB VRAM is about the floor for the multi-pass stuff

Heavier dependencies are declared in the manifest and installed by the library manager:
`diffusers`, `torch`, `transformers`, `usd-core`, `OpenEXR`, `opencv`, `chromadb`,
`static-ffmpeg`, `PyMuPDF`, `langchain-text-splitters`.

## Install

1. Clone the repo.
2. Griptape Nodes, **Settings → Libraries → Add Library**, point it at
   `griptape-nodes-library.json`.
3. Set the secrets below and restart the engine.

## Configuration

| Secret | Needed for | Notes |
|---|---|---|
| `NUKE_BIN_PATH` | Nuke and USD inspector nodes | Absolute path to your Nuke binary. Falls back to `Nuke` on `PATH` if unset. Set it to your own install, there's no sensible default |
| `OPENAI_API_KEY` | Agent nodes | Not needed if you're on Ollama locally |
| `GT_CLOUD_API_KEY` | Griptape Cloud KB tool | |

---

## Known issues

The ones that'll actually catch you out, rather than a blanket disclaimer:

- **`CryptomatteCompiler`** — no NaN/Inf guard on the hash bit-cast, so roughly 1 name in
  256 will break matte picking. Fine in testing, fails on a shot with fifty objects.
  `cryptomatte/0/conversion` isn't written yet, and the metadata key is `0` rather than a
  hex digest of the layer name. Overlapping masks are last write wins, fine for disjoint
  SAM2/3/3.1 masks and silently lossy otherwise. `total_frames` isn't checked against source
  resolution, so a wrong-res mask throws a numpy broadcast error instead of anything useful
- **Nuke nodes** — output is hardcoded to MOV, index-selected codec, 24fps. That's lossy
  for motion-estimated results. Needs an EXR sequence branch
- **Script generation** — parameter values are interpolated into the generated Nuke source.
  Fine for trusted graphs, not for untrusted input
- **`StringMultiplexer`** — fixed at 5 branches, and the default keys (`key_0` to `key_4`)
  can match by accident on an unconfigured node
- **Categories** — some node `category` values aren't declared in the manifest's
  `categories` block, and `json`/`JSON` both appear

## Conventions

- One node per file, named after the node
- Anything long-running returns `AsyncResult`
- Nodes that can fail meaningfully extend `SuccessFailureNode` and expose status params
- Every node gets a real description. "json make good" is not a description
