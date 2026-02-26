# llama.cpp Study Plan — From C++ Beginner to Contributor

> A phased, progressive guide to understanding the entire llama.cpp codebase.
> Each phase builds on the previous one. Take your time — don't rush.

---

## Table of Contents

- [Overview: What is llama.cpp?](#overview-what-is-llamacpp)
- [Phase 0: C++ Fundamentals You Need First](#phase-0-c-fundamentals-you-need-first)
- [Phase 1: Build, Run, and Explore](#phase-1-build-run-and-explore)
- [Phase 2: The Simplest Example — Mental Model](#phase-2-the-simplest-example--mental-model)
- [Phase 3: The Public API — include/llama.h](#phase-3-the-public-api--includellamah)
- [Phase 4: GGML — The Tensor Engine Under the Hood](#phase-4-ggml--the-tensor-engine-under-the-hood)
- [Phase 5: GGUF — The Model File Format](#phase-5-gguf--the-model-file-format)
- [Phase 6: Core llama.cpp Internals (src/)](#phase-6-core-llamacpp-internals-src)
- [Phase 7: The Common Library (common/)](#phase-7-the-common-library-common)
- [Phase 8: Tools & Applications](#phase-8-tools--applications)
- [Phase 9: Model Conversion Pipeline (Python)](#phase-9-model-conversion-pipeline-python)
- [Phase 10: Testing & CI](#phase-10-testing--ci)
- [Phase 11: Advanced Topics](#phase-11-advanced-topics)
- [Appendix A: Directory Map](#appendix-a-directory-map)
- [Appendix B: Key Concepts Glossary](#appendix-b-key-concepts-glossary)
- [Appendix C: Recommended Reading Order for Source Files](#appendix-c-recommended-reading-order-for-source-files)

---

## Overview: What is llama.cpp?

llama.cpp is a **high-performance C/C++ inference engine** for Large Language Models (LLMs). It lets you run LLMs (like LLaMA, Mistral, Qwen, Gemma, etc.) on consumer hardware — CPUs, GPUs, and even phones — without needing Python or PyTorch at runtime.

**The architecture has 3 layers (bottom-up):**

```
┌─────────────────────────────────────────────────────┐
│  Applications / Tools                               │
│  (llama-cli, llama-server, examples/)               │
├─────────────────────────────────────────────────────┤
│  llama library  (src/ + include/llama.h)            │
│  Model loading, tokenization, inference logic       │
├─────────────────────────────────────────────────────┤
│  ggml library   (ggml/)                             │
│  Tensor operations, memory management, backends     │
│  (CPU, CUDA, Metal, Vulkan, OpenCL, etc.)           │
└─────────────────────────────────────────────────────┘
```

**The data flow for inference is:**

```
Model File (.gguf)
   │
   ▼
Load model weights into memory (llama_model_load_from_file)
   │
   ▼
Create inference context (llama_init_from_model)
   │
   ▼
Tokenize input text → token IDs (llama_tokenize)
   │
   ▼
Feed tokens into model → compute logits (llama_decode)
   │
   ▼
Sample next token from logits (llama_sampler_sample)
   │
   ▼
Convert token ID back to text (llama_token_to_piece)
   │
   ▼
Repeat until done (autoregressive loop)
```

---

## Phase 0: C++ Fundamentals You Need First

**Goal:** Learn enough C++ to read the codebase comfortably.

You do NOT need to be a C++ expert. llama.cpp deliberately uses **simple, old-school C++** — no heavy template metaprogramming, minimal STL usage, and C-style APIs.

### Must-Know Concepts

| Concept | Why it matters here |
|---------|-------------------|
| **Pointers and references** (`int * p`, `int & r`) | The entire API passes pointers around. Every `llama_context *` is a pointer. |
| **Structs** (`struct foo { ... }`) | Core data types like `llama_model`, `llama_context`, `llama_batch` are structs. |
| **Enums** (`enum llama_vocab_type { ... }`) | Used extensively for token types, model architectures, rope types, etc. |
| **`std::vector<T>`** | The main dynamic array container used throughout. |
| **`std::string`** | Used for text handling. |
| **`std::map` / `std::unordered_map`** | Used in vocab lookup and model metadata. |
| **Header files (`.h`) vs source files (`.cpp`)** | Headers declare interfaces; `.cpp` files contain implementations. |
| **`#include` and `#pragma once`** | How C++ organizes code across multiple files. |
| **`static` functions** | Functions local to a single file (not visible outside). |
| **Function pointers / callbacks** | Used in ggml for operation dispatch and logging. |
| **`extern "C"`** | Makes C++ functions callable from C. The public API uses this. |
| **`const` correctness** | `const char *` means "pointer to data you can't modify". Used everywhere. |
| **Casting** (`static_cast`, `reinterpret_cast`, C-style casts) | Used when converting between numeric types or void pointers. |
| **Memory management** (`new`/`delete`, `malloc`/`free`) | ggml manages its own memory pools. Understanding manual memory helps. |
| **`size_t`, `int32_t`, `uint8_t`** | Fixed-width integer types used in the public API. |
| **Basic file I/O** (`fopen`, `fread`, `mmap`) | Model loading reads large binary files. |
| **`#define` macros** | Used for constants, platform detection, export markers. |

### Concepts You Can Learn as You Go

- Templates (used lightly in some utilities)
- `std::unique_ptr` / `std::shared_ptr` (used in some newer code)
- RAII (Resource Acquisition Is Initialization)
- Virtual functions / inheritance (used in memory backends)
- Lambda functions (used occasionally)

### Recommended Resources

1. [learncpp.com](https://www.learncpp.com/) — Best free tutorial (chapters 1-12 cover everything above)
2. [cppreference.com](https://en.cppreference.com/) — Reference when you see unfamiliar syntax
3. The C++ code in `examples/simple/simple.cpp` — Read it alongside learning

**Estimated time: 1-2 weeks** (if you code daily)

---

## Phase 1: Build, Run, and Explore

**Goal:** Get the project building and running on your machine.

### Step 1.1 — Clone and Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Basic CPU-only build
cmake -B build
cmake --build build --config Release -j $(nproc)
```

The binaries land in `build/bin/`. Read [docs/build.md](docs/build.md) for GPU / platform-specific instructions.

### Step 1.2 — Get a Small Model

Download a small GGUF model to experiment with. A good choice is a quantized 1B-3B parameter model. You can find models on [Hugging Face](https://huggingface.co/models?sort=trending&search=gguf).

### Step 1.3 — Run Examples

```bash
# Simple text generation
./build/bin/llama-simple -m your-model.gguf -n 64 "Once upon a time"

# Interactive chat
./build/bin/llama-cli -m your-model.gguf -cnv

# Start the HTTP server
./build/bin/llama-server -m your-model.gguf
```

### Step 1.4 — Understand the Build System

Read these files to understand how the project is structured:

| File | What it does |
|------|-------------|
| [CMakeLists.txt](CMakeLists.txt) | Root build file. Includes ggml, src, common, examples, tools, tests. |
| [ggml/CMakeLists.txt](ggml/CMakeLists.txt) | Builds the ggml tensor library + backends (CUDA, Metal, etc.) |
| [src/CMakeLists.txt](src/CMakeLists.txt) | Builds the core `llama` library |
| [common/CMakeLists.txt](common/CMakeLists.txt) | Builds shared utility code used by tools/examples |
| [tools/CMakeLists.txt](tools/CMakeLists.txt) | Builds main tools (CLI, server, quantize, etc.) |
| [examples/CMakeLists.txt](examples/CMakeLists.txt) | Builds example programs |

**Key insight:** The root `CMakeLists.txt` orchestrates everything. It uses `add_subdirectory()` to pull in each component.

### What to Focus On

- Don't read every CMake line. Just understand: **ggml → llama lib → common → tools/examples**.
- Notice how each tool/example links against `llama` and/or `common`.

**Estimated time: 1-2 days**

---

## Phase 2: The Simplest Example — Mental Model

**Goal:** Understand the complete lifecycle of LLM inference by reading one small program.

### Read: `examples/simple/simple.cpp`

This is a ~220 line self-contained program. It shows the **entire inference pipeline** with no abstractions hiding the details.

**Read it in this order, understanding each block:**

#### Block 1: Initialization (lines 1-88)
```
Parse command-line arguments → model path, prompt, GPU layers, prediction count
```
- **C++ concepts:** `argc`/`argv`, `strcmp`, `std::stoi`, `std::string`

#### Block 2: Load Backend + Model (lines 88-100)
```cpp
ggml_backend_load_all();                              // Load compute backends (CPU, CUDA, etc.)
llama_model * model = llama_model_load_from_file(...); // Load model weights from .gguf file
```
- **Key concept:** The model is loaded once and can serve multiple inference requests.

#### Block 3: Tokenization (lines 100-115)
```cpp
llama_tokenize(vocab, prompt, ...)  // Convert text → token IDs
```
- **Key concept:** LLMs don't work with text directly. Text → integers (tokens) → model → integers → text.

#### Block 4: Create Context (lines 117-130)
```cpp
llama_context * ctx = llama_init_from_model(model, ctx_params);
```
- **Key concept:** A context holds the inference state (KV cache, compute buffers). You can have multiple contexts per model.

#### Block 5: Create Sampler (lines 132-138)
```cpp
llama_sampler * smpl = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
```
- **Key concept:** The sampler decides which token to pick from the model's output probabilities.

#### Block 6: The Inference Loop (lines 170-209)
```cpp
for (...) {
    llama_decode(ctx, batch);                          // Run the model forward pass
    new_token_id = llama_sampler_sample(smpl, ctx, -1); // Pick next token
    llama_token_to_piece(vocab, new_token_id, ...)      // Convert token → text
    // Prepare next batch with the new token
    batch = llama_batch_get_one(&new_token_id, 1);
}
```
- **Key concept:** This is the **autoregressive loop** — feed tokens in, get one token out, repeat.

#### Block 7: Cleanup (lines 213-220)
```cpp
llama_sampler_free(smpl);
llama_free(ctx);
llama_model_free(model);
```
- **Key concept:** Manual resource management — allocate in order, free in reverse order.

### The Core Mental Model

```
┌─────────────────────────────────────────────────────┐
│                   simple.cpp                        │
│                                                     │
│  1. Load backends        ggml_backend_load_all()    │
│  2. Load model           llama_model_load_from_file │
│  3. Tokenize prompt      llama_tokenize             │
│  4. Create context       llama_init_from_model      │
│  5. Create sampler       llama_sampler_chain_init    │
│  6. LOOP:                                           │
│     a. Decode batch      llama_decode               │
│     b. Sample token      llama_sampler_sample       │
│     c. Print token       llama_token_to_piece       │
│     d. Prepare next      llama_batch_get_one        │
│  7. Free everything                                 │
└─────────────────────────────────────────────────────┘
```

### Also Read

- `examples/simple-chat/simple-chat.cpp` — Same idea but with chat templates (multi-turn conversations).

**Estimated time: 1-2 days**

---

## Phase 3: The Public API — include/llama.h

**Goal:** Understand the full public interface of the llama library.

### Read: `include/llama.h` (~1561 lines)

This is the **single most important file** in the project. Every tool and example calls functions declared here.

### How to Read It

Don't try to memorize everything. Instead, read it in sections:

#### Section 1: Types and Enums (lines 1-300)
These define the vocabulary of the library:

| Type | Purpose |
|------|---------|
| `llama_model` | Opaque struct — holds model weights and metadata |
| `llama_context` | Opaque struct — holds inference state (KV cache, buffers) |
| `llama_vocab` | Opaque struct — holds the tokenizer |
| `llama_sampler` | Opaque struct — holds sampling strategy |
| `llama_batch` | Struct describing a batch of tokens to process |
| `llama_token` | `int32_t` — a single token ID |
| `llama_pos` | `int32_t` — token position in the sequence |
| `llama_seq_id` | `int32_t` — sequence ID (for parallel decoding) |

Key enums:
- `llama_vocab_type` — tokenizer type (BPE, SentencePiece, etc.)
- `llama_rope_type` — rotary position embedding type
- `llama_token_type` — special vs normal tokens
- `llama_split_mode` — how model is split across GPUs

#### Section 2: Parameter Structs (lines 300-500)
- `llama_model_params` — controls model loading (GPU layers, split mode, etc.)
- `llama_context_params` — controls context creation (context size, batch size, threads, etc.)
- `llama_sampler_chain_params` — controls sampler behavior

#### Section 3: Core Functions (lines 500+)
Organized by category:

| Category | Key Functions | What They Do |
|----------|---------------|-------------|
| **Model** | `llama_model_load_from_file`, `llama_model_free` | Load/free models |
| **Context** | `llama_init_from_model`, `llama_free` | Create/free contexts |
| **Vocab** | `llama_tokenize`, `llama_token_to_piece`, `llama_vocab_*` | Tokenization |
| **Decode** | `llama_decode`, `llama_encode` | Run the model |
| **Batch** | `llama_batch_get_one`, `llama_batch_init` | Manage input batches |
| **Sampler** | `llama_sampler_*` | Token sampling |
| **Memory/KV** | `llama_memory_*` | KV cache management |

### Tips

- Functions prefixed with `llama_model_` operate on models.
- Functions prefixed with `llama_vocab_` operate on the vocabulary.
- Functions prefixed with `llama_sampler_` operate on samplers.
- The naming pattern is always `<namespace>_<action>_<noun>`.

**Estimated time: 2-3 days** (return to this file often as a reference)

---

## Phase 4: GGML — The Tensor Engine Under the Hood

**Goal:** Understand the low-level tensor library that powers all computation.

### What is GGML?

GGML is a **standalone C tensor library** (like a mini-NumPy/PyTorch written in C) designed for inference on consumer hardware. It provides:

- Tensor data structures
- Mathematical operations (matmul, softmax, RoPE, etc.)
- Computation graphs (build a graph first, execute it later)
- Backend system (CPU, CUDA, Metal, Vulkan, etc.)
- Quantization (compress model weights for speed/memory)

### Directory Structure

```
ggml/
├── include/          # Public headers
│   ├── ggml.h           ← Core tensor types & operations (~2754 lines)
│   ├── ggml-backend.h   ← Backend interface (CPU, CUDA, Metal, etc.)
│   ├── ggml-alloc.h     ← Memory allocator for tensors
│   ├── ggml-cpu.h       ← CPU-specific functions
│   ├── ggml-cuda.h      ← CUDA backend interface
│   ├── ggml-metal.h     ← Metal backend interface (Apple GPU)
│   └── ...              ← Other backend interfaces
└── src/              # Implementation
    ├── ggml.c / ggml.cpp   ← Core tensor implementation
    ├── ggml-alloc.c        ← Memory allocator
    ├── ggml-backend.cpp    ← Backend dispatcher
    ├── ggml-quants.c       ← Quantization kernels
    ├── ggml-cpu/           ← CPU backend implementation
    ├── ggml-cuda/          ← CUDA backend implementation
    └── ...                 ← Other backends
```

### Key Concepts (Read in Order)

#### 4.1 — Tensors (`ggml.h` top section)

A `ggml_tensor` is a multi-dimensional array (up to 4D):

```c
struct ggml_tensor {
    enum ggml_type type;     // data type (F32, F16, Q4_0, Q8_0, ...)
    int64_t ne[4];           // number of elements per dimension
    size_t  nb[4];           // stride in bytes per dimension
    void * data;             // pointer to raw data
    // ... operation info, source tensors, etc.
};
```

**Important:** Data is stored **row-major**. Dimension 0 = columns, Dimension 1 = rows.

#### 4.2 — Quantization Types

GGML supports many data types — this is how it makes big models fit in small memory:

| Type | Bits/weight | Description |
|------|------------|-------------|
| `GGML_TYPE_F32` | 32 | Full precision float |
| `GGML_TYPE_F16` | 16 | Half precision float |
| `GGML_TYPE_Q4_0` | ~4.5 | 4-bit quantization |
| `GGML_TYPE_Q4_K_M` | ~4.8 | K-quant 4-bit (better quality) |
| `GGML_TYPE_Q8_0` | ~8.5 | 8-bit quantization |

The quantization code lives in `ggml/src/ggml-quants.c`.

#### 4.3 — Computation Graphs

GGML uses **deferred execution**: you build a graph of operations, then execute them all at once.

```c
// 1. Create context
struct ggml_context * ctx = ggml_init(params);

// 2. Create tensors
struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);

// 3. Define operations (no computation yet!)
struct ggml_tensor * c = ggml_add(ctx, a, b);  // c = a + b

// 4. Build and execute graph
struct ggml_cgraph * graph = ggml_new_graph(ctx);
ggml_build_forward_expand(graph, c);
ggml_graph_compute(graph, ...);  // NOW the computation happens
```

#### 4.4 — Backends (`ggml-backend.h`)

The backend system dispatches operations to different hardware:

```
ggml_backend_load_all()     → Discover available backends
ggml_backend_sched           → Scheduler that decides which backend runs which op
ggml_backend_cpu             → CPU implementation
ggml_backend_cuda            → NVIDIA GPU implementation
ggml_backend_metal           → Apple GPU implementation
```

### What to Study

| Priority | File | What to Look For |
|----------|------|-----------------|
| **HIGH** | `ggml/include/ggml.h` (top 200 lines) | Read the big comment explaining the library philosophy |
| **HIGH** | `ggml/include/ggml.h` (tensor ops section) | Skim the list of operations: `ggml_add`, `ggml_mul_mat`, `ggml_rope`, `ggml_soft_max`, etc. |
| **MEDIUM** | `ggml/include/ggml-backend.h` | Understand backend interface |
| **LOW** | `ggml/src/ggml.c` | Only after you're comfortable with the concepts |

### Key Mental Model

```
                    ggml Computation Flow

  Build Phase                     Execute Phase
  ──────────                      ─────────────
  ggml_new_tensor()               ggml_backend_sched_alloc()
  ggml_mul_mat()        →         ggml_backend_sched_compute()
  ggml_soft_max()                    │
  ggml_add()                         ├─ CPU backend
  ...                                ├─ CUDA backend
  ggml_build_forward()               └─ Metal backend
       │
       ▼
  Computation Graph
  (nodes + edges)
```

**Estimated time: 3-5 days**

---

## Phase 5: GGUF — The Model File Format

**Goal:** Understand how model files are structured.

### What is GGUF?

GGUF (GGML Universal Format) is a **binary file format** for storing LLM model weights and metadata. It's the successor to GGML format.

### Structure

```
┌──────────────────────────┐
│  Magic number ("GGUF")   │  4 bytes
│  Version                 │  4 bytes
│  Tensor count            │  8 bytes
│  Metadata KV count       │  8 bytes
├──────────────────────────┤
│  Metadata Key-Value pairs│  Variable size
│  (model name, arch,      │
│   vocab, hyperparams...) │
├──────────────────────────┤
│  Tensor descriptors      │  (name, shape, type, offset)
├──────────────────────────┤
│  Tensor data (aligned)   │  The actual weights
└──────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `ggml/include/gguf.h` | GGUF read/write API |
| `ggml/src/gguf.cpp` | GGUF implementation |
| `src/llama-model-loader.cpp` | How llama.cpp loads GGUF files |
| `src/llama-model-saver.cpp` | How llama.cpp saves GGUF files |
| `gguf-py/` | Python library for reading/writing GGUF |
| `convert_hf_to_gguf.py` | Converts HuggingFace models → GGUF |

### What to Understand

1. GGUF stores both **metadata** (architecture type, vocab, hyperparameters) and **tensor data** (weights).
2. Tensors can be stored in various quantization formats for compression.
3. `llama-model-loader.cpp` reads the GGUF header, maps tensors into memory (often via `mmap`), and sets up the model.

**Estimated time: 1-2 days**

---

## Phase 6: Core llama.cpp Internals (src/)

**Goal:** Understand how the llama library is implemented internally.

This is the biggest phase. Take it module by module.

### Directory Overview

```
src/
├── llama.cpp                  ← Main entry point, ties everything together
├── llama-arch.cpp/.h          ← Model architecture definitions (LLaMA, GPT, Qwen, etc.)
├── llama-model.cpp/.h         ← Model struct + weight loading
├── llama-model-loader.cpp/.h  ← GGUF file reading + tensor mapping
├── llama-context.cpp/.h       ← Context (inference state) management
├── llama-batch.cpp/.h         ← Batch processing (input tokens)
├── llama-graph.cpp/.h         ← Computation graph building
├── llama-vocab.cpp/.h         ← Vocabulary / tokenizer
├── llama-sampler.cpp/.h       ← Token sampling strategies
├── llama-kv-cache.cpp/.h      ← Key-Value cache (attention mechanism memory)
├── llama-memory*.cpp/.h       ← Memory management abstractions
├── llama-hparams.cpp/.h       ← Hyperparameters (model config)
├── llama-cparams.cpp/.h       ← Context parameters
├── llama-adapter.cpp/.h       ← LoRA adapter support
├── llama-quant.cpp/.h         ← Model quantization
├── llama-grammar.cpp/.h       ← Grammar-constrained generation
├── llama-mmap.cpp/.h          ← Memory-mapped file I/O
├── llama-io.cpp/.h            ← Serialization/deserialization
├── llama-impl.cpp/.h          ← Internal utilities
├── models/                    ← Per-architecture model implementations
│   ├── llama.cpp              ← LLaMA architecture
│   ├── gpt2.cpp               ← GPT-2 architecture
│   ├── qwen2.cpp              ← Qwen-2 architecture
│   ├── gemma.cpp              ← Gemma architecture
│   └── ...                    ← 100+ model architectures
├── unicode.cpp/.h             ← Unicode text handling
└── unicode-data.cpp/.h        ← Unicode lookup tables
```

### Study Order (follow this sequence)

#### 6.1 — Architecture Registry (`llama-arch.h`, `llama-arch.cpp`)

**Start here.** This file maps architecture names to their tensor names and hyperparameter keys.

- `enum llm_arch` — Every supported model architecture (LLaMA, GPT-2, Qwen, Gemma, etc.)
- `LLM_TENSOR_*` — Tensor name constants (attention weights, FFN weights, etc.)
- `LLM_KV_*` — Metadata key constants in GGUF files

**Why this matters:** When you see `LLM_TENSOR_ATTN_Q`, it means "the query projection weight tensor for the attention layer".

#### 6.2 — Hyperparameters (`llama-hparams.h`)

```cpp
struct llama_hparams {
    uint32_t n_vocab;       // vocabulary size
    uint32_t n_ctx_train;   // training context length
    uint32_t n_embd;        // embedding dimension
    uint32_t n_layer;       // number of transformer layers
    uint32_t n_head;        // number of attention heads
    uint32_t n_head_kv;     // number of key-value heads (for GQA)
    uint32_t n_ff;          // feed-forward hidden dimension
    // ... many more
};
```

These are the numbers that define a model's shape. Loaded from the GGUF metadata.

#### 6.3 — Model Loading (`llama-model.cpp`, `llama-model-loader.cpp`)

**Reading order:**
1. `llama-model-loader.cpp` — How GGUF files are opened and tensors are mapped
2. `llama-model.cpp` — The `llama_model` struct and how weights are organized

**Key functions to trace:**
- `llama_model_load_from_file()` → entry point
- The loader reads GGUF metadata → fills `llama_hparams`
- The loader reads tensor descriptors → maps tensor data (often via `mmap`)
- Weights are stored in the `llama_model` struct organized by layer

#### 6.4 — Vocabulary (`llama-vocab.cpp`)

The tokenizer converts text ↔ token IDs. Supports multiple algorithms:
- **BPE** (Byte Pair Encoding) — used by GPT-2, LLaMA 3, etc.
- **SPM** (SentencePiece Model) — used by LLaMA 1/2, etc.
- **WPM** (WordPiece Model) — used by BERT

**Key functions:**
- `llama_tokenize()` — text → tokens
- `llama_token_to_piece()` — token → text
- `llama_vocab_bos()` / `llama_vocab_eos()` — special tokens

#### 6.5 — Context and Batch (`llama-context.cpp`, `llama-batch.cpp`)

The context manages the inference state:
- Allocates compute buffers
- Manages the KV cache
- Runs the computation graph

A batch describes a set of input tokens to process in one `llama_decode()` call.

#### 6.6 — Graph Building (`llama-graph.cpp` + `src/models/`)

**This is where the magic happens.** Each model architecture builds its computation graph here.

```
For each transformer layer:
  1. Normalize input            (RMS norm / Layer norm)
  2. Compute Q, K, V            (linear projections)
  3. Apply RoPE                  (positional encoding)
  4. Compute attention           (Q @ K^T → softmax → @ V)
  5. Output projection           (linear)
  6. Feed-forward network        (up proj → activation → down proj)
  7. Residual connection         (add input back)
```

Look at `src/models/llama.cpp` for the LLaMA implementation — it's the canonical example.

#### 6.7 — KV Cache (`llama-kv-cache.cpp`)

The KV (Key-Value) cache stores the key and value tensors from previous tokens so they don't need to be recomputed. This is what makes autoregressive generation efficient.

#### 6.8 — Sampling (`llama-sampler.cpp`)

Converts model output (logits) into a selected token:

| Sampler | What it does |
|---------|-------------|
| `greedy` | Always picks the highest probability token |
| `top_k` | Considers only the top K most likely tokens |
| `top_p` | Considers tokens until cumulative probability reaches P |
| `temperature` | Scales logits to control randomness |
| `min_p` | Filters tokens below a minimum probability |
| `mirostat` | Maintains a target perplexity |
| `grammar` | Constrains output to match a grammar |

Samplers are chained: `temperature → top_k → top_p → sample`.

**Estimated time: 2-3 weeks** (the biggest phase — take your time)

---

## Phase 7: The Common Library (common/)

**Goal:** Understand the shared utility code used by all tools and examples.

### Key Files

| File | Purpose |
|------|---------|
| `common/common.cpp/.h` | General utilities, context/model parameter helpers |
| `common/arg.cpp/.h` | Command-line argument parsing |
| `common/sampling.cpp/.h` | High-level sampling wrapper |
| `common/chat.cpp/.h` | Chat template handling (system/user/assistant roles) |
| `common/json-schema-to-grammar.cpp/.h` | JSON schema → grammar conversion |
| `common/console.cpp/.h` | Terminal/console utilities |
| `common/log.cpp/.h` | Logging system |
| `common/ngram-cache.cpp/.h` | N-gram cache for speculative decoding |
| `common/download.cpp/.h` | Model downloading from URLs |

### What to Understand

- `common` is a **convenience layer** — it wraps the raw llama.h API into higher-level functions.
- The tools (CLI, server) use `common` heavily instead of calling llama.h directly.
- `arg.cpp` implements a rich argument parser (model path, context size, GPU layers, temperature, etc.)
- `chat.cpp` handles chat templates (Jinja2-style) for multi-turn conversations.

**Estimated time: 3-5 days**

---

## Phase 8: Tools & Applications

**Goal:** Study the main user-facing programs.

### Key Tools (in `tools/`)

| Tool | Directory | What It Does |
|------|-----------|-------------|
| **llama-cli** | `tools/cli/` | Interactive command-line chat + text generation |
| **llama-server** | `tools/server/` | OpenAI-compatible HTTP API server |
| **llama-quantize** | `tools/quantize/` | Quantize models (FP16 → Q4_K_M, etc.) |
| **llama-bench** | `tools/llama-bench/` | Benchmarking tool |
| **llama-perplexity** | `tools/perplexity/` | Measure model quality |
| **llama-imatrix** | `tools/imatrix/` | Compute importance matrix for quantization |
| **llama-tokenize** | `tools/tokenize/` | Test tokenization |
| **llama-mtmd** | `tools/mtmd/` | Multimodal model support (images) |
| **llama-tts** | `tools/tts/` | Text-to-speech |

### Study Order

1. **`tools/cli/`** — Read after Phase 6. It's a complete inference loop with all features (chat, grammar, speculative decoding, etc.)
2. **`tools/quantize/`** — Short and teaches you about quantization.
3. **`tools/server/`** — Large codebase, read [tools/server/README-dev.md](tools/server/README-dev.md) first. This is an HTTP server with OpenAI-compatible endpoints.

### Key Examples (in `examples/`)

| Example | What to Learn |
|---------|--------------|
| `simple/` | Minimal inference (you already read this in Phase 2) |
| `simple-chat/` | Chat with templates |
| `batched/` | Parallel sequence generation |
| `embedding/` | Text embeddings |
| `speculative/` | Speculative decoding (draft + verify) |
| `parallel/` | Serving multiple requests |

**Estimated time: 1-2 weeks**

---

## Phase 9: Model Conversion Pipeline (Python)

**Goal:** Understand how models go from HuggingFace format to GGUF.

### Key Files

| File | Purpose |
|------|---------|
| `convert_hf_to_gguf.py` | Main conversion script: HuggingFace → GGUF |
| `convert_hf_to_gguf_update.py` | Updates tokenizer configs for new models |
| `convert_lora_to_gguf.py` | Converts LoRA adapters to GGUF |
| `gguf-py/` | Python GGUF library (reading/writing GGUF files) |

### What to Understand

1. HuggingFace models store weights as PyTorch tensors (`.safetensors` / `.bin`).
2. The conversion script reads those weights, optionally quantizes them, and writes a `.gguf` file.
3. The `gguf-py` library provides Python classes for GGUF metadata and tensor writing.

**Estimated time: 2-3 days** (requires some Python knowledge)

---

## Phase 10: Testing & CI

**Goal:** Understand how the project ensures code quality.

### Test Files (`tests/`)

| Test | What It Tests |
|------|-------------|
| `test-tokenizer-*.cpp` | Tokenizer correctness |
| `test-grammar-*.cpp` | Grammar-constrained generation |
| `test-sampling.cpp` | Sampler behavior |
| `test-backend-ops.cpp` | GGML operations across backends |
| `test-quantize-*.cpp` | Quantization accuracy/performance |
| `test-chat-template.cpp` | Chat template rendering |
| `test-rope.cpp` | Rotary position embeddings |
| `test-json-schema-to-grammar.cpp` | JSON schema conversion |

### CI System

- `ci/run.sh` — Main CI script that builds and runs tests
- Read [ci/README.md](ci/README.md) for how to run CI locally

### How to Run Tests

```bash
cd build
ctest --output-on-failure
```

Or run individual tests:
```bash
./build/bin/test-tokenizer-0 -m your-model.gguf
```

**Estimated time: 1-2 days**

---

## Phase 11: Advanced Topics

**Goal:** Deep-dive into specialized areas based on your interest.

Pick topics that interest you:

### 11.1 — Quantization Deep Dive
- Read `ggml/src/ggml-quants.c` — block quantization algorithms
- Read `src/llama-quant.cpp` — model quantization pipeline
- Read `tools/imatrix/` — importance matrix for optimal quantization
- Key concept: how Q4_K, Q5_K, Q6_K etc. trade quality vs size

### 11.2 — GPU Backend (CUDA)
- `ggml/src/ggml-cuda/` — CUDA kernel implementations
- Each operation (matmul, softmax, RoPE) has optimized GPU code
- Understand how tensor offloading works (some layers on GPU, some on CPU)

### 11.3 — GPU Backend (Metal / Apple Silicon)
- `ggml/src/ggml-metal/` — Metal shader implementations
- How Apple's unified memory model is leveraged

### 11.4 — Speculative Decoding
- Read `examples/speculative/` and `common/speculative.cpp`
- A small "draft" model generates candidate tokens; the main model verifies them in parallel

### 11.5 — Grammar-Constrained Generation
- `src/llama-grammar.cpp` — GBNF grammar engine
- `common/json-schema-to-grammar.cpp` — JSON schema → grammar
- Forces model output to match a specific format

### 11.6 — LoRA Adapters
- `src/llama-adapter.cpp` — LoRA application during inference
- `convert_lora_to_gguf.py` — Converting LoRA weights

### 11.7 — The HTTP Server
- `tools/server/` — Full HTTP server with streaming, embeddings, chat completions
- OpenAI-compatible API
- Read [tools/server/README-dev.md](tools/server/README-dev.md)

### 11.8 — Multimodal (Vision + Language)
- `tools/mtmd/` — Multimodal support
- Read [docs/multimodal.md](docs/multimodal.md)

**Estimated time: Ongoing — pick what interests you**

---

## Appendix A: Directory Map

```
llama.cpp/
│
├── include/                    # PUBLIC API HEADERS
│   ├── llama.h                    # The main llama C API
│   └── llama-cpp.h                # C++ wrapper helpers
│
├── src/                        # CORE LIBRARY IMPLEMENTATION
│   ├── llama.cpp                  # Main entry point
│   ├── llama-model*.cpp           # Model loading/saving
│   ├── llama-context.cpp          # Inference context
│   ├── llama-vocab.cpp            # Tokenizer
│   ├── llama-sampler.cpp          # Sampling
│   ├── llama-kv-cache.cpp         # KV cache
│   ├── llama-graph.cpp            # Computation graph
│   ├── llama-arch.cpp             # Architecture definitions
│   └── models/                    # Per-architecture implementations
│       ├── llama.cpp              # LLaMA
│       ├── gpt2.cpp               # GPT-2
│       └── ...                    # 100+ architectures
│
├── ggml/                       # TENSOR COMPUTATION ENGINE
│   ├── include/                   # ggml public headers
│   │   ├── ggml.h                 # Core tensor API
│   │   ├── ggml-backend.h         # Backend interface
│   │   └── ...
│   └── src/                       # ggml implementation
│       ├── ggml.c                 # Core tensor ops
│       ├── ggml-quants.c          # Quantization
│       ├── ggml-cpu/              # CPU backend
│       ├── ggml-cuda/             # CUDA backend
│       ├── ggml-metal/            # Metal backend
│       └── ...
│
├── common/                     # SHARED UTILITIES
│   ├── common.cpp                 # General helpers
│   ├── arg.cpp                    # CLI argument parsing
│   ├── sampling.cpp               # High-level sampling
│   ├── chat.cpp                   # Chat templates
│   └── ...
│
├── tools/                      # MAIN APPLICATIONS
│   ├── cli/                       # llama-cli (chat/generate)
│   ├── server/                    # llama-server (HTTP API)
│   ├── quantize/                  # llama-quantize
│   └── ...
│
├── examples/                   # EXAMPLE PROGRAMS
│   ├── simple/                    # Minimal inference
│   ├── simple-chat/               # Minimal chat
│   └── ...
│
├── tests/                      # TEST SUITE
│
├── convert_hf_to_gguf.py      # MODEL CONVERSION (Python)
├── gguf-py/                    # Python GGUF library
│
├── docs/                       # DOCUMENTATION
│   ├── build.md                   # Build instructions
│   └── ...
│
├── CMakeLists.txt              # ROOT BUILD FILE
├── CONTRIBUTING.md             # Contribution guidelines
└── AGENTS.md                   # AI usage policy
```

---

## Appendix B: Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **Token** | The smallest unit of text the model works with (a word piece or character) |
| **Tokenizer / Vocab** | Converts text → token IDs and back |
| **Embedding** | A dense vector representing a token |
| **Transformer** | The neural network architecture used by LLMs |
| **Attention** | Mechanism that lets the model focus on relevant parts of the input |
| **KV Cache** | Stored key/value tensors from previous tokens to avoid recomputation |
| **Logits** | Raw (unnormalized) output scores for each possible next token |
| **Sampling** | Choosing the next token from logits (greedy, top-k, top-p, etc.) |
| **Quantization** | Reducing precision of weights (FP32 → 4-bit) to save memory |
| **GGUF** | Binary file format for storing model weights and metadata |
| **Context length** | Maximum number of tokens the model can "see" at once |
| **Batch** | A group of tokens processed together in one forward pass |
| **Autoregressive** | Generating one token at a time, feeding it back as input |
| **RoPE** | Rotary Position Embedding — how position info is encoded |
| **GQA** | Grouped Query Attention — shares KV heads across query heads |
| **MoE** | Mixture of Experts — routes tokens to specialized sub-networks |
| **LoRA** | Low-Rank Adaptation — lightweight model fine-tuning |
| **Speculative decoding** | Using a fast draft model to speed up generation |
| **mmap** | Memory-mapping: maps file contents into virtual memory (fast loading) |
| **Backend** | Hardware-specific implementation (CPU, CUDA, Metal, Vulkan) |
| **Computation graph** | A DAG of tensor operations built before execution |
| **Perplexity** | Measure of how well a model predicts text (lower = better) |
| **BPE** | Byte Pair Encoding — a tokenization algorithm |
| **SPM** | SentencePiece Model — another tokenization algorithm |
| **FFN** | Feed-Forward Network — the MLP part of each transformer layer |

---

## Appendix C: Recommended Reading Order for Source Files

For a focused, depth-first reading of the code:

### Tier 1 — Start Here (Understand the API + flow)
1. `examples/simple/simple.cpp`
2. `include/llama.h`
3. `examples/simple-chat/simple-chat.cpp`

### Tier 2 — Core Internals (How it works)
4. `src/llama-arch.h` (architecture enum + tensor names)
5. `src/llama-hparams.h` (model shape)
6. `src/llama-vocab.cpp` (tokenizer)
7. `src/llama-model-loader.cpp` (GGUF loading)
8. `src/llama-model.cpp` (model struct)
9. `src/llama-context.cpp` (context management)
10. `src/llama-batch.cpp` (batching)

### Tier 3 — The Neural Network (Forward pass)
11. `src/llama-graph.cpp` (graph builder base)
12. `src/models/llama.cpp` (LLaMA architecture — the reference implementation)
13. `src/llama-kv-cache.cpp` (KV cache)
14. `src/llama-sampler.cpp` (sampling)

### Tier 4 — GGML Foundations
15. `ggml/include/ggml.h` (tensor types + ops — read the doc comment at top)
16. `ggml/include/ggml-backend.h` (backend abstraction)
17. `ggml/src/ggml-quants.c` (quantization kernels — skim for structure)

### Tier 5 — Utilities and Applications
18. `common/arg.cpp` (CLI parsing)
19. `common/sampling.cpp` (high-level sampling)
20. `common/chat.cpp` (chat templates)
21. `tools/cli/` (the full CLI tool)
22. `tools/server/` (the HTTP server)

---

## Study Tips

1. **Don't read linearly.** Jump between the API (llama.h) and implementations (src/) as needed.
2. **Use a debugger.** Build in Debug mode, set breakpoints in `simple.cpp`, and step through.
3. **Trace one function.** Pick `llama_decode()` and trace it from the API call down to ggml operations.
4. **Read tests.** Tests in `tests/` show exactly how features are used.
5. **Start small.** Modify `examples/simple/simple.cpp` — change the prompt, add temperature sampling, etc.
6. **Use `grep`.** When you see an unfamiliar function, `grep -rn "function_name" src/` to find its definition.
7. **Read contributing guidelines** before making any changes: [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).
8. **Ask specific questions.** "How does `llama_decode` work?" is better than "explain the code."

---

> **Remember:** This is a large codebase (~200+ source files). Nobody understands all of it at once.
> Focus on the path: **simple.cpp → llama.h → src/llama-*.cpp → ggml → tools**.
> Each pass through deepens your understanding.

**Total estimated time: 6-10 weeks** (studying a few hours daily)
