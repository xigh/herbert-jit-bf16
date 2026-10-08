# herbert-jit-bf16 — Qwen3.5 in bf16, no quantization, all compute emitted at run time

An inference engine for a single model, Qwen3.5 (text), with a single numerical contract: bf16 weights, bf16 activations, bf16 KV cache, f32 accumulators. Nothing is quantized. All the compute of a token (projections, norms, attention, GatedDeltaNet mixer, language-model head) is machine code emitted at startup, assembled into one program per regime (prefill, decode) that every thread of the pool runs from end to end. Rust only loads, emits and drives the conversation.

This repository holds the code only: no tests, no documentation.

## Targets

| target | floor | emitter |
|---|---|---|
| x86_64 Linux and Windows | AVX2 + FMA; zmm kernels with AVX-512 BF16 (`vdpbf16ps`) | `iced-x86` |
| AArch64 macOS | NEON + BF16 + LSE; prefill projections on SME2 when present, or on Apple's AMX with `--amx` | `crates/a64` |
| riscv64 Linux | RV64GC + V 1.0 at VLEN ≥ 256 + Zba + Zbb + Zvfbfmin + Zvfbfwma | `crates/rv` |

`build.rs` and `src/lib.rs` refuse any other target, and the machine's ISA is checked at startup, before any allocation. `crates/a64` and `crates/rv` are two dependency-free assemblers and decoders: the equivalent of `iced-x86` for ARMv8/ARMv9 (integer, NEON, BF16, SME2, and the words of Apple's AMX) and for RISC-V (RV64GC, RVV 1.0, bf16).

The data buffers the kernels read and write (packed weights, norms, activations, KV cache, recurrent states, logits) all start on a 64-byte cache line, whatever the allocator (`src/core/avec.rs`).

## Build and run

```sh
cargo build --release
target/release/herbert-bf16-chat ~/models/Qwen3.5-4B
target/release/herbert-bf16-chat ~/models/Qwen3.5-4B --no-think --ask 'What is the capital of France?'
```

The model directory is the checkpoint's (`config.json`, `tokenizer.json`, safetensors shards); a Hugging Face cache directory is resolved to its snapshot.

Without `--ask`, `main` runs a demonstration: two turns, the date/time tool enabled for a third, then the conversation saved to disk, restored and continued. Common options: `--threads N`, `--no-think`, `--max-tokens N`, `--system TEXT | --no-system`, `--temperature T`, `--ask TEXT` (repeatable), `--dump-asm DIR` (writes the disassembly of both programs and exits). The full list, with the settings of each kernel, is at the top of `src/main.rs`.

## Apple AMX

`--amx` moves the prefill projections to AMX, the undocumented matrix coprocessor of Apple's M-series chips. It is off by default. Each lane of the outer product is one fused multiply-add, rounded to nearest even, taken in the same order over `k` as the NEON `bfmlal` kernel, so the output is bit-identical to `--no-matrix --neon-mac bfmlal`. The decode is unchanged.

## RISC-V + RVV, under QEMU

The `riscv64` target runs under emulation, in a Linux guest of `qemu-system-riscv64`. It therefore gives no speed: it gives outputs, comparable token by token with those of the two other targets. It was developed with Qwen3.5-0.8B, QEMU 11.1.1 and a 6.18 kernel.

The binary builds on any host, static (musl), linked by the Rust toolchain's `rust-lld`:

```sh
rustup target add riscv64gc-unknown-linux-musl
export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_MUSL_LINKER=rust-lld
export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C target-feature=+crt-static -C link-self-contained=yes"
cargo build --release --target riscv64gc-unknown-linux-musl --bin herbert-bf16-chat
```

The guest receives the binary and the model through a 9p share (here the `share` directory), with a riscv64 kernel and initramfs (Alpine, for instance):

```sh
qemu-system-riscv64 -M virt -m 6G -smp 4 -nographic -no-reboot -bios default \
    -cpu rv64,v=true,vlen=256,elen=64,vext_spec=v1.0,zvfh=true,zvfbfwma=true,zba=true,zbb=true,zbs=true \
    -kernel Image -initrd initramfs.cpio.gz -append "console=ttyS0" \
    -virtfs local,path=share,mount_tag=host0,security_model=none
```

Then, in the guest:

```sh
mkdir -p /share
mount -t 9p -o trans=virtio,version=9p2000.L,msize=524288 host0 /share
cd /share
./herbert-bf16-chat Qwen3.5-0.8B --no-system --no-think --ask 'What is the capital of France?'
```

`vlen` sets the length of the vector registers: 256 at least, 1024 at most with upstream QEMU. Three settings are specific to this target: `--rvv-gemm seg|wide` (the GEMM micro-kernel: `seg`, the default, works at `vl = 16` and emits the same instructions at any VLEN; `wide` takes the whole vector length), `--rvv-width full|N` (the lanes a folded sum accumulates over: `N` gives the same bits at any VLEN) and `--rvv-nr N` (the columns of the decode GEMM tile).

## The code, in the order `main` uses it

| module | what it does |
|---|---|
| `core` | the model's `config.json`, bf16 conversions, the ISA and memory checks, the aligned buffers, the sampler |
| `pool` | the worker threads, pinned to the physical cores, that run the generated program |
| `model` | the safetensors checkpoint, packed into bf16 weights (R16 layout) |
| `jit` | the emitters (GEMM, norms, attention, GatedDeltaNet), the generated program and its scheduler |
| `engine` | the model mounted in both regimes (prefill, decode) and driven step by step |
| `chat` | the Qwen3.5 chat template, tokens, streaming detokenization, tools, the session |

## License

MIT, see `LICENSE`.

The emitted scalar `exp` and `log` follow the algorithm and coefficients of Arm optimized-routines, under the MIT option of its dual license: the notice and the original revision are in `third_party/arm-optimized-routines/`.
