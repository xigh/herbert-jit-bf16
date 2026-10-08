//! herbert-bf16-chat MODEL_DIR [options]
//!
//! Loads Qwen3.5-4B in bf16, mounts the generated engine, and runs a
//! conversation. Without `--ask`, the scripted demonstration runs: two turns,
//! then the date-and-time tool switched on for a third, then the conversation
//! saved, restored and continued.
//!
//!   --threads N          workers (default: the performance cores)
//!   --no-think           close the thinking block in the header
//!   --rewrite-history    keep only the final answer of each turn in the state
//!   --max-tokens N       ceiling per answer (default 1024)
//!   --system TEXT        the system prompt (default: English, answers in French)
//!   --no-system          no system message at all
//!   --ask TEXT           a user turn (repeatable) instead of the demonstration
//!   --temperature T      sampling instead of greedy (Qwen's top-k 20 / top-p 0.95)
//!   --pmu EVENT          cycles, instructions, branch-misses, l1i-misses, itlb-misses, llc-misses
//!   --sync-trace         rdtscp stamps per node and participant
//!   --trace-blocks N     blocks kept per trace (default 64)
//!   --trace-dir DIR      where the JSON traces go (default: traces)
//!   --no-matrix          keep the prefill projections on the vector tile (no SME)
//!   --amx                the prefill projections on Apple's AMX (the bits of --neon-mac bfmlal)
//!   --task-grain N       hand the tasks of a node out in chunks, about N per
//!                        participant (0 = the static split; AArch64)
//!   --task-guided        recompute the chunk from the remaining tasks at every grab
//!   --task-chunk-min N   the smallest chunk a grab takes (default 1)
//!   --task-chunk-max N   the largest chunk a grab takes (0 = none)
//!   --ticket-line        the ticket counter on its own cache line
//!   --fuse-projections   the projections of one input in one node (AArch64)
//!   --argmax-tasks N     the argmax of the logits in N slices on the team (AArch64)
//!   --neon-nr N          columns of the decode GEMM tile: 16, 32, 48, 64
//!   --neon-prefetch B    prefetch B bytes ahead of every weight block (0 = none)
//!   --neon-prefetch-l1   prefetch into L1 instead of L2
//!   --neon-mac M         the multiply-accumulate of the vector GEMM: bfdot | bfmlal
//!   --neon-acc-sets N    accumulator sets of the vector GEMM: 1 | 2
//!   --neon-load L        the load of a pair of weights: ldp | ld1
//!   --gemm-tail-rows R   residue rows up to R (≤ 6) of a prefill block on a vector
//!                        kernel inside the SME task instead of a padded tile
//!   --gemm-tail-mac M    the multiply-accumulate of that kernel: bfdot | bfmlal
//!   --gemm-tail-bfdot-max N  residues up to N rows take the bfdot tail (default 1)
//!   --rvv-nr N           RISC-V: columns of the decode GEMM tile, a multiple of 16 up to 112
//!   --rvv-gemm F         RISC-V: the GEMM micro-kernel, seg (vl = 16, default) or wide
//!                        (the whole vector length, folded at the end of the tile)
//!   --rvv-width W        RISC-V: the lanes a folded sum accumulates over, full (VLMAX,
//!                        default) or N (a power of two ≥ 4: the same bits at every VLEN)
//!   --gemm-tile MRxNR    pin the vector GEMM tile (default: the ISA's)
//!   --prefill-pass N     rows of one prefill pass (default 256)
//!   --prefill-workers N  participants of the prefill (default: the whole pool)
//!   --decode-workers N   participants of the decode (default: the whole pool; 16 on a Mac with more than 16 P cores)
//!   --attn-block, --no-attn-block  prefill attention by blocks of 16 queries, as two GEMMs (AArch64 default), or row by row
//!   --elem-team          norms and residual additions on the whole team
//!   --attn-chain-team    the chain around the attention on the whole team
//!   --deltanet-fma       the DeltaNet recurrence with fused multiply-adds
//!   --deltanet-store-once  the recurrence stores its state once per position (AArch64)
//!   --deltanet-pairs     the recurrence loads and stores its state by pairs (AArch64)
//!   --deltanet-q-hoist   q·scale once per position in the recurrence (AArch64)
//!   --deltanet-pass1-lanes N  lanes per loop of the recurrence's first pass: 32 | 64
//!   --deltanet-vd-split N  each value head in N slices of columns, one task each
//!   --deltanet-chunked, --no-deltanet-chunked  the prefill recurrence by chunks of 16 on SME2 (WY form; the default with SME2)
//!   --no-deltanet-team   the DeltaNet stages on the caller alone
//!   --fuse-serial        no barrier around the serial nodes
//!   --dump-asm DIR       write the disassembly of both mounted programs (layers 0 and 3,
//!                        the embedding, the head) into DIR/{prefill,decode}/ and exit

use herbert_bf16::chat::{current_date_time, Block, Chat, Tokens};
use herbert_bf16::core::sampler::Sampling;
use herbert_bf16::engine::{Engine, Measure, Tuning};
use herbert_bf16::jit::emit::{NeonLoad, NeonMac, RvvGemm, RvvWidth};
use herbert_bf16::jit::execution::PmuEvent;
use herbert_bf16::jit::ir::TileShape;
use herbert_bf16::jit::TilePolicy;
use std::io::Write;
use std::path::PathBuf;

struct Options {
    model_dir: PathBuf,
    threads: usize,
    think: bool,
    rewrite_history: bool,
    max_tokens: usize,
    system: Option<String>,
    asks: Vec<String>,
    sampling: Sampling,
    measure: Measure,
    trace_dir: PathBuf,
    dump_asm: Option<PathBuf>,
    tuning: Tuning,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut it = args.iter();
    let model_dir = PathBuf::from(it.next().ok_or("MODEL_DIR is required")?);
    let mut o = Options {
        model_dir,
        threads: std::env::var("HERBERT_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(herbert_bf16::core::cpu_detection::get_performance_cpu_count),
        think: true,
        rewrite_history: false,
        max_tokens: 1024,
        system: Some("You are a helpful assistant. Always answer in French.".to_string()),
        asks: Vec::new(),
        sampling: Sampling::greedy(),
        measure: Measure::none(),
        trace_dir: PathBuf::from("traces"),
        dump_asm: None,
        tuning: Tuning::default(),
    };
    let mut trace_blocks = 64usize;
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--threads" => o.threads = value()?.parse().map_err(|e| format!("--threads: {e}"))?,
            "--no-think" => o.think = false,
            "--rewrite-history" => o.rewrite_history = true,
            "--max-tokens" => o.max_tokens = value()?.parse().map_err(|e| format!("--max-tokens: {e}"))?,
            "--system" => o.system = Some(value()?),
            "--no-system" => o.system = None,
            "--ask" => o.asks.push(value()?),
            "--temperature" => {
                let t: f32 = value()?.parse().map_err(|e| format!("--temperature: {e}"))?;
                o.sampling = Sampling::qwen_thinking(42);
                o.sampling.temperature = t;
            }
            "--pmu" => {
                let event = PmuEvent::parse(&value()?).map_err(|e| e.to_string())?;
                o.measure.pmu = Some(event);
                o.measure.sync_trace = true;
            }
            "--sync-trace" => o.measure.sync_trace = true,
            "--trace-blocks" => trace_blocks = value()?.parse().map_err(|e| format!("--trace-blocks: {e}"))?,
            "--trace-dir" => o.trace_dir = PathBuf::from(value()?),
            "--dump-asm" => o.dump_asm = Some(PathBuf::from(value()?)),
            "--no-matrix" => o.tuning.kernels.gemm_matrix = false,
            "--amx" => o.tuning.kernels.gemm_amx = true,
            "--task-grain" => o.tuning.kernels.task_grain = number(flag, value()?)?,
            "--task-guided" => o.tuning.kernels.task_guided = true,
            "--no-task-guided" => o.tuning.kernels.task_guided = false,
            "--task-chunk-min" => o.tuning.kernels.task_chunk_min = number(flag, value()?)?,
            "--task-chunk-max" => o.tuning.kernels.task_chunk_max = number(flag, value()?)?,
            "--ticket-line" => o.tuning.kernels.ticket_line = true,
            "--fuse-projections" => o.tuning.kernels.fuse_projections = true,
            "--argmax-tasks" => {
                let tasks: usize = number(flag, value()?)?;
                if tasks > 64 {
                    return Err("--argmax-tasks: at most 64".into());
                }
                o.tuning.kernels.argmax_tasks = tasks;
            }
            "--neon-nr" => {
                let nr: usize = number(flag, value()?)?;
                if !matches!(nr, 16 | 32 | 48 | 64) {
                    return Err("--neon-nr: 16, 32, 48 or 64".into());
                }
                o.tuning.kernels.neon.nr = nr;
            }
            "--neon-prefetch" => o.tuning.kernels.neon.prefetch = number(flag, value()?)?,
            "--neon-prefetch-l1" => o.tuning.kernels.neon.prefetch_l1 = true,
            "--neon-mac" => {
                o.tuning.kernels.neon.mac = match value()?.as_str() {
                    "bfdot" => NeonMac::BfDot,
                    "bfmlal" => NeonMac::BfMlal,
                    other => return Err(format!("--neon-mac: bfdot or bfmlal, not {other}")),
                }
            }
            "--neon-acc-sets" => o.tuning.kernels.neon.acc_sets = number(flag, value()?)?,
            "--neon-load" => {
                o.tuning.kernels.neon.load = match value()?.as_str() {
                    "ldp" => NeonLoad::Ldp,
                    "ld1" => NeonLoad::Ld1x4,
                    other => return Err(format!("--neon-load: ldp or ld1, not {other}")),
                }
            }
            "--gemm-tail-rows" => {
                let rows: usize = number(flag, value()?)?;
                if rows > herbert_bf16::jit::compile::MAX_TAIL_ROWS {
                    return Err(format!("--gemm-tail-rows: at most {}", herbert_bf16::jit::compile::MAX_TAIL_ROWS));
                }
                o.tuning.kernels.neon.tail_rows = rows as u8;
            }
            "--gemm-tail-mac" => {
                o.tuning.kernels.neon.tail_mac = match value()?.as_str() {
                    "bfdot" => NeonMac::BfDot,
                    "bfmlal" => NeonMac::BfMlal,
                    other => return Err(format!("--gemm-tail-mac: bfdot or bfmlal, not {other}")),
                }
            }
            "--gemm-tail-bfdot-max" => o.tuning.kernels.neon.tail_bfdot_max = number(flag, value()?)?,
            "--rvv-nr" => {
                let nr: u16 = number(flag, value()?)?;
                if nr == 0 || !nr.is_multiple_of(16) || nr > 112 {
                    return Err("--rvv-nr: a multiple of 16, at most 112".into());
                }
                o.tuning.kernels.rvv.nr = nr;
            }
            "--rvv-gemm" => {
                o.tuning.kernels.rvv.gemm = match value()?.as_str() {
                    "seg" => RvvGemm::Seg,
                    "wide" => RvvGemm::Wide,
                    other => return Err(format!("--rvv-gemm: seg or wide, not {other}")),
                }
            }
            "--rvv-width" => {
                o.tuning.kernels.rvv.width = match value()?.as_str() {
                    "full" => RvvWidth::Full,
                    n => {
                        let lanes: u16 = number(flag, n.to_string())?;
                        if lanes < 4 || !lanes.is_power_of_two() {
                            return Err("--rvv-width: full, or a power of two ≥ 4".into());
                        }
                        RvvWidth::Lanes(lanes)
                    }
                }
            }
            "--gemm-tile" => {
                let v = value()?;
                let (mr, nr) = v.split_once('x').ok_or("--gemm-tile: MRxNR")?;
                o.tuning.kernels.tile = TilePolicy::Fixed(TileShape { mr: number(flag, mr.to_string())?, nr: number(flag, nr.to_string())? });
            }
            "--prefill-pass" => o.tuning.kernels.prefill_pass = number(flag, value()?)?,
            "--prefill-workers" => o.tuning.prefill_workers = Some(number(flag, value()?)?),
            "--decode-workers" => o.tuning.decode_workers = Some(number(flag, value()?)?),
            "--attn-block" => o.tuning.kernels.attn_block = true,
            "--no-attn-block" => o.tuning.kernels.attn_block = false,
            "--elem-team" => o.tuning.kernels.elem_team = true,
            "--attn-chain-team" => o.tuning.kernels.attn_chain_team = true,
            "--deltanet-fma" => o.tuning.kernels.deltanet_fma = true,
            "--deltanet-store-once" => o.tuning.kernels.deltanet_store_once = true,
            "--deltanet-pairs" => o.tuning.kernels.deltanet_pairs = true,
            "--deltanet-q-hoist" => o.tuning.kernels.deltanet_q_hoist = true,
            "--deltanet-pass1-lanes" => {
                let lanes: u8 = number(flag, value()?)?;
                if !matches!(lanes, 32 | 64) {
                    return Err("--deltanet-pass1-lanes: 32 or 64".into());
                }
                o.tuning.kernels.deltanet_pass1_lanes = lanes;
            }
            "--deltanet-chunked" => o.tuning.kernels.deltanet_chunked = true,
            "--no-deltanet-chunked" => o.tuning.kernels.deltanet_chunked = false,
            "--deltanet-vd-split" => o.tuning.kernels.deltanet_vd_split = number(flag, value()?)?,
            "--no-deltanet-team" => {
                o.tuning.kernels.deltanet_recurrence_team = false;
                o.tuning.kernels.deltanet_conv_team = false;
                o.tuning.kernels.deltanet_norm_team = false;
            }
            "--fuse-serial" => o.tuning.fuse_serial = true,
            other => return Err(format!("unknown option {other}")),
        }
    }
    o.measure.trace_blocks = trace_blocks;
    Ok(o)
}

/// A numeric value of a flag.
fn number<T: std::str::FromStr>(flag: &str, value: String) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    value.parse().map_err(|e| format!("{flag}: {e}"))
}

fn main() {
    if let Some(code) = herbert_bf16::core::isa::refuse_if_below_floor() {
        std::process::exit(code);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let options = match parse(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}\nusage: herbert-bf16-chat MODEL_DIR [--threads N] [--no-think] [--rewrite-history] [--max-tokens N] [--system TEXT | --no-system] [--ask TEXT]... [--temperature T] [--pmu EVENT] [--sync-trace] [--trace-blocks N] [--trace-dir DIR] [--dump-asm DIR] [--no-matrix] [--amx] [--task-grain N] [--task-guided | --no-task-guided] [--task-chunk-min N] [--task-chunk-max N] [--ticket-line] [--fuse-projections] [--argmax-tasks N] [--neon-nr N] [--neon-prefetch B] [--neon-prefetch-l1] [--neon-mac bfdot|bfmlal] [--neon-acc-sets N] [--neon-load ldp|ld1] [--gemm-tail-rows R] [--gemm-tail-mac bfdot|bfmlal] [--gemm-tail-bfdot-max N] [--rvv-nr N] [--rvv-gemm seg|wide] [--rvv-width full|N] [--gemm-tile MRxNR] [--prefill-pass N] [--prefill-workers N] [--decode-workers N] [--attn-block | --no-attn-block] [--elem-team] [--attn-chain-team] [--deltanet-fma] [--deltanet-store-once] [--deltanet-pairs] [--deltanet-q-hoist] [--deltanet-pass1-lanes N] [--deltanet-vd-split N] [--deltanet-chunked | --no-deltanet-chunked] [--no-deltanet-team] [--fuse-serial]");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(options) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(o: Options) -> herbert_bf16::core::error::Result<()> {
    eprintln!("[isa] {}", herbert_bf16::core::isa::summary());
    let engine = Engine::load(&o.model_dir, o.threads, 8192, &o.tuning, &o.measure)?;
    if let Some(dir) = &o.dump_asm {
        for (prefill, name) in [(true, "prefill"), (false, "decode")] {
            let program = engine.runner().execution_program(prefill).expect("both regimes are mounted");
            // Layer 0 is the first GatedDeltaNet layer, layer 3 the first attention layer.
            let n = herbert_bf16::jit::listing::write_listings(program, &dir.join(name), &[0, 3])?;
            eprintln!("[asm] {n} listings of the {name} program in {}", dir.join(name).display());
        }
        return Ok(());
    }
    let tokens = Tokens::load(&o.model_dir)?;
    eprintln!("[tokens] {}", tokens.summary());
    let mut chat = Chat::new(engine, tokens, o.sampling);
    chat.think = o.think;
    chat.rewrite_history = o.rewrite_history;
    chat.max_tokens = o.max_tokens;
    if let Some(system) = &o.system {
        chat.system(system)?;
    }
    if o.measure.is_on() {
        chat.trace_dir = Some(o.trace_dir.clone());
        eprintln!("[trace] one JSON per generated call in {}", o.trace_dir.display());
    }
    let mut turns = Turns { count: 0 };

    if !o.asks.is_empty() {
        for ask in &o.asks {
            turns.run(&mut chat, ask)?;
        }
        return Ok(());
    }

    // The demonstration: two turns, the tool switched on for the third, the
    // conversation saved, restored into a fresh session and continued.
    turns.run(&mut chat, "Quelle est la capitale de la France ?")?;
    turns.run(&mut chat, "Et celle de l'Italie ?")?;
    chat.enable_tools(vec![current_date_time()]);
    turns.run(&mut chat, "Quel jour sommes-nous, et quelle heure est-il ?")?;
    let saved = std::env::temp_dir().join("herbert-bf16-chat.state");
    chat.save(&saved)?;
    eprintln!("[state] saved to {} ({} tokens)", saved.display(), chat.ledger().len());
    let engine = chat.into_engine();
    let tokens = Tokens::load(&o.model_dir)?;
    let mut chat = Chat::restore(engine, tokens, o.sampling, &saved)?;
    if o.measure.is_on() {
        chat.trace_dir = Some(o.trace_dir.clone());
    }
    chat.think = o.think;
    chat.rewrite_history = o.rewrite_history;
    chat.max_tokens = o.max_tokens;
    chat.enable_tools_announced(vec![current_date_time()]);
    eprintln!("[state] restored: {} tokens, position {}", chat.ledger().len(), chat.engine().pos());
    turns.run(&mut chat, "Merci. Et la capitale de l'Espagne ?")?;
    Ok(())
}

struct Turns {
    count: usize,
}

impl Turns {
    /// One turn, printed as it streams: the blocks on their own line, the
    /// text as it comes, the ids and the figures on stderr at the end.
    fn run(&mut self, chat: &mut Chat, user: &str) -> herbert_bf16::core::error::Result<()> {
        self.count += 1;
        println!("\n>>> {user}");
        let mut out = std::io::stdout();
        let stats = chat.chat(
            user,
            &mut |block| {
                let label = match block {
                    Block::Thinking => "[thinking]".to_string(),
                    Block::Text => "[answer]".to_string(),
                    Block::ToolCall => "[tool call]".to_string(),
                    Block::ToolResult { name } => format!("[tool result: {name}]"),
                };
                println!("\n{label}");
            },
            &mut |text| {
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
            },
        )?;
        println!();
        // The generated ids, for a comparison with another engine's output.
        let ledger = chat.ledger();
        let generated = &ledger[ledger.len().saturating_sub(stats.generated_tokens)..];
        eprintln!("[ids] {}", generated.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
        eprintln!("[decoded] {:?}", chat.tokens().decode(generated));
        eprintln!(
            "[turn {}] {} prompt tokens in {:.3} s ({:.1} tok/s) · {} generated in {:.2} s ({:.2} tok/s) · {} tool calls · rewrite {:.3} s · position {}",
            self.count,
            stats.prompt_tokens,
            stats.prefill_seconds,
            stats.prefill_tokens_per_second(),
            stats.generated_tokens,
            stats.decode_seconds,
            stats.decode_tokens_per_second(),
            stats.tool_calls,
            stats.rewrite_seconds,
            stats.position
        );
        Ok(())
    }
}
