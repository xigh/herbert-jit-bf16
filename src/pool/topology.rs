//! The CPU topology: which logical CPU each worker is pinned to.
//!
//! On Linux, sysfs gives the physical cores (one CPU per core, SMT siblings
//! skipped, so each worker keeps a whole L2), the L3 groups (CCDs, interleaved
//! so the first workers spread over the fabric) and the P/E split of a hybrid
//! (P cores first). `/proc/stat` says whether a targeted CPU is already busy.
//! On Windows there is no topology to read: no pinning.

use std::sync::OnceLock;

/// Returns logical CPU IDs — one per physical core, skipping SMT siblings.
///
/// On Linux: parses `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`
/// to pick the first logical CPU per physical core.
/// On non-Linux: returns empty vec (no affinity support).
///
/// **The order is NOT ascending** as soon as there are multiple CCDs: the
/// `physical_core_ids_ccd_aware` path deliberately interleaves L3 groups so
/// that the first N workers are distributed across all CCDs. On a 9900X
/// (2 CCDs) this yields `[0, 6, 1, 7, …]`. This order determines the
/// distribution of workers across fabric links, and thus the aggregate
/// bandwidth actually available.
pub fn physical_core_ids() -> &'static [usize] {
    static IDS: OnceLock<Vec<usize>> = OnceLock::new();
    IDS.get_or_init(|| {
        // `PoolSettings::cpus`: the list of CPUs for the workers, in order —
        // worker i is pinned to the i-th. Without it the automatic path below
        // decides (`taskset` cannot: `sched_setaffinity` overrides the
        // inherited mask).
        if let Some(ids) = crate::pool::settings::settings().cpus.clone().filter(|ids| !ids.is_empty()) {
            return ids;
        }
        #[cfg(target_os = "linux")]
        {
            physical_core_ids_linux()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Vec::new()
        }
    })
}

#[cfg(target_os = "linux")]
fn physical_core_ids_linux() -> Vec<usize> {
    // Intel hybrid (P + E): P cores first, then E cores — measured on a Lunar
    // Lake, 3 P cores alone decode faster than 8 threads at equal share, and
    // the CCD-aware path below, designed for multi-CCD EPYCs, yields
    // [0, 4, 5, 6, 7, 1, 2, 3] there (each E core is its own "CCD"). Only the
    // ORDER is decided here: who works when fewer than all are requested.
    let hybrid = physical_core_ids_hybrid();
    if !hybrid.is_empty() {
        return hybrid;
    }
    // Try CCD-aware distribution first, fall back to sequential
    let ccd_result = physical_core_ids_ccd_aware();
    if !ccd_result.is_empty() {
        return ccd_result;
    }

    physical_core_ids_sequential()
}

/// CCD-aware core distribution: detects L3 cache groups (CCDs) and distributes
/// cores round-robin across them for maximum aggregate DRAM bandwidth.
///
/// On EPYC 7443P (3 CCDs × 8 cores): instead of [0,1,...,11] (CCD0×8, CCD1×4),
/// produces [0,8,16,1,9,17,...] (CCD0×4, CCD1×4, CCD2×4).
#[cfg(target_os = "linux")]
fn physical_core_ids_ccd_aware() -> Vec<usize> {
    use std::collections::{BTreeMap, BTreeSet};

    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    let Ok(entries) = std::fs::read_dir(cpu_dir) else {
        return Vec::new();
    };

    let mut cpus: Vec<usize> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("cpu")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect();
    cpus.sort_unstable();

    // Step 1: Get physical cores (skip SMT siblings)
    let mut seen_cores = BTreeSet::new();
    let mut physical_cores = Vec::new();

    for &cpu_id in &cpus {
        let siblings_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("topology/thread_siblings_list");
        let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) else {
            continue;
        };
        let siblings = parse_cpu_list(siblings_str.trim());
        if siblings.is_empty() {
            continue;
        }
        let canonical = *siblings.iter().min().expect("siblings non-empty");
        if seen_cores.insert(canonical) {
            physical_cores.push(cpu_id);
        }
    }

    if physical_cores.is_empty() {
        return Vec::new();
    }

    // Step 2: Detect CCD groups via L3 cache (index3/shared_cpu_list)
    // Each CCD has its own L3 cache; cores sharing an L3 are on the same CCD.
    let mut ccd_groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();

    for &cpu_id in &physical_cores {
        let l3_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("cache/index3/shared_cpu_list");
        let l3_key = if let Ok(l3_str) = std::fs::read_to_string(&l3_path) {
            let mut key = parse_cpu_list(l3_str.trim());
            key.sort_unstable();
            key
        } else {
            // No L3 info — treat each core as its own group (degrades to sequential)
            vec![cpu_id]
        };

        ccd_groups.entry(l3_key).or_default().push(cpu_id);
    }

    let num_ccds = ccd_groups.len();
    if num_ccds <= 1 {
        // Single CCD (or couldn't detect multiple): sequential is fine
        return Vec::new();
    }

    // Step 3: Round-robin distribution across CCDs
    // Sort CCD groups by their first core ID for deterministic ordering
    let mut ccds: Vec<Vec<usize>> = ccd_groups.into_values().collect();
    ccds.sort_by_key(|group| group.first().copied().unwrap_or(0));

    let max_cores_per_ccd = ccds.iter().map(|g| g.len()).max().unwrap_or(0);
    let mut result = Vec::new();

    for slot in 0..max_cores_per_ccd {
        for ccd in &ccds {
            if slot < ccd.len() {
                result.push(ccd[slot]);
            }
        }
    }

    eprintln!("[pool] {num_ccds} CCDs, {} cores, worker order {:?}…", result.len(), result.iter().take(12).collect::<Vec<_>>());

    result
}

/// Sequential physical core IDs (original behavior, used as fallback).
/// Hybrid Intel: `/sys/devices/cpu_core/cpus` (P) and `/sys/devices/cpu_atom/cpus`
/// (E). Returns P (one CPU per physical core, the first of `thread_siblings`)
/// then E; empty if the machine is not hybrid.
#[cfg(target_os = "linux")]
fn physical_core_ids_hybrid() -> Vec<usize> {
    use std::collections::BTreeSet;
    let read = |path: &str| -> Vec<usize> {
        std::fs::read_to_string(path)
            .map(|s| parse_cpu_list(s.trim()))
            .unwrap_or_default()
    };
    let p = read("/sys/devices/cpu_core/cpus");
    let e = read("/sys/devices/cpu_atom/cpus");
    if p.is_empty() || e.is_empty() {
        return Vec::new();
    }
    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for cpu_id in p.into_iter().chain(e) {
        let siblings_path = cpu_dir.join(format!("cpu{cpu_id}")).join("topology/thread_siblings_list");
        let canonical = std::fs::read_to_string(&siblings_path)
            .ok()
            .and_then(|s| parse_cpu_list(s.trim()).into_iter().min())
            .unwrap_or(cpu_id);
        if seen.insert(canonical) {
            result.push(cpu_id);
        }
    }
    result
}

#[cfg(target_os = "linux")]
fn physical_core_ids_sequential() -> Vec<usize> {
    use std::collections::BTreeSet;

    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    let mut seen_cores = BTreeSet::new();
    let mut result = Vec::new();

    let Ok(entries) = std::fs::read_dir(cpu_dir) else {
        return Vec::new();
    };

    let mut cpus: Vec<usize> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("cpu")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect();
    cpus.sort_unstable();

    for cpu_id in cpus {
        let siblings_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("topology/thread_siblings_list");
        let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) else {
            continue;
        };
        let siblings = parse_cpu_list(siblings_str.trim());
        if siblings.is_empty() {
            continue;
        }
        let canonical = *siblings.iter().min().expect("siblings non-empty");
        if seen_cores.insert(canonical) {
            result.push(cpu_id);
        }
    }

    result.sort_unstable();
    result
}

/// Parse a CPU list string like "0,8" or "0-3" or "0-3,8-11" into a Vec of CPU IDs.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                for i in s..=e {
                    result.push(i);
                }
            }
        } else if let Ok(n) = part.parse::<usize>() {
            result.push(n);
        }
    }
    result
}

/// Logical CPU occupancy on a window, read from `/proc/stat` (Linux).
///
/// What the pool cannot otherwise know: a CPU it is going to pin might
/// already be taken — by another process, or by an interrupt storm (a
/// laptop's touchscreen controller firing 13 000 to 42 000 IRQ/s on one CPU
/// cost a third of its time, and every measurement that included it was false).
///
/// The shares are reported against the **wall time** of the window, not the sum
/// of the counters: without `CONFIG_IRQ_TIME_ACCOUNTING` (generic Ubuntu), the time
/// spent in interrupt on an idle CPU is counted **nowhere** — on
/// one laptop the sum of cpu1 jiffies is 209 out of 300 expected, `irq` stays
/// at 0 %. This hole is `lost`; it is included in `occupied`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuOccupation {
    pub cpu: usize,
    /// Everything that is not idle: `1 − (idle + iowait) / window`.
    pub occupied: f32,
    /// The portion counted as interrupts (`irq` + `softirq`) — 0 without IRQ accounting.
    pub irq: f32,
    /// The share that the kernel counted nowhere — the signature of an IRQ storm.
    pub lost: f32,
    /// The time stolen by the hypervisor (`steal`) — the signature of a loaded VM.
    pub stolen: f32,
}

/// Measures the occupation of requested `cpus` on `window_of` (two readings of
/// `/proc/stat`, a sleep between the two). Outside Linux: empty list. A CPU
/// absent from `/proc/stat` is omitted. Resolution: a jiffy (10 ms at `USER_HZ`
/// 100), so a short window is noisy — 300 ms give ±3 %.
pub fn occupation_cpus(cpus: &[usize], window_of: std::time::Duration) -> Vec<CpuOccupation> {
    #[cfg(target_os = "linux")]
    {
        let user_hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f32;
        let before = read_proc_stat();
        let top = std::time::Instant::now();
        std::thread::sleep(window_of);
        let after = read_proc_stat();
        let expected = top.elapsed().as_secs_f32() * user_hz;
        cpus.iter()
            .filter_map(|&cpu| {
                let a = before.iter().find(|(c, _)| *c == cpu)?.1;
                let b = after.iter().find(|(c, _)| *c == cpu)?.1;
                let mut d = [0f32; 8];
                for i in 0..8 {
                    d[i] = b[i].saturating_sub(a[i]) as f32 / expected;
                }
                // user nice system idle iowait irq softirq steal
                let total: f32 = d.iter().sum();
                Some(CpuOccupation {
                    cpu,
                    occupied: (1.0 - d[3] - d[4]).clamp(0.0, 1.0),
                    irq: d[5] + d[6],
                    lost: (1.0 - total).clamp(0.0, 1.0),
                    stolen: d[7],
                })
            })
            .collect()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (cpus, window_of);
        Vec::new()
    }
}

/// CPU time consumed by this process (`utime` + `stime` from `/proc/self/stat`,
/// in jiffies), to tell the reader of a warning what share of the observed
/// load is ours (model loading, kernel prefetch…).
#[cfg(target_os = "linux")]
pub fn cpu_time_self_jiffies() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Field 2 (`comm`) is in parentheses and may contain spaces:
    // we start from the last closing parenthesis.
    let rest = &text[text.rfind(')')? + 1..];
    let words: Vec<&str> = rest.split_whitespace().collect();
    // After `)`: state(0) ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime(11) stime(12)
    let utime: u64 = words.get(11)?.parse().ok()?;
    let stime: u64 = words.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// The first eight counters of each `cpuN` line in `/proc/stat`
/// (`user nice system idle iowait irq softirq steal`, in jiffies).
#[cfg(target_os = "linux")]
fn read_proc_stat() -> Vec<(usize, [u64; 8])> {
    let Ok(text) = std::fs::read_to_string("/proc/stat") else {
        return Vec::new();
    };
    text
        .lines()
        .filter_map(|l| {
            let mut words = l.split_whitespace();
            let cpu = words.next()?.strip_prefix("cpu")?.parse::<usize>().ok()?;
            let mut v = [0u64; 8];
            for x in v.iter_mut() {
                *x = words.next()?.parse().ok()?;
            }
            Some((cpu, v))
        })
        .collect()
}
