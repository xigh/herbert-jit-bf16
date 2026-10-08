//! How many workers by default: the performance cores of the machine.

/// Number of performance cores.
///
/// On Linux, the physical P cores of a hybrid (`/sys/devices/cpu_core/cpus`),
/// otherwise the physical cores, SMT siblings excluded — one full L2 per
/// thread. On Windows, the total number of CPUs, for lack of a topology this
/// module knows how to read. On an Apple machine with few performance cores,
/// the performance cores and half of the efficiency cores: on the M5 (4 + 6),
/// 7 workers decode at 14.1 tokens/s where 4 give 9.7 and all 10 give 9.4 —
/// the spinning barriers then wait for the cores the system takes back
/// (tasks handed out by ticket, `Knobs::task_grain`). With eight performance
/// cores or more, the performance cores alone: on the M3 Ultra (20 + 8) the
/// CPU's share of the memory saturates at 256 GB/s from 8 threads, 20 decode
/// at 26.5 tokens/s, 16 at 27.0, 8 at 27.6, and 24 (P + E/2) fall to 4–13 —
/// the efficiency cores of the second die hold every barrier.
///
/// Cached after the first call.
pub fn get_performance_cpu_count() -> usize {
    use std::sync::OnceLock;
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        #[cfg(target_os = "linux")]
        {
            // Hybrid Intel (P + E, `/sys/devices/cpu_core/cpus`): only the P
            // cores — measured on a Lunar Lake, 3 P cores alone decode faster
            // than 8 threads split equally. The pool pins P cores first
            // (`pool::topology`), so "as many workers as P cores" = the P cores.
            let p = count_physical_p_cores_linux();
            if p > 0 {
                return p;
            }
            let count = count_physical_cores_linux();
            if count > 0 {
                count
            } else {
                std::thread::available_parallelism().map_or(1, |n| n.get())
            }
        }

        #[cfg(target_os = "macos")]
        {
            use crate::core::isa::cores_of_level;
            if let Some(p) = cores_of_level(0) {
                return if p >= 8 { p } else { p + cores_of_level(1).unwrap_or(0) / 2 };
            }
            std::thread::available_parallelism().map_or(1, |n| n.get())
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        }
    })
}

/// Participants of the decode by default, when not the whole pool.
///
/// The decode reads the weights once per token and is bound by the memory
/// the CPU can draw; the prefill wants every performance core. On a Mac with
/// more than sixteen performance cores the decode takes sixteen of the pool
/// (`Tuning::decode_workers`) — measured on the M3 Ultra, pool of 20, three
/// interleaved repetitions: 27.37 tokens/s at 20, 27.57 at 8, 27.53
/// at 12, **27.69 at 16** (26.56 → 27.09 after a 1 666-token prompt), the
/// prefill unchanged (244 tokens/s). Elsewhere, `None`: the whole pool.
pub fn default_decode_workers() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        if let Some(p) = crate::core::isa::cores_of_level(0) {
            return decode_workers_for(p);
        }
    }
    None
}

/// Does the default pool mix performance and efficiency cores? On a Mac
/// with fewer than eight performance cores the pool takes half of the
/// efficiency cores too ([`get_performance_cpu_count`]): the guided chunks
/// (`Knobs::task_guided`) are then the default. Elsewhere, `false`.
pub fn default_pool_includes_efficiency_cores() -> bool {
    #[cfg(target_os = "macos")]
    {
        use crate::core::isa::cores_of_level;
        if let (Some(p), Some(e)) = (cores_of_level(0), cores_of_level(1)) {
            return p < 8 && e >= 2;
        }
    }
    false
}

/// The rule of [`default_decode_workers`] for `p` performance cores.
pub const fn decode_workers_for(p: usize) -> Option<usize> {
    if p > 16 { Some(16) } else { None }
}

/// Physical P cores of an Intel hybrid (`/sys/devices/cpu_core/cpus`, one per
/// set of `thread_siblings`); 0 if the machine is not hybrid.
#[cfg(target_os = "linux")]
fn count_physical_p_cores_linux() -> usize {
    use std::collections::BTreeSet;
    let Ok(list) = std::fs::read_to_string("/sys/devices/cpu_core/cpus") else {
        return 0;
    };
    if !std::path::Path::new("/sys/devices/cpu_atom/cpus").exists() {
        return 0;
    }
    let mut seen = BTreeSet::new();
    for part in list.trim().split(',') {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.parse::<usize>().ok(), b.parse::<usize>().ok()),
            None => (part.parse::<usize>().ok(), part.parse::<usize>().ok()),
        };
        let (Some(a), Some(b)) = (a, b) else { continue };
        for cpu in a..=b {
            let siblings = std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ))
            .unwrap_or_default();
            let canonical = siblings
                .trim()
                .split(',')
                .filter_map(|s| s.split('-').next().and_then(|x| x.parse::<usize>().ok()))
                .min()
                .unwrap_or(cpu);
            seen.insert(canonical);
        }
    }
    seen.len()
}

/// Count physical cores on Linux by parsing thread_siblings_list.
#[cfg(target_os = "linux")]
fn count_physical_cores_linux() -> usize {
    use std::collections::BTreeSet;

    let cpu_dir = std::path::Path::new("/sys/devices/system/cpu");
    let Ok(entries) = std::fs::read_dir(cpu_dir) else {
        return 0;
    };

    let mut seen_cores = BTreeSet::new();

    let cpus: Vec<usize> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("cpu")
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect();

    for cpu_id in cpus {
        let siblings_path = cpu_dir
            .join(format!("cpu{}", cpu_id))
            .join("topology/thread_siblings_list");
        let Ok(siblings_str) = std::fs::read_to_string(&siblings_path) else {
            continue;
        };

        // Parse siblings list and use the smallest as canonical
        let mut min_sibling = usize::MAX;
        for part in siblings_str.trim().split(',') {
            let part = part.trim();
            if let Some((start, _end)) = part.split_once('-') {
                if let Ok(s) = start.parse::<usize>() {
                    min_sibling = min_sibling.min(s);
                }
            } else if let Ok(n) = part.parse::<usize>() {
                min_sibling = min_sibling.min(n);
            }
        }

        if min_sibling != usize::MAX {
            seen_cores.insert(min_sibling);
        }
    }

    seen_cores.len()
}
