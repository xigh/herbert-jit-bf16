//! The physical memory check: refuse before allocating what the machine
//! cannot hold, rather than be killed by the OOM killer mid-prefill.
//!
//! Only `/proc/meminfo` is read, so the check is inert elsewhere: a check
//! that cannot see must not refuse. Swap is never counted as capacity.

use crate::core::error::{HerbertError, Result};

/// Bytes to GiB, for messages.
pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// A field of `/proc/meminfo`, in bytes.
#[cfg(target_os = "linux")]
fn meminfo(name: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines().find(|l| l.starts_with(name))?.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
}

/// What the kernel can hand out without swapping: `MemAvailable`, which
/// counts the reclaimable page cache — most of the apparent occupancy right
/// after a checkpoint has been streamed.
pub fn available() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        meminfo("MemAvailable:")
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Physical memory installed.
pub fn total() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        meminfo("MemTotal:")
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Total size of the `*.safetensors` of a model directory: the floor of the
/// resident weights, since bf16 goes in and bf16 stays.
pub fn checkpoint_bytes(model_dir: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(model_dir) else { return 0 };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("safetensors")))
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// Refuse a plan the machine cannot hold. `planned` is what will be
/// allocated; a tenth of the machine, at least 1 GiB, is kept for what is
/// never counted (prefill buffers, the tokenizer, the allocator's slack).
pub fn refuse_if_it_cannot_fit(label: &str, planned: u64) -> Result<()> {
    let (Some(avail), Some(total_ram)) = (available(), total()) else {
        return Ok(());
    };
    let margin = (total_ram / 10).max(1 << 30);
    if planned.saturating_add(margin) <= avail {
        return Ok(());
    }
    Err(HerbertError::ModelLoad(format!(
        "{label}: not enough physical memory — {:.1} GiB planned plus {:.1} GiB of margin, {:.1} GiB of {:.1} GiB available",
        gib(planned),
        gib(margin),
        gib(avail),
        gib(total_ram),
    )))
}
