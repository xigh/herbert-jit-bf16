use super::ExecutionError;
use std::fmt;

/// Linux architectural performance event sampled by generated `RDPMC`
/// instructions. One event is selected per mount so Linux never multiplexes
/// the counter behind a trace that is meant to describe precise intervals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PmuEvent {
    /// Unhalted user-mode core cycles.
    Cycles,
    /// Retired user-mode instructions.
    Instructions,
    /// Mispredicted retired branches.
    BranchMisses,
    /// Level-one instruction-cache read misses.
    L1iMisses,
    /// Instruction-TLB read misses.
    ItlbMisses,
    /// Generic last-level cache misses reported by Linux.
    LlcMisses,
}

impl PmuEvent {
    /// The event named on the command line: `cycles`, `instructions`, `branch-misses`,
    /// `l1i-misses`, `itlb-misses` or `llc-misses`.
    pub fn parse(value: &str) -> Result<Self, ExecutionError> {
        match value {
            "cycles" => Ok(Self::Cycles),
            "instructions" => Ok(Self::Instructions),
            "branch-misses" => Ok(Self::BranchMisses),
            "l1i-misses" => Ok(Self::L1iMisses),
            "itlb-misses" => Ok(Self::ItlbMisses),
            "llc-misses" => Ok(Self::LlcMisses),
            _ => Err(ExecutionError(format!(
                "HERBERT_JIT_SPMD_PMU unknown event {value:?}; expected cycles, instructions, branch-misses, l1i-misses, itlb-misses or llc-misses"
            ))),
        }
    }
}

impl fmt::Display for PmuEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cycles => "cycles",
            Self::Instructions => "instructions",
            Self::BranchMisses => "branch-misses",
            Self::L1iMisses => "l1i-misses",
            Self::ItlbMisses => "itlb-misses",
            Self::LlcMisses => "llc-misses",
        })
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use crate::pool::thread_pool::ThreadPool;
    use memmap2::{Mmap, MmapOptions};
    use std::fs::File;
    use std::os::fd::FromRawFd;
    use std::sync::Mutex;

    const PERF_TYPE_HARDWARE: u32 = 0;
    const PERF_TYPE_HW_CACHE: u32 = 3;
    const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
    const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;
    const PERF_COUNT_HW_BRANCH_MISSES: u64 = 5;
    const PERF_COUNT_HW_CACHE_MISSES: u64 = 3;
    const PERF_COUNT_HW_CACHE_L1I: u64 = 1;
    const PERF_COUNT_HW_CACHE_ITLB: u64 = 4;
    const PERF_COUNT_HW_CACHE_OP_READ: u64 = 0;
    const PERF_COUNT_HW_CACHE_RESULT_MISS: u64 = 1;
    const PERF_FLAG_FD_CLOEXEC: libc::c_ulong = 1 << 3;
    const ATTR_PINNED: u64 = 1 << 2;
    const ATTR_EXCLUDE_KERNEL: u64 = 1 << 5;
    const ATTR_EXCLUDE_HV: u64 = 1 << 6;
    const CAP_USER_RDPMC: u64 = 1 << 2;

    // PERF_ATTR_SIZE_VER0. All fields used here are in the original stable
    // 64-byte ABI, which also lets this work on older Linux kernels.
    #[repr(C)]
    struct PerfEventAttr {
        event_type: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events: u32,
        bp_type: u32,
        config1: u64,
    }

    fn encoding(event: PmuEvent) -> (u32, u64) {
        let cache =
            |id| id | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        match event {
            PmuEvent::Cycles => (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES),
            PmuEvent::Instructions => (PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS),
            PmuEvent::BranchMisses => (PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES),
            PmuEvent::L1iMisses => (PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_L1I)),
            PmuEvent::ItlbMisses => (PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_ITLB)),
            PmuEvent::LlcMisses => (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES),
        }
    }

    pub(super) struct Handle {
        _file: File,
        page: Mmap,
        owner_tid: libc::pid_t,
    }

    impl Handle {
        fn open(event: PmuEvent) -> Result<Self, ExecutionError> {
            let (event_type, config) = encoding(event);
            let attr = PerfEventAttr {
                event_type,
                size: std::mem::size_of::<PerfEventAttr>() as u32,
                config,
                sample_period: 0,
                sample_type: 0,
                read_format: 0,
                flags: ATTR_PINNED | ATTR_EXCLUDE_KERNEL | ATTR_EXCLUDE_HV,
                wakeup_events: 0,
                bp_type: 0,
                config1: 0,
            };
            let fd = unsafe {
                libc::syscall(
                    libc::SYS_perf_event_open,
                    &attr,
                    0,
                    -1,
                    -1,
                    PERF_FLAG_FD_CLOEXEC,
                ) as libc::c_int
            };
            if fd < 0 {
                return Err(ExecutionError(format!(
                    "cannot open per-thread PMU event {event}: {}",
                    std::io::Error::last_os_error()
                )));
            }
            let file = unsafe { File::from_raw_fd(fd) };
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page_size <= 0 {
                return Err(ExecutionError(
                    "cannot determine Linux page size for PMU mmap".into(),
                ));
            }
            let page = unsafe { MmapOptions::new().len(page_size as usize).map(&file) }
                .map_err(|e| ExecutionError(format!("cannot mmap PMU event {event}: {e}")))?;
            // perf_event_mmap_page.capabilities is at byte 40. Bit 2 is the
            // non-deprecated cap_user_rdpmc flag.
            let capabilities = unsafe { (page.as_ptr().add(40) as *const u64).read_volatile() };
            let width = unsafe { (page.as_ptr().add(48) as *const u16).read_volatile() };
            if capabilities & CAP_USER_RDPMC == 0 || width == 0 || width > 64 {
                return Err(ExecutionError(format!(
                    "Linux does not expose user RDPMC for {event} (cap_user_rdpmc={}, width={width})",
                    usize::from(capabilities & CAP_USER_RDPMC != 0)
                )));
            }
            Ok(Self {
                _file: file,
                page,
                owner_tid: unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t },
            })
        }

        fn page(&self) -> *const u8 {
            self.page.as_ptr()
        }
    }

    pub(crate) struct Participants {
        handles: Vec<Handle>,
        pages: Vec<*const u8>,
    }

    // The mappings are read only, and generated code only reads participant i's
    // mapping while running on the thread that created event i.
    unsafe impl Send for Participants {}
    unsafe impl Sync for Participants {}

    impl Participants {
        pub(crate) fn prepare(pool: &ThreadPool, event: PmuEvent) -> Result<Self, ExecutionError> {
            let slots: Vec<_> = (0..pool.num_workers()).map(|_| Mutex::new(None)).collect();
            pool.collective(
                pool.num_workers(),
                |id| {
                    *slots[id].lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(Handle::open(event));
                },
                || {},
            )
            .map_err(|e| ExecutionError(format!("PMU participant setup failed: {e}")))?;
            let mut handles = Vec::with_capacity(slots.len());
            for (id, slot) in slots.into_iter().enumerate() {
                handles.push(
                    slot.into_inner()
                        .unwrap_or_else(|e| e.into_inner())
                        .ok_or_else(|| {
                            ExecutionError(format!("PMU participant {id} did not initialize"))
                        })??,
                );
            }
            let pages = handles.iter().map(Handle::page).collect();
            Ok(Self { handles, pages })
        }

        pub(crate) fn pages(&self) -> *const *const u8 {
            self.pages.as_ptr()
        }

        pub(crate) fn validate_caller(&self, pool: &ThreadPool) -> Result<(), ExecutionError> {
            if pool.caller_participates() {
                let current = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };
                if self.handles[0].owner_tid != current {
                    return Err(ExecutionError(format!(
                        "SPMD PMU participant 0 was mounted on Linux TID {} but this call uses TID {current}; remount on the inference thread",
                        self.handles[0].owner_tid
                    )));
                }
            }
            Ok(())
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::*;
    use crate::pool::thread_pool::ThreadPool;

    pub(crate) struct Participants;
    impl Participants {
        pub(crate) fn prepare(_: &ThreadPool, event: PmuEvent) -> Result<Self, ExecutionError> {
            Err(ExecutionError(format!(
                "SPMD PMU event {event} requires Linux perf_event_open and user RDPMC"
            )))
        }
        pub(crate) fn pages(&self) -> *const *const u8 {
            std::ptr::null()
        }
        pub(crate) fn validate_caller(&self, _: &ThreadPool) -> Result<(), ExecutionError> {
            Ok(())
        }
    }
}

pub(crate) use imp::Participants;
