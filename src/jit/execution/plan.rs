use crate::jit::target::{Native, Target};
use std::fmt;

/// A failure in plan construction, binding, or execution.
#[derive(Debug, PartialEq, Eq)]
pub struct ExecutionError(pub String);
impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ExecutionError {}

/// Relocatable code produced by a kernel emitter. Calls and branches must stay
/// within this kernel. No Rust computation pointer can be registered here.
pub struct GeneratedKernel {
    pub(crate) bytes: Vec<u8>,
}
impl GeneratedKernel {
    /// A kernel that only returns, assembled like every other kernel: the
    /// structural no-op of a plan (the block-begin node, a compaction with
    /// nothing to move).
    pub fn ret() -> Result<Self, ExecutionError> {
        // SAFETY: a single `ret`, no memory access and no control transfer.
        unsafe { Self::new(Native::ret_kernel()?) }
    }
    /// Check the control-flow boundary of an already emitted kernel.
    ///
    /// # Safety
    /// Code must implement the SysV64 ABI, return normally, preserve callee-saved
    /// registers and access only memory authorized by its arguments. The decoder
    /// check is a boundary audit, not a machine-code safety proof.
    pub unsafe fn new(bytes: Vec<u8>) -> Result<Self, ExecutionError> {
        Native::audit_kernel(&bytes)?;
        Ok(Self { bytes })
    }
}

/// Distribution within an operation, independent of the collective team size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distribution {
    /// The pool's contiguous ceil-divided ranges, including empty final ranges.
    Contiguous,
    /// Strided task indices, useful for existing cyclic kernel schedules.
    Cyclic,
}

/// A synchronized point in a block. All participants pass every operation,
/// including workers with no assigned tasks.
#[derive(Clone, Debug)]
pub enum Operation {
    /// A grid of invocations of one generated kernel.
    Kernel {
        /// Index in the plan's owned generated kernels.
        kernel: usize,
        /// Maximum number of tasks allowed by reserved buffers.
        max_tasks: usize,
        /// Existing operation-specific worker cap (must be positive).
        worker_limit: usize,
        /// Existing operation-specific distribution.
        distribution: Distribution,
    },
    /// Optional progress after a fully synchronized layer.
    LayerEnd {
        /// Zero-based layer index.
        layer: usize,
    },
}

/// A node and its earlier dependencies. The initial executor deliberately
/// synchronizes every node; this preserves deterministic dependency order.
#[derive(Clone, Debug)]
pub struct Node {
    /// Work or progress at this point.
    pub operation: Operation,
    /// Indices of prerequisite nodes in this block.
    pub dependencies: Vec<usize>,
}

/// Semantic identity of one synchronized plan node. Names are static lowering
/// vocabulary, so trace decoding needs no hot-path allocation or lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeLabel {
    /// Generated operation performed by the node.
    pub operation: &'static str,
    /// Model layer containing the operation, or `None` for block input/output.
    pub layer: Option<usize>,
}

/// Common immutable plan, built at model mount and shared by both drivers.
pub struct ExecutionPlan {
    pub(crate) kernels: Vec<GeneratedKernel>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) labels: Vec<NodeLabel>,
}
impl ExecutionPlan {
    /// Validate the topological order, kernel references and worker caps.
    pub fn new(kernels: Vec<GeneratedKernel>, nodes: Vec<Node>) -> Result<Self, ExecutionError> {
        let labels = nodes
            .iter()
            .map(|node| match node.operation {
                Operation::Kernel { .. } => NodeLabel {
                    operation: "kernel",
                    layer: None,
                },
                Operation::LayerEnd { layer } => NodeLabel {
                    operation: "layer-end",
                    layer: Some(layer),
                },
            })
            .collect();
        Self::new_labeled(kernels, nodes, labels)
    }

    pub(crate) fn new_labeled(
        kernels: Vec<GeneratedKernel>,
        nodes: Vec<Node>,
        labels: Vec<NodeLabel>,
    ) -> Result<Self, ExecutionError> {
        if nodes.is_empty() {
            return Err(ExecutionError("execution plan has no nodes".into()));
        }
        if labels.len() != nodes.len() || labels.iter().any(|label| label.operation.is_empty()) {
            return Err(ExecutionError(
                "execution plan labels do not match nodes".into(),
            ));
        }
        for (index, node) in nodes.iter().enumerate() {
            if node
                .dependencies
                .iter()
                .any(|&dependency| dependency >= index)
            {
                return Err(ExecutionError(format!(
                    "node {index} has a non-preceding dependency"
                )));
            }
            if let Operation::Kernel {
                kernel,
                worker_limit,
                max_tasks,
                ..
            } = node.operation
            {
                if kernel >= kernels.len() || worker_limit == 0 || max_tasks > isize::MAX as usize {
                    return Err(ExecutionError(format!(
                        "node {index} has an invalid kernel, task capacity or worker cap"
                    )));
                }
            }
        }
        Ok(Self {
            kernels,
            nodes,
            labels,
        })
    }

    /// Ordered nodes of this plan.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    /// Semantic labels in the same order as [`Self::nodes`].
    pub fn labels(&self) -> &[NodeLabel] {
        &self.labels
    }
}

/// A register argument, advanced by `task * stride` for each invocation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Argument {
    /// Address or scalar bits for task zero.
    pub base: u64,
    /// Byte stride for addresses, integer stride for scalars.
    pub stride: u64,
}

/// One node's parameters in one block. Tables are reserved before execution;
/// generated code only reads them. SysV integer and SSE arguments are separate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TaskBinding {
    /// Number of tasks this block actually needs (zero is legal).
    pub tasks: usize,
    /// Optional count produced by an earlier synchronized generated node.
    /// Null selects `tasks`. The pointed value must stay immutable during this node.
    pub task_count: *const usize,
    /// Optional current row count for preserving an operation's M=1 policy.
    pub worker_rows: *const usize,
    /// Worker cap at M=1 when `worker_rows` is non-null. Must be within the
    /// operation's static worker limit; participants still join all barriers.
    pub worker_limit_m1: usize,
    /// rdi, rsi, rdx, rcx, r8, r9.
    pub arguments: [Argument; 6],
    /// Scalar f32 bits for xmm0 through xmm7.
    pub floats: [u32; 8],
    /// Additional offset per participant for each integer argument, e.g. scores.
    pub participant_strides: [u64; 6],
    /// Optional dynamic base values produced by earlier generated nodes. Null
    /// means zero; a non-null base is added to the constant `Argument::base`.
    pub argument_bases: [*const u64; 6],
}

/// Storage for call parameters, reserved before a hot call. The caller fills
/// the active prefix before launch; it may change budgets between calls.
pub struct Bindings {
    pub(super) table: Vec<TaskBinding>,
    pub(super) nodes: usize,
    max_blocks: usize,
    repeated: bool,
}
impl Bindings {
    /// Reserve `max_blocks * node_count` records, checking arithmetic overflow.
    pub fn new(plan: &ExecutionPlan, max_blocks: usize) -> Result<Self, ExecutionError> {
        let count = max_blocks
            .checked_mul(plan.nodes.len())
            .filter(|n| *n <= isize::MAX as usize / std::mem::size_of::<TaskBinding>())
            .ok_or_else(|| ExecutionError("binding capacity overflow".into()))?;
        Ok(Self {
            table: vec![TaskBinding::default(); count],
            nodes: plan.nodes.len(),
            max_blocks,
            repeated: false,
        })
    }
    /// Reserve one immutable table reused for every generated block. Dynamic
    /// counts/bases carry the current block; no per-block Rust preparation is needed.
    pub fn repeated(plan: &ExecutionPlan, max_blocks: usize) -> Result<Self, ExecutionError> {
        let mut bindings = Self::new(plan, 1)?;
        bindings.max_blocks = max_blocks;
        bindings.repeated = true;
        Ok(bindings)
    }
    pub(super) fn block_stride(&self) -> usize {
        if self.repeated {
            0
        } else {
            self.nodes * std::mem::size_of::<TaskBinding>()
        }
    }
    /// Mutable parameters for a block. Not accessible during an executor call.
    /// A repeated table is initialized through block zero only.
    pub fn block_mut(&mut self, block: usize) -> &mut [TaskBinding] {
        assert!(
            !self.repeated || block == 0,
            "a repeated table has only block zero"
        );
        &mut self.table[block * self.nodes..(block + 1) * self.nodes]
    }
    /// Validate all active records before launching any participant.
    pub(super) fn validate(
        &self,
        plan: &ExecutionPlan,
        blocks: usize,
        participants: usize,
    ) -> Result<(), ExecutionError> {
        if self.nodes != plan.nodes.len() || blocks == 0 || blocks > self.max_blocks {
            return Err(ExecutionError(
                "call exceeds binding capacity or contains zero blocks".into(),
            ));
        }
        let stored_blocks = if self.repeated { 1 } else { blocks };
        for records in self.table[..stored_blocks * self.nodes].chunks_exact(self.nodes) {
            for (record, node) in records.iter().zip(&plan.nodes) {
                if let Operation::Kernel {
                    max_tasks,
                    worker_limit,
                    ..
                } = node.operation
                {
                    if record.task_count.is_null() && record.tasks > max_tasks {
                        return Err(ExecutionError("task count exceeds plan capacity".into()));
                    }
                    if !record.worker_rows.is_null()
                        && (record.worker_limit_m1 == 0 || record.worker_limit_m1 > worker_limit)
                    {
                        return Err(ExecutionError("invalid single-row worker limit".into()));
                    }
                    if !(record.worker_rows as usize).is_multiple_of(std::mem::align_of::<usize>())
                        || !(record.task_count as usize)
                            .is_multiple_of(std::mem::align_of::<usize>())
                        || record
                            .argument_bases
                            .iter()
                            .any(|p| !(*p as usize).is_multiple_of(std::mem::align_of::<u64>()))
                    {
                        return Err(ExecutionError("unaligned dynamic task parameter".into()));
                    }
                    let count = if record.task_count.is_null() {
                        record.tasks
                    } else {
                        max_tasks
                    };
                    for (i, a) in record.arguments.iter().enumerate() {
                        if a.stride
                            .checked_mul(count.saturating_sub(1) as u64)
                            .and_then(|last| a.base.checked_add(last))
                            .and_then(|last| {
                                record.participant_strides[i]
                                    .checked_mul(participants.saturating_sub(1) as u64)
                                    .and_then(|stride| last.checked_add(stride))
                            })
                            .is_none()
                        {
                            return Err(ExecutionError("task argument address overflow".into()));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Layer/block progress. Reported exactly once by participant zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionProgress {
    /// Zero-based block index.
    pub block: usize,
    /// Total blocks in this call.
    pub blocks: usize,
    /// Completed layer, or `None` for a completed block.
    pub layer: Option<usize>,
}

fn statically_serial(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Kernel {
            max_tasks,
            worker_limit,
            ..
        } if *max_tasks <= 1 || *worker_limit == 1
    )
}

/// A skipped boundary is internal to a strict serial dependency chain. The
/// barrier before its first node publishes all earlier producers; participant
/// zero then executes the chain in program order, and its final barrier
/// publishes the accumulated writes before any parallel consumer can run.
pub(crate) fn serial_barrier_can_be_fused(plan: &ExecutionPlan, node: usize) -> bool {
    let Some(next) = plan.nodes.get(node + 1) else {
        return false;
    };
    statically_serial(&plan.nodes[node].operation)
        && statically_serial(&next.operation)
        && next.dependencies.as_slice() == [node]
}

pub(crate) fn serial_barriers_fused(plan: &ExecutionPlan, enabled: bool) -> usize {
    if enabled {
        (0..plan.nodes.len())
            .filter(|&node| serial_barrier_can_be_fused(plan, node))
            .count()
    } else {
        0
    }
}
