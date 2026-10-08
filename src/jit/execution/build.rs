//! Mount-time assembly helpers. Every table and context allocation belongs to
//! the mounted program; block state is subsequently produced by emitted nodes.
use super::abi::{CallFrame, TaskGrid};
use crate::jit::target::{KernelSet, Native};
use super::*;
use std::mem::{align_of, size_of};

pub(crate) struct Builder {
    pub frame: Box<CallFrame>,
    pub batch: usize,
    kernels: Vec<GeneratedKernel>,
    nodes: Vec<Node>,
    labels: Vec<NodeLabel>,
    records: Vec<TaskBinding>,
    grids: Vec<TaskGrid>,
    grid_of: Vec<Option<usize>>,
    storage: Vec<Box<[u64]>>,
    final_only: bool,
    layer: Option<usize>,
    pending_label: Option<&'static str>,
}
impl Builder {
    pub fn new(batch: usize) -> Result<Self, ExecutionError> {
        if batch == 0 {
            return Err(ExecutionError("zero mounted block size".into()));
        }
        let frame = Box::new(CallFrame {
            tokens: std::ptr::null(),
            total_tokens: 0,
            position: 0,
            processed: 0,
            rows: 0,
            token_cursor: std::ptr::null(),
            last_block: 0,
            counts: std::ptr::null_mut(),
        });
        let mut result = Self {
            frame,
            batch,
            kernels: Vec::new(),
            nodes: Vec::new(),
            labels: Vec::new(),
            records: Vec::new(),
            grids: Vec::new(),
            grid_of: Vec::new(),
            storage: Vec::new(),
            final_only: false,
            layer: None,
            pending_label: None,
        };
        let mut binding = TaskBinding::default();
        binding.arguments[0].base = &raw mut *result.frame as u64;
        result.label("block-begin");
        result.once(GeneratedKernel::ret()?, binding)?;
        Ok(result)
    }
    /// Set the semantic name consumed by the next emitted node.
    pub fn label(&mut self, operation: &'static str) {
        self.pending_label = Some(operation);
    }
    /// Associate subsequent nodes with one model layer.
    pub fn begin_layer(&mut self, layer: usize) {
        self.layer = Some(layer);
    }
    /// Contexts contain plain data only; word storage preserves eight-byte alignment.
    pub fn keep<T: Copy>(&mut self, value: T) -> *mut T {
        assert!(align_of::<T>() <= align_of::<u64>());
        let mut storage = vec![0u64; size_of::<T>().div_ceil(8).max(1)].into_boxed_slice();
        let ptr = storage.as_mut_ptr().cast::<T>();
        unsafe {
            ptr.write(value);
        }
        self.storage.push(storage);
        ptr
    }
    pub fn once(
        &mut self,
        kernel: GeneratedKernel,
        mut binding: TaskBinding,
    ) -> Result<(), ExecutionError> {
        binding.tasks = 1;
        if self.final_only {
            self.grid(
                kernel,
                binding,
                TaskGrid {
                    rows_per_task: self.batch,
                    tasks_per_tile: 1,
                    last_block_only: true,
                },
                1,
                1,
            )
        } else {
            self.push(kernel, binding, None, 1, 1)
        }
    }
    pub fn rows(
        &mut self,
        kernel: GeneratedKernel,
        binding: TaskBinding,
        workers: usize,
        workers_m1: usize,
    ) -> Result<(), ExecutionError> {
        self.grid(
            kernel,
            binding,
            TaskGrid {
                rows_per_task: 1,
                tasks_per_tile: 1,
                last_block_only: false,
            },
            workers,
            workers_m1,
        )
    }
    pub fn grid(
        &mut self,
        kernel: GeneratedKernel,
        binding: TaskBinding,
        mut grid: TaskGrid,
        workers: usize,
        workers_m1: usize,
    ) -> Result<(), ExecutionError> {
        grid.last_block_only |= self.final_only;
        grid.capacity(self.batch)?;
        self.push(kernel, binding, Some(grid), workers, workers_m1)
    }
    fn push(
        &mut self,
        kernel: GeneratedKernel,
        mut binding: TaskBinding,
        grid: Option<TaskGrid>,
        workers: usize,
        workers_m1: usize,
    ) -> Result<(), ExecutionError> {
        if workers == 0 || workers_m1 == 0 || workers_m1 > workers {
            return Err(ExecutionError("invalid mounted worker policy".into()));
        }
        let max_tasks = grid.map_or(Ok(binding.tasks), |g| g.capacity(self.batch))?;
        binding.worker_rows = &raw const self.frame.rows;
        binding.worker_limit_m1 = workers_m1;
        let index = self.nodes.len();
        self.nodes.push(Node {
            operation: Operation::Kernel {
                kernel: self.kernels.len(),
                max_tasks,
                worker_limit: workers,
                distribution: Distribution::Contiguous,
            },
            dependencies: if index == 0 { vec![] } else { vec![index - 1] },
        });
        self.labels.push(NodeLabel {
            operation: self.pending_label.take().unwrap_or("kernel"),
            layer: self.layer,
        });
        self.kernels.push(kernel);
        self.records.push(binding);
        self.grid_of.push(grid.map(|g| {
            self.grids.push(g);
            self.grids.len() - 1
        }));
        Ok(())
    }
    pub fn final_only(&mut self, enabled: bool) {
        self.final_only = enabled;
    }
    pub fn layer_end(&mut self, layer: usize) {
        self.nodes.push(Node {
            operation: Operation::LayerEnd { layer },
            dependencies: vec![self.nodes.len() - 1],
        });
        self.labels.push(NodeLabel {
            operation: "layer-end",
            layer: Some(layer),
        });
        self.records.push(TaskBinding::default());
        self.grid_of.push(None);
        self.layer = None;
    }
    pub fn finish_with_instrumentation(
        mut self,
        max_blocks: usize,
        spin: Option<u32>,
        trace_sync: bool,
        trace_pmu: bool,
        shared_barrier: bool,
        fuse_serial: bool,
    ) -> Result<Mounted, ExecutionError> {
        self.final_only = false;
        let mut binding = TaskBinding::default();
        binding.arguments[0].base = &raw mut *self.frame as u64;
        self.label("block-end");
        self.once(Native::end_block()?, binding)?;
        self.kernels[0] = Native::begin_block(self.batch, &self.grids)?;
        let mut counts = vec![0usize; self.grids.len()].into_boxed_slice();
        self.frame.counts = counts.as_mut_ptr();
        for (record, grid) in self.records.iter_mut().zip(&self.grid_of) {
            if let Some(index) = grid {
                record.task_count = unsafe { counts.as_ptr().add(*index) };
            }
        }
        let plan = ExecutionPlan::new_labeled(self.kernels, self.nodes, self.labels)?;
        let mut bindings = Bindings::repeated(&plan, max_blocks)?;
        bindings.block_mut(0).copy_from_slice(&self.records);
        Ok(Mounted {
            program: CompiledExecution::compile_configured(
                plan,
                spin,
                trace_sync,
                trace_pmu,
                shared_barrier,
                fuse_serial,
            )?,
            bindings,
            frame: self.frame,
            _counts: counts,
            _storage: self.storage,
        })
    }
}

pub(crate) struct Mounted {
    pub program: CompiledExecution,
    pub bindings: Bindings,
    pub frame: Box<CallFrame>,
    _counts: Box<[usize]>,
    _storage: Vec<Box<[u64]>>,
}
