//! The trace of one generated call, written as JSON.
//!
//! When a regime was mounted with `Measure::sync_trace` or `Measure::pmu`, every
//! node of every block records, per participant, its `rdtscp` stamps (work,
//! progress, barrier, sleep) and, for the PMU event, the `rdpmc` delta. This
//! writes the latest traced call of one regime: the plan's nodes (operation,
//! layer, kernel, task count), then the records laid out `[block][node][participant]`.
//! Nothing is written when the call was not traced.
use crate::jit::execution::Operation;
use crate::jit::Runner;
use serde_json::json;
use std::path::Path;

pub fn write_execution_trace(
    runner: &Runner,
    prefill: bool,
    path: &Path,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(plan) = runner.execution_plan(prefill) else {
        return Ok(());
    };
    let trace = runner.execution_sync_trace();
    let pmu = runner.execution_pmu_trace();
    if trace.is_none() && pmu.is_none() {
        return Ok(());
    }
    let nodes: Vec<_> = plan
        .nodes()
        .iter()
        .enumerate()
        .zip(plan.labels())
        .map(|((index, node), label)| match node.operation {
            Operation::Kernel { kernel, max_tasks, worker_limit, distribution } => json!({
                "index":index,"kind":"kernel","operation":label.operation,"layer":label.layer,
                "kernel":kernel,"max_tasks":max_tasks,"worker_limit":worker_limit,
                "distribution":format!("{distribution:?}")
            }),
            Operation::LayerEnd { layer } => {
                json!({"index":index,"kind":"layer_end","operation":label.operation,"layer":layer})
            }
        })
        .collect();
    let records = trace.as_ref().map(|trace| {
        trace
            .records
            .iter()
            .enumerate()
            .map(|(index, r)| {
                let participant = index % trace.participants;
                let slot = index / trace.participants;
                json!({
                    "block":slot / trace.nodes_per_block,
                    "node":slot % trace.nodes_per_block,
                    "participant":participant,
                    "work_start":r.work_start,"work_end":r.work_end,
                    "progress_start":r.progress_start,"progress_end":r.progress_end,
                    "barrier_start":r.barrier_start,"sleep_start":r.sleep_start,
                    "sleep_end":r.sleep_end,"barrier_end":r.barrier_end,
                    "work_start_aux":r.work_start_aux,"work_end_aux":r.work_end_aux,
                    "progress_start_aux":r.progress_start_aux,"progress_end_aux":r.progress_end_aux,
                    "barrier_start_aux":r.barrier_start_aux,"sleep_start_aux":r.sleep_start_aux,
                    "sleep_end_aux":r.sleep_end_aux,"barrier_end_aux":r.barrier_end_aux
                })
            })
            .collect::<Vec<_>>()
    });
    let pmu_records = pmu.as_ref().map(|trace| {
        trace
            .records
            .iter()
            .enumerate()
            .map(|(index, r)| {
                let participant = index % trace.participants;
                let slot = index / trace.participants;
                json!({
                    "block":slot / trace.nodes_per_block,
                    "node":slot % trace.nodes_per_block,
                    "participant":participant,
                    "work_start":r.work_start,"work_end":r.work_end,
                    "progress_start":r.progress_start,"progress_end":r.progress_end,
                    "barrier_start":r.barrier_start,"sleep_start":r.sleep_start,
                    "sleep_end":r.sleep_end,"barrier_end":r.barrier_end
                })
            })
            .collect::<Vec<_>>()
    });
    let blocks = trace.as_ref().map(|t| t.blocks).or_else(|| pmu.as_ref().map(|t| t.blocks));
    let nodes_per_block = trace
        .as_ref()
        .map(|t| t.nodes_per_block)
        .or_else(|| pmu.as_ref().map(|t| t.nodes_per_block));
    let participants = trace
        .as_ref()
        .map(|t| t.participants)
        .or_else(|| pmu.as_ref().map(|t| t.participants));
    let barrier_stats = runner.execution_barriers(prefill);
    let file = std::fs::File::create(path)?;
    serde_json::to_writer(
        std::io::BufWriter::new(file),
        &json!({
            "clock":trace.as_ref().map(|_| "rdtscp"),
            "pmu_event":pmu.as_ref().map(|trace| trace.event.to_string()),
            "layout":"records are [block][node][participant]; final node is block progress",
            "blocks":blocks,
            "nodes_per_block":nodes_per_block,
            "participants":participants,
            "barrier_stats":barrier_stats,
            "nodes":nodes,
            "records":records,
            "pmu_records":pmu_records
        }),
    )?;
    Ok(())
}
