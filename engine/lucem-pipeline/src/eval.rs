//! Runs a graph of ops: [`evaluate`] computes one node's value from its inputs, recursively, each
//! node once (memoized), with the source rasters given by the caller (an image region, a mask
//! drawn by hand, …). The caller builds the plan from its own document, so this crate knows
//! nothing about projects or the app.

use std::collections::{BTreeMap, HashMap};

use lucem_core::cubecl::prelude::Client;

use crate::ops::{Op, PortType, Value};

/// One node of a plan.
#[derive(Debug, Clone)]
pub struct PlanNode {
    /// `None`: a source; its value comes from the caller.
    pub op: Option<Op>,
    /// Off: the node passes its first input through (when the types allow it).
    pub enabled: bool,
    /// The node feeding each input port.
    pub inputs: Vec<Option<String>>,
}

/// The graph to run, by node id.
pub type Plan = BTreeMap<String, PlanNode>;

/// The value of `target`; `source(id)` gives each source node's value.
pub fn evaluate(client: &Client, plan: &Plan, target: &str, source: &mut dyn FnMut(&str) -> Result<Value, String>) -> Result<Value, String> {
    let mut memo: HashMap<String, Value> = HashMap::new();
    eval(client, plan, target, source, &mut memo, 0)
}

fn eval(client: &Client, plan: &Plan, id: &str, source: &mut dyn FnMut(&str) -> Result<Value, String>, memo: &mut HashMap<String, Value>, depth: usize) -> Result<Value, String> {
    if let Some(v) = memo.get(id) {
        return Ok(v.clone());
    }
    if depth > plan.len() + 1 {
        return Err("the graph has a loop".into());
    }
    let node = plan.get(id).ok_or_else(|| format!("no node {id:?}"))?;
    let value = match &node.op {
        None => source(id)?,
        Some(op) => {
            let ports = op.inputs();
            let mut inputs = Vec::with_capacity(ports.len());
            for (i, (want, name)) in ports.iter().enumerate() {
                let from = node.inputs.get(i).cloned().flatten().ok_or_else(|| format!("{} ({id}): its {name} input is not connected", op.title()))?;
                let v = eval(client, plan, &from, source, memo, depth + 1)?;
                if v.port() != *want {
                    return Err(format!("{} ({id}): its {name} input takes {}, {from} gives {}", op.title(), want.name(), v.port().name()));
                }
                inputs.push(v);
            }
            if !node.enabled {
                match inputs.into_iter().next() {
                    Some(v) if v.port() == op.output() => v,
                    _ => return Err(format!("{} ({id}) is off, and its input is not a {}: turn it on or remove it", op.title(), op.output().name())),
                }
            } else {
                // Inputs of one op must be the same size (a region of one image)
                let dims: Vec<(usize, usize)> = inputs.iter().filter_map(Value::dims).collect();
                if dims.windows(2).any(|w| w[0] != w[1]) {
                    return Err(format!("{} ({id}): its inputs differ in size", op.title()));
                }
                op.run(client, &inputs)?
            }
        }
    };
    memo.insert(id.to_string(), value.clone());
    Ok(value)
}

/// Every node of `targets` (and what they need), each computed once; a node that fails gives its
/// error and the others go on.
pub fn evaluate_all(client: &Client, plan: &Plan, targets: &[String], source: &mut dyn FnMut(&str) -> Result<Value, String>) -> BTreeMap<String, Result<Value, String>> {
    let mut memo: HashMap<String, Value> = HashMap::new();
    let mut out = BTreeMap::new();
    for t in targets {
        let r = eval(client, plan, t, source, &mut memo, 0);
        out.insert(t.clone(), r);
    }
    out
}

/// The plan with every op set for a level of downsample `ds` (see [`Op::at_downsample`]).
pub fn at_downsample(plan: &Plan, ds: f64) -> Plan {
    plan.iter()
        .map(|(id, n)| (id.clone(), PlanNode { op: n.op.as_ref().map(|o| o.at_downsample(ds)), enabled: n.enabled, inputs: n.inputs.clone() }))
        .collect()
}

/// Pixels of context `target` needs around a region: the largest sum of halos on any path to a
/// source.
pub fn halo(plan: &Plan, target: &str) -> usize {
    fn walk(plan: &Plan, id: &str, depth: usize) -> usize {
        if depth > plan.len() {
            return 0;
        }
        let Some(n) = plan.get(id) else { return 0 };
        let own = match (&n.op, n.enabled) {
            (Some(op), true) => op.halo(),
            _ => 0,
        };
        own + n.inputs.iter().flatten().map(|i| walk(plan, i, depth + 1)).max().unwrap_or(0)
    }
    walk(plan, target, 0)
}

/// What `id` gives (sources: `source_type`).
pub fn output_type(plan: &Plan, id: &str, source_type: &dyn Fn(&str) -> Option<PortType>) -> Option<PortType> {
    let n = plan.get(id)?;
    match &n.op {
        None => source_type(id),
        Some(op) => Some(op.output()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halos_add_along_the_longest_path() {
        let mut plan = Plan::new();
        plan.insert("src".into(), PlanNode { op: None, enabled: true, inputs: vec![] });
        plan.insert("blur".into(), PlanNode { op: Some(Op::GaussianBlur { sigma: 2.0 }), enabled: true, inputs: vec![Some("src".into())] });
        plan.insert("otsu".into(), PlanNode { op: Some(Op::OtsuThreshold { bins: 256 }), enabled: true, inputs: vec![Some("blur".into())] });
        plan.insert("open".into(), PlanNode { op: Some(Op::Open { radius: 3 }), enabled: true, inputs: vec![Some("otsu".into())] });
        assert_eq!(halo(&plan, "open"), 8 + 6);
        plan.get_mut("blur").unwrap().enabled = false;
        assert_eq!(halo(&plan, "open"), 6);
    }
}
