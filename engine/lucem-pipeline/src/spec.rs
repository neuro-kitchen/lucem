//! A pipeline as data, for headless runs (`lucem run`) and for the record kept next to every
//! output: which channels feed it, its operators and what to write.
//!
//! ```json
//! {
//!   "sources": {"nuclei": 0},
//!   "nodes": {
//!     "blur": {"op": {"op": "gaussian_blur", "sigma": 2.0}, "inputs": ["nuclei"]},
//!     "mask": {"op": {"op": "otsu_threshold", "bins": 256}, "inputs": ["blur"]}
//!   },
//!   "target": "mask"
//! }
//! ```

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::eval::{Plan, PlanNode};
use crate::ops::Op;

fn yes() -> bool {
    true
}

/// One operator of a spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SpecNode {
    pub op: Op,
    /// The nodes or sources feeding each input port.
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default = "yes")]
    pub enabled: bool,
}

/// A pipeline to run over an image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    /// Source name to channel index.
    pub sources: BTreeMap<String, usize>,
    pub nodes: BTreeMap<String, SpecNode>,
    /// The node whose result is written.
    pub target: String,
    /// The pyramid level it runs on.
    #[serde(default)]
    pub level: usize,
    /// Scale each channel to 0–1 by its type's range, as the app does.
    #[serde(default = "yes")]
    pub normalize: bool,
}

impl PipelineSpec {
    /// The graph to evaluate.
    pub fn plan(&self) -> Plan {
        let mut plan = Plan::new();
        for id in self.sources.keys() {
            plan.insert(id.clone(), PlanNode { op: None, enabled: true, inputs: Vec::new() });
        }
        for (id, n) in &self.nodes {
            plan.insert(id.clone(), PlanNode { op: Some(n.op.clone()), enabled: n.enabled, inputs: n.inputs.iter().cloned().map(Some).collect() });
        }
        plan
    }

    /// Names resolve, parameters are valid, ports are all connected and no node feeds itself.
    pub fn validate(&self) -> Result<(), String> {
        if self.sources.is_empty() {
            return Err("a spec needs at least one source".into());
        }
        if let Some(id) = self.nodes.keys().find(|id| self.sources.contains_key(*id)) {
            return Err(format!("{id:?} is both a source and a node"));
        }
        if !self.nodes.contains_key(&self.target) && !self.sources.contains_key(&self.target) {
            return Err(format!("the target {:?} is not in the spec", self.target));
        }
        for (id, n) in &self.nodes {
            n.op.validate().map_err(|e| format!("{id}: {e}"))?;
            let ports = n.op.inputs().len();
            if n.inputs.len() != ports {
                return Err(format!("{id}: {} takes {ports} input(s), {} given", n.op.title(), n.inputs.len()));
            }
            if let Some(i) = n.inputs.iter().find(|i| !self.nodes.contains_key(*i) && !self.sources.contains_key(*i)) {
                return Err(format!("{id}: its input {i:?} is not in the spec"));
            }
        }
        // Depth-first over the nodes: a grey node met again is a loop
        let mut state: BTreeMap<&str, u8> = BTreeMap::new();
        fn visit<'a>(spec: &'a PipelineSpec, id: &'a str, state: &mut BTreeMap<&'a str, u8>) -> Result<(), String> {
            match state.get(id) {
                Some(1) => return Err(format!("the graph has a loop through {id:?}")),
                Some(_) => return Ok(()),
                None => {}
            }
            state.insert(id, 1);
            if let Some(n) = spec.nodes.get(id) {
                for i in &n.inputs {
                    visit(spec, i, state)?;
                }
            }
            state.insert(id, 2);
            Ok(())
        }
        for id in self.nodes.keys() {
            visit(self, id, &mut state)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> PipelineSpec {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn wrong_specs_are_refused_with_the_reason() {
        let missing = parse(r#"{"sources":{"a":0},"nodes":{"b":{"op":{"op":"sobel"},"inputs":["nope"]}},"target":"b"}"#);
        assert!(missing.validate().unwrap_err().contains("nope"));
        let looped = parse(r#"{"sources":{"a":0},"nodes":{"b":{"op":{"op":"sobel"},"inputs":["c"]},"c":{"op":{"op":"sobel"},"inputs":["b"]}},"target":"b"}"#);
        assert!(looped.validate().unwrap_err().contains("loop"));
        let ports = parse(r#"{"sources":{"a":0},"nodes":{"b":{"op":{"op":"combine","a":1,"b":1},"inputs":["a"]}},"target":"b"}"#);
        assert!(ports.validate().unwrap_err().contains("takes 2"));
        assert!(serde_json::from_str::<PipelineSpec>(r#"{"sources":{},"nodes":{},"target":"a","extra":1}"#).is_err());
    }
}
