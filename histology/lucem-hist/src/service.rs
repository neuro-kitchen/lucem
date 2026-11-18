//! Histology as a service: what the app computes, answered by method with JSON, transport-free.
//! A server registers it as its `hist` plugin (`lucem serve`) so a page computes on the server;
//! the app calls the same functions locally. As dsp-kitchen's `curation::service`.
//!
//! Each image keeps its [`Analysis`] (what the steps so far left) for the server's life.
//!
//! | method | body | answer |
//! |---|---|---|
//! | `apply` | `{path, z?, action}` (a session [`Action`]) | `{outcome, tissue, cells, tiles}` |
//! | `summary` | `{path}` | `{tissue, cells, tiles}` |
//! | `objects` | `{path, kind: "cells" \| "tiles", offset?, limit?}` | `{total, objects: [HistObject…]}` |
//! | `reset` | `{path}` | `{}`: the image's analysis is forgotten |

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::engine::{apply, Analysis, Ctx, NoModels};
use crate::session::Action;

/// The plugin id a server registers it under.
pub const PLUGIN: &str = "hist";

/// Objects per `objects` answer, at most.
pub const OBJECTS_MAX: usize = 50_000;

/// Who calls: their name as the server knows it, and whether they may write.
#[derive(Debug, Clone, Copy)]
pub struct Caller<'a> {
    pub name: &'a str,
    pub may_write: bool,
}

#[derive(Deserialize)]
struct ApplyCall {
    path: String,
    #[serde(default)]
    z: usize,
    action: Action,
    /// The caller's tissue regions (level-0 outlines), when it keeps them (the app's `Tissue`
    /// annotations, edited by people): they replace what the server kept.
    #[serde(default)]
    tissue: Option<Vec<crate::objects::Polygon>>,
}

#[derive(Deserialize)]
struct PathCall {
    path: String,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Cells,
    Tiles,
}

#[derive(Deserialize)]
struct ObjectsCall {
    path: String,
    kind: Kind,
    #[serde(default)]
    offset: usize,
    #[serde(default = "objects_max")]
    limit: usize,
}

fn objects_max() -> usize {
    OBJECTS_MAX
}

/// The images' analyses, kept between calls.
#[derive(Default)]
pub struct Service {
    analyses: Mutex<HashMap<PathBuf, Analysis>>,
}

/// A body that does not read, said with the method's name.
fn bad(method: &'static str) -> impl Fn(serde_json::Error) -> String {
    move |e| format!("{method}: {e}")
}

fn counts(a: &Analysis) -> Value {
    json!({"tissue": a.tissue.len(), "cells": a.cells.len(), "tiles": a.tiles.len()})
}

impl Service {
    /// Answers `method` with `body`; `resolve` turns a path the caller names into one it may read
    /// (the server's data folder; refused outside it).
    pub fn call(&self, method: &str, body: Value, resolve: &dyn Fn(&str) -> Result<PathBuf, String>, _caller: Caller) -> Result<Value, String> {
        match method {
            "apply" => {
                let call: ApplyCall = serde_json::from_value(body).map_err(bad("apply"))?;
                let path = resolve(&call.path)?;
                let pyramid = lucem_io::Pyramid::open(&path).map_err(|e| format!("{}: {e}", call.path))?;
                let client = lucem_core::ComputeTarget::from_env().and_then(|t| t.client()).map_err(|e| e.to_string())?;
                // The analysis is taken out while the step runs: other images go on meanwhile
                let mut analysis = self.analyses.lock().unwrap_or_else(|e| e.into_inner()).remove(&path).unwrap_or_default();
                if let Some(tissue) = call.tissue {
                    analysis.tissue = tissue;
                }
                let (mut models, cancel) = (NoModels, AtomicBool::new(false));
                let mut progress = |_: &str, _: Option<f32>| {};
                let mut ctx = Ctx { client: &client, pyramid: &pyramid, z: call.z, models: &mut models, progress: &mut progress, cancel: &cancel };
                let outcome = apply(&mut ctx, &mut analysis, &call.action);
                let mut answer = counts(&analysis);
                // The regions themselves (few, small): the caller shows them as annotations
                answer["regions"] = json!(analysis.tissue);
                self.analyses.lock().unwrap_or_else(|e| e.into_inner()).insert(path, analysis);
                answer["outcome"] = serde_json::to_value(outcome.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                Ok(answer)
            }
            "summary" => {
                let call: PathCall = serde_json::from_value(body).map_err(bad("summary"))?;
                let path = resolve(&call.path)?;
                Ok(self.analyses.lock().unwrap_or_else(|e| e.into_inner()).get(&path).map_or_else(|| counts(&Analysis::default()), counts))
            }
            "objects" => {
                let call: ObjectsCall = serde_json::from_value(body).map_err(bad("objects"))?;
                let path = resolve(&call.path)?;
                let analyses = self.analyses.lock().unwrap_or_else(|e| e.into_inner());
                let all = analyses.get(&path).map_or(&[][..], |a| if call.kind == Kind::Cells { &a.cells[..] } else { &a.tiles[..] });
                let page = all.iter().skip(call.offset).take(call.limit.min(OBJECTS_MAX));
                Ok(json!({"total": all.len(), "objects": page.collect::<Vec<_>>()}))
            }
            "reset" => {
                let call: PathCall = serde_json::from_value(body).map_err(bad("reset"))?;
                let path = resolve(&call.path)?;
                self.analyses.lock().unwrap_or_else(|e| e.into_inner()).remove(&path);
                Ok(json!({}))
            }
            other => Err(format!("the histology service has no method {other:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calls are read by method; an image with no analysis yet answers empty; paths go through
    /// the caller's `resolve` (refused there, refused here); unknown methods and bodies are refused.
    #[test]
    fn methods_answer_and_refuse() {
        let service = Service::default();
        let resolve = |p: &str| -> Result<PathBuf, String> { if p.contains("..") { Err("not allowed".into()) } else { Ok(PathBuf::from("/data").join(p)) } };
        let caller = Caller { name: "test", may_write: false };
        assert_eq!(service.call("summary", json!({"path": "a.ome.zarr"}), &resolve, caller).unwrap(), json!({"tissue": 0, "cells": 0, "tiles": 0}));
        assert_eq!(service.call("objects", json!({"path": "a.ome.zarr", "kind": "cells"}), &resolve, caller).unwrap(), json!({"total": 0, "objects": []}));
        assert_eq!(service.call("reset", json!({"path": "a.ome.zarr"}), &resolve, caller).unwrap(), json!({}));
        assert!(service.call("summary", json!({"path": "../x"}), &resolve, caller).is_err());
        assert!(service.call("objects", json!({"path": "a", "kind": "nuclei"}), &resolve, caller).is_err(), "no such kind");
        assert!(service.call("nothing", json!({}), &resolve, caller).is_err());
    }
}
