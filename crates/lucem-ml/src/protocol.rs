//! Requests and events between lucem and a plugin: one JSON object per line. lucem writes a
//! [`Request`] to the program's stdin; the program answers with [`Event`]s on its stdout,
//! `progress` any number of times, then exactly one `done` or `error`. Rasters travel as raw files
//! (row by row, no header) named in the request, so no image goes through JSON.

use serde::{Deserialize, Serialize};

/// A raster in a file: `width × height × channels` bytes, row by row (RGB for images, one byte
/// per pixel for masks).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RasterRef {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

impl RasterRef {
    pub fn len(&self) -> usize {
        self.width as usize * self.height as usize * self.channels as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A prompt, in pixels of the image sent.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Prompt {
    /// A point on (`positive`) or off the object.
    Point { x: f32, y: f32, positive: bool },
    /// A box around the object.
    Box { x0: f32, y0: f32, x1: f32, y1: f32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
pub enum Request {
    /// The plugin's manifest (as `<program> info`).
    Info,
    /// Downloads and verifies a model's weights.
    Install { model: String },
    /// Masks of the object the prompts point at. `image_key` names the image's content: the same
    /// key lets the plugin reuse what it computed for that image (SAM's embedding).
    Segment {
        model: String,
        image: RasterRef,
        image_key: String,
        prompts: Vec<Prompt>,
        /// Several candidate masks (else the best one).
        multimask: bool,
        /// Where to write the masks.
        out_dir: String,
    },
    /// Every object of an image as labels (nuclei, cells): a `u32` raster, `0` background.
    /// `pixel_um` lets the model work at the scale it was trained for.
    Instances {
        model: String,
        image: RasterRef,
        #[serde(default)]
        pixel_um: Option<f64>,
        out_dir: String,
    },
    /// Ends a server.
    Quit,
}

/// A label's class, when the model classifies what it finds (`Neoplastic`, `Inflammatory`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstanceClass {
    pub label: u32,
    pub class: String,
    #[serde(default)]
    pub score: Option<f32>,
}

/// The answer to [`Request::Instances`]: the labels (`channels` 1, 4 bytes per pixel,
/// little-endian `u32`) and the classes of those it classified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstancesResult {
    pub labels: RasterRef,
    #[serde(default)]
    pub classes: Vec<InstanceClass>,
    #[serde(default)]
    pub ms: f64,
}

/// A mask the plugin wrote, and how sure it is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskOut {
    pub raster: RasterRef,
    pub score: f32,
}

/// The answer to [`Request::Segment`], best first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentResult {
    pub masks: Vec<MaskOut>,
    /// Milliseconds spent encoding the image (0 when reused) and decoding the prompts.
    #[serde(default)]
    pub encode_ms: f64,
    #[serde(default)]
    pub decode_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Progress {
        stage: String,
        /// 0–1 when known.
        #[serde(default)]
        fraction: Option<f32>,
    },
    /// The request's answer (its JSON depends on the request).
    Done { result: serde_json::Value },
    Error { message: String },
}

impl Event {
    /// One line of the protocol.
    pub fn line(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| format!("{{\"event\":\"error\",\"message\":\"{e}\"}}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_and_events_are_one_json_line() {
        let r = Request::Segment {
            model: "sam2-hiera-tiny".into(),
            image: RasterRef { path: "/tmp/i.rgb".into(), width: 4, height: 2, channels: 3 },
            image_key: "abc".into(),
            prompts: vec![Prompt::Point { x: 1.0, y: 1.0, positive: true }, Prompt::Box { x0: 0.0, y0: 0.0, x1: 3.0, y1: 1.0 }],
            multimask: false,
            out_dir: "/tmp".into(),
        };
        let line = serde_json::to_string(&r).unwrap();
        assert!(!line.contains('\n') && line.contains("\"request\":\"segment\""));
        assert_eq!(serde_json::from_str::<Request>(&line).unwrap(), r);
        let e = Event::Progress { stage: "encoding".into(), fraction: Some(0.5) };
        assert_eq!(serde_json::from_str::<Event>(&e.line()).unwrap(), e);
    }
}
