//! A server's plugins: what it computes for a domain the stream knows nothing of (histology's
//! measurements, a model's run). A client sends a [`crate::protocol::Request::Plugin`] (plugin id,
//! method, JSON body); the server hands it to the plugin of that id with what it may use of the
//! server, and sends back the plugin's JSON. Plugins are registered by whoever builds the server
//! (`lucem serve`), as dsp-kitchen's `dspk serve` registers its Synapse plugin.

use std::path::PathBuf;

use serde_json::Value;

/// What a plugin call may use of the server.
pub struct PluginContext {
    /// The data folder: paths a client names are under it.
    pub root: PathBuf,
    /// The client, by the name it was paired under.
    pub peer: String,
    /// The client may change files (paired with control): a plugin refuses writes otherwise.
    pub may_write: bool,
}

impl PluginContext {
    /// `relative` under the data folder, checked: no `..`, nothing hidden, not empty.
    pub fn resolve(&self, relative: &str) -> Result<PathBuf, String> {
        let path = std::path::Path::new(relative);
        let plain = path.components().all(|c| matches!(c, std::path::Component::Normal(n) if !n.to_string_lossy().starts_with('.')));
        if relative.is_empty() || !plain {
            return Err(format!("{relative:?}: that path is not allowed"));
        }
        Ok(self.root.join(path))
    }
}

/// One plugin of a server.
pub trait ServerPlugin: Send + Sync {
    /// Its id, as clients name it (`"hist"`).
    fn id(&self) -> &'static str;

    /// Answers `method` with `body`; run on a blocking thread (it may compute for seconds).
    fn call(&self, method: &str, body: Value, context: &PluginContext) -> Result<Value, String>;
}
