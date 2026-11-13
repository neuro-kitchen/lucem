//! What runs in the background: kitchen-shell's job registry (it began as this module), so the
//! frame's status bar lists every job with its progress and a cancel button. Each job has a kind
//! (`histology`, `model`, `export`…), a name saying what it is doing, how far it is, and a cancel
//! flag; one job that drives the GPU at a time; a job ends when its [`JobHandle`] drops.

pub use kitchen_shell::jobs::*;
