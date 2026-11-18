//! What differs between the desktop and a browser tab, in one place (as dsp-kitchen's
//! `platform`, and Kotlin Multiplatform's `expect` / `actual`): each target's file gives the same
//! names, and the rest of the app calls them and has no `cfg` of its own for them.
//!
//! | | desktop | browser, served (container) | browser, alone (docs viewer) |
//! |---|---|---|---|
//! | threads | the system's | web workers | web workers |
//! | a lock on the UI thread | waits | may not wait: spins | spins |
//! | files | the disk, a file dialog | the server's | files the user picks, read by the engine worker |
//! | servers | QUIC, paired servers | the page's own (WebTransport, WebSocket) | none |
//! | compute | this computer's GPU | the server's | none yet |
//! | the session | the config folder | browser storage | browser storage |

#[cfg(not(target_family = "wasm"))]
mod desktop;
#[cfg(not(target_family = "wasm"))]
pub use desktop::*;

#[cfg(target_family = "wasm")]
mod web;
#[cfg(target_family = "wasm")]
pub use web::*;

/// Served images on this target (`crate::data::remote` re-exports it).
#[cfg(not(target_family = "wasm"))]
#[path = "desktop_remote.rs"]
pub mod remote;
#[cfg(target_family = "wasm")]
#[path = "web_remote.rs"]
pub mod remote;

use std::sync::{Mutex, MutexGuard};

/// Locks `mutex`. On a browser page's main thread a lock may not wait (the page would stop:
/// `Atomics.wait` traps there), so there it is tried until it is free; the app's shared locks
/// are held for a few instructions. Elsewhere, a plain lock (a poisoned one still opens: what it
/// holds is caches and queues, not invariants). Use this, never `Mutex::lock` (clippy.toml).
#[allow(clippy::disallowed_methods)]
pub fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    if cfg!(target_family = "wasm") {
        loop {
            match mutex.try_lock() {
                Ok(guard) => return guard,
                Err(std::sync::TryLockError::Poisoned(e)) => return e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => std::hint::spin_loop(),
            }
        }
    }
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether files are opened from this computer's disk (a file dialog), as the desktop does; a
/// browser page opens its server's images instead.
pub const LOCAL_FILES: bool = cfg!(not(target_family = "wasm"));

/// Whether this build can wait for an answer on a worker thread (a browser build with threads,
/// or any computer).
pub const THREADS: bool = cfg!(any(not(target_family = "wasm"), target_feature = "atomics"));
