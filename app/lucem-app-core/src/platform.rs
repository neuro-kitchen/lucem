//! What differs between the desktop and a browser tab, in one place (as dsp-kitchen's
//! `platform`): the rest of the app calls these and has no `cfg` of its own for them.

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

/// Whether this build can wait for an answer on a worker thread (a browser build with threads,
/// or any computer).
pub const THREADS: bool = cfg!(any(not(target_family = "wasm"), target_feature = "atomics"));
