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

/// Whether files are opened from this computer's disk (a file dialog), as the desktop does; a
/// browser page opens its server's images instead.
pub const LOCAL_FILES: bool = cfg!(not(target_family = "wasm"));

/// Whether this build can wait for an answer on a worker thread (a browser build with threads,
/// or any computer).
pub const THREADS: bool = cfg!(any(not(target_family = "wasm"), target_feature = "atomics"));

/// Where the session is kept between runs: the desktop's config folder; a browser page's storage
/// (a served page: next, on its server under its sign-in).
const SESSION_FILE: &str = "current-session.lucem.json";
#[cfg(target_family = "wasm")]
const SESSION_KEY: &str = "lucem-session";

/// Keeps `json` (the session as it is now) for the next start; what went wrong, if anything.
pub fn keep_session(json: &str) -> Result<(), String> {
    #[cfg(not(target_family = "wasm"))]
    {
        let dir = crate::data::prefs::Prefs::dir().ok_or("no config folder")?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let file = dir.join(SESSION_FILE);
        let tmp = file.with_extension("json.tmp");
        std::fs::write(&tmp, json).and_then(|_| std::fs::rename(&tmp, &file)).map_err(|e| e.to_string())
    }
    #[cfg(target_family = "wasm")]
    {
        let _ = SESSION_FILE;
        let storage = web_sys::window().and_then(|w| w.local_storage().ok().flatten()).ok_or("no browser storage")?;
        storage.set_item(SESSION_KEY, json).map_err(|e| format!("the browser did not keep it: {e:?}"))
    }
}

/// The session kept by the last run, if any.
pub fn kept_session() -> Option<String> {
    #[cfg(not(target_family = "wasm"))]
    {
        std::fs::read_to_string(crate::data::prefs::Prefs::dir()?.join(SESSION_FILE)).ok()
    }
    #[cfg(target_family = "wasm")]
    {
        web_sys::window()?.local_storage().ok()??.get_item(SESSION_KEY).ok()?
    }
}
