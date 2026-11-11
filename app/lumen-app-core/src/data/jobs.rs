//! What runs in the background, in one place: each job has a kind (`histology`, `model`,
//! `export`…), a name saying what it is doing, how far it is, and a cancel flag. Only one job
//! that drives the GPU runs at a time (a second is refused, saying what runs). The status bar
//! and the panels read it; the work itself (off the UI thread) updates it through its
//! [`JobHandle`], which ends the job when dropped (finished, failed or cancelled alike).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// A job running.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: u64,
    pub kind: &'static str,
    /// What it is doing now ("Finding cells: tile 3 of 12").
    pub name: String,
    /// How far (0..1), when known.
    pub progress: Option<f32>,
    /// Whether it drives the GPU (one such job at a time).
    pub gpu: bool,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Default)]
struct Inner {
    list: Vec<Job>,
    next: u64,
}

fn inner() -> &'static Mutex<Inner> {
    static J: OnceLock<Mutex<Inner>> = OnceLock::new();
    J.get_or_init(Default::default)
}

fn with<T>(f: impl FnOnce(&mut Inner) -> T) -> T {
    let mut g = inner().lock().unwrap_or_else(|e| e.into_inner());
    f(&mut g)
}

/// Starts a job; refused while another GPU job runs (when this one is one too).
pub fn start(kind: &'static str, name: &str, gpu: bool) -> Result<JobHandle, String> {
    with(|j| {
        if gpu && let Some(other) = j.list.iter().find(|o| o.gpu) {
            return Err(format!("{} is still running", other.name));
        }
        j.next += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        j.list.push(Job { id: j.next, kind, name: name.to_string(), progress: None, gpu, cancel: cancel.clone() });
        Ok(JobHandle { id: j.next, cancel })
    })
}

/// The jobs running, oldest first.
pub fn running() -> Vec<Job> {
    with(|j| j.list.clone())
}

/// The latest job of `kind`.
pub fn of(kind: &str) -> Option<Job> {
    with(|j| j.list.iter().rev().find(|o| o.kind == kind).cloned())
}

/// Asks every job of `kind` to stop (after its current tile).
pub fn cancel(kind: &str) {
    with(|j| j.list.iter().filter(|o| o.kind == kind).for_each(|o| o.cancel.store(true, Ordering::Relaxed)));
}

/// A running job, held by the work doing it.
pub struct JobHandle {
    pub id: u64,
    pub cancel: Arc<AtomicBool>,
}

impl JobHandle {
    /// Says what it is doing now, and how far.
    pub fn set(&self, name: &str, progress: Option<f32>) {
        with(|j| {
            if let Some(o) = j.list.iter_mut().find(|o| o.id == self.id) {
                o.name = name.to_string();
                o.progress = progress;
            }
        });
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        let id = self.id;
        with(|j| j.list.retain(|o| o.id != id));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn one_gpu_job_at_a_time_and_a_job_ends_with_its_handle() {
        let a = super::start("test-a", "first", true).unwrap();
        assert!(super::start("test-b", "second", true).is_err(), "a second GPU job is refused");
        let cpu = super::start("test-c", "on the CPU", false).unwrap();
        a.set("first: half", Some(0.5));
        assert_eq!(super::of("test-a").unwrap().progress, Some(0.5));
        super::cancel("test-a");
        assert!(a.cancelled());
        drop(a);
        assert!(super::of("test-a").is_none());
        assert!(super::start("test-b", "second", true).is_ok(), "free again");
        drop(cpu);
    }
}
