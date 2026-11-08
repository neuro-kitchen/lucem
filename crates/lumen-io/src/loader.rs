//! Reading the blocks of a walk ahead of the work on them.
//!
//! [`BlockLoader`] decodes on a few threads while the caller processes the current block, and hands
//! blocks back in walk order. At most `read_ahead` blocks wait in memory, so an image of any size
//! goes through a bounded amount of it.

use std::collections::BTreeMap;
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use lumen_core::Block;

use crate::error::IoResult;
use crate::pyramid::Pyramid;

/// One block of one channel.
#[derive(Debug, Clone)]
pub struct LoadedBlock {
    pub block: Block,
    pub channel: usize,
    /// The read part of the block, C-ordered.
    pub data: Vec<f32>,
}

struct Progress {
    next_job: usize,
    consumed: usize,
    stopped: bool,
}

struct Shared {
    progress: Mutex<Progress>,
    wake: Condvar,
}

/// Blocks in walk order: for each block, each channel in turn.
pub struct BlockLoader {
    shared: Arc<Shared>,
    rx: Receiver<(usize, IoResult<LoadedBlock>)>,
    waiting: BTreeMap<usize, IoResult<LoadedBlock>>,
    next: usize,
    total: usize,
    workers: Vec<JoinHandle<()>>,
}

impl BlockLoader {
    /// Starts reading `blocks` of `channels` at `level` on `threads` threads, holding at most
    /// `read_ahead` finished blocks.
    pub fn new(pyramid: Arc<Pyramid>, level: usize, channels: Vec<usize>, blocks: Vec<Block>, read_ahead: usize, threads: usize) -> Self {
        let total = blocks.len() * channels.len();
        let shared = Arc::new(Shared { progress: Mutex::new(Progress { next_job: 0, consumed: 0, stopped: false }), wake: Condvar::new() });
        let (tx, rx) = channel();
        let read_ahead = read_ahead.max(1);
        let jobs: Arc<Vec<(Block, usize)>> = Arc::new(blocks.iter().flat_map(|b| channels.iter().map(move |c| (*b, *c))).collect());
        let workers = (0..threads.clamp(1, 32))
            .map(|_| {
                let (pyramid, shared, tx, jobs) = (pyramid.clone(), shared.clone(), tx.clone(), jobs.clone());
                thread::spawn(move || loop {
                    let job = {
                        let mut p = shared.progress.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if p.stopped || p.next_job >= total {
                                return;
                            }
                            if p.next_job < p.consumed + read_ahead {
                                break;
                            }
                            p = shared.wake.wait(p).unwrap_or_else(|e| e.into_inner());
                        }
                        p.next_job += 1;
                        p.next_job - 1
                    };
                    let (block, channel) = jobs[job];
                    let read = pyramid.read_block(level, channel, block.read_origin, block.read_size).map(|data| LoadedBlock { block, channel, data });
                    if tx.send((job, read)).is_err() {
                        return;
                    }
                })
            })
            .collect();
        Self { shared, rx, waiting: BTreeMap::new(), next: 0, total, workers }
    }

    /// Blocks times channels.
    pub fn total(&self) -> usize {
        self.total
    }
}

impl Iterator for BlockLoader {
    type Item = IoResult<LoadedBlock>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.total {
            return None;
        }
        let item = loop {
            if let Some(found) = self.waiting.remove(&self.next) {
                break found;
            }
            let (i, read) = self.rx.recv().ok()?;
            self.waiting.insert(i, read);
        };
        self.next += 1;
        self.shared.progress.lock().unwrap_or_else(|e| e.into_inner()).consumed = self.next;
        self.shared.wake.notify_all();
        Some(item)
    }
}

impl Drop for BlockLoader {
    fn drop(&mut self) {
        self.shared.progress.lock().unwrap_or_else(|e| e.into_inner()).stopped = true;
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use lumen_core::tiling::schedule;

    use super::*;
    use crate::pyramid::ChannelInfo;

    #[test]
    fn blocks_arrive_in_order_whatever_thread_read_them() {
        let (d, h, w) = (4usize, 64usize, 64usize);
        let info: Vec<ChannelInfo> = (0..2).map(|_| ChannelInfo { label: "c".into(), color: [255, 255, 255], window: (0.0, 1.0), range: (0.0, 1.0) }).collect();
        let a: Vec<f32> = (0..d * h * w).map(|i| i as f32).collect();
        let b: Vec<f32> = a.iter().map(|v| v + 0.5).collect();
        let py = Arc::new(Pyramid::from_volume("v", info, vec![a, b], [d, h, w], None, None, "float32").unwrap());
        let blocks = schedule([d, h, w], [2, 16, 16], [0, 2, 2]);
        let loader = BlockLoader::new(py.clone(), 0, vec![0, 1], blocks.clone(), 3, 4);
        assert_eq!(loader.total(), blocks.len() * 2);
        for (n, got) in loader.enumerate() {
            let got = got.unwrap();
            assert_eq!(got.block, blocks[n / 2]);
            assert_eq!(got.channel, n % 2);
            assert_eq!(got.data, py.read_block(0, got.channel, got.block.read_origin, got.block.read_size).unwrap());
        }
    }

    #[test]
    fn dropping_a_loader_early_stops_its_threads() {
        let info = vec![ChannelInfo { label: "c".into(), color: [255, 255, 255], window: (0.0, 1.0), range: (0.0, 1.0) }];
        let py = Arc::new(Pyramid::from_volume("v", info, vec![vec![0.0; 256]], [4, 8, 8], None, None, "float32").unwrap());
        let mut loader = BlockLoader::new(py, 0, vec![0], schedule([4, 8, 8], [1, 2, 2], [0; 3]), 1, 2);
        assert!(loader.next().is_some());
        drop(loader);
    }
}
