//! The device memory budget: refuse an allocation that would not fit, instead of letting it fail.
//!
//! CubeCL 0.11.3 can run out of device memory **silently**: it logs a warning, skips work, and the
//! run ends with exit code 0 and wrong output (CorridorKey: an all-zero result that was *faster*
//! than smaller inputs). Large allocations are therefore checked here before they are made.
//!
//! CubeCL 0.11 reports what its own memory pools hold ([`memory_usage`]) but not the device's total
//! or free memory, so the limit is configured: `LUMEN_DEVICE_MEMORY_LIMIT` (e.g. `7GiB`, `6500MB`,
//! or bytes) or [`set_device_memory_limit`] (the app or CLI, e.g. from a driver query). Without a
//! limit, allocations are not checked. A vendor query (NVML on NVIDIA, as CorridorKey does) can set
//! the limit at start-up later.

use std::sync::{Mutex, Once};

use cubecl::prelude::*;
use cubecl::{MemoryScope, MemoryUsage};

use crate::error::{CoreError, CoreResult};

/// Environment variable: the device memory lumen may use, e.g. `7GiB`, `6500MB`, `7516192768`.
pub const DEVICE_MEMORY_LIMIT_ENV: &str = "LUMEN_DEVICE_MEMORY_LIMIT";

/// Allocations below this size are not checked: asking CubeCL for its memory report waits on the
/// device thread, which costs more than a small allocation is worth checking.
pub const CHECKED_ALLOCATION_BYTES: u64 = 64 << 20;

static LIMIT: Mutex<Option<u64>> = Mutex::new(None);
static FROM_ENV: Once = Once::new();

/// Sets the device memory limit in bytes (`None`: no checks). Overrides `LUMEN_DEVICE_MEMORY_LIMIT`.
pub fn set_device_memory_limit(bytes: Option<u64>) {
    read_env();
    *LIMIT.lock().unwrap_or_else(|e| e.into_inner()) = bytes;
}

/// The device memory limit in bytes, if one is set.
pub fn device_memory_limit() -> Option<u64> {
    read_env();
    *LIMIT.lock().unwrap_or_else(|e| e.into_inner())
}

/// What CubeCL's memory pools hold on `client`'s device (all streams).
pub fn memory_usage(client: &Client) -> MemoryUsage {
    client.memory_report(MemoryScope::Device).usage()
}

/// `Ok` if `bytes` more fit the limit next to what the device's pools already use; a no-op without a
/// limit or below [`CHECKED_ALLOCATION_BYTES`].
pub fn check_allocation(client: &Client, what: &str, bytes: u64) -> CoreResult<()> {
    let Some(limit) = device_memory_limit() else { return Ok(()) };
    if bytes < CHECKED_ALLOCATION_BYTES {
        return Ok(());
    }
    let in_use = memory_usage(client).bytes_in_use;
    let available = limit.saturating_sub(in_use);
    if bytes <= available {
        Ok(())
    } else {
        Err(CoreError::OutOfDeviceMemory { what: what.to_string(), requested: bytes, available, limit })
    }
}

fn read_env() {
    FROM_ENV.call_once(|| {
        if let Some(bytes) = std::env::var(DEVICE_MEMORY_LIMIT_ENV).ok().and_then(|v| parse_bytes(&v)) {
            *LIMIT.lock().unwrap_or_else(|e| e.into_inner()) = Some(bytes);
        }
    });
}

/// `"7GiB"`, `"6.5 GB"`, `"512MiB"`, `"1024"` → bytes (decimal units for kB/MB/GB, binary for
/// KiB/MiB/GiB). `None` for anything else.
fn parse_bytes(text: &str) -> Option<u64> {
    let text = text.trim();
    let split = text.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let value: f64 = number.parse().ok()?;
    let scale: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "kb" => 1e3,
        "mb" => 1e6,
        "gb" => 1e9,
        "kib" => 1024.0,
        "mib" => 1024.0 * 1024.0,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    (value >= 0.0).then(|| (value * scale) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes() {
        assert_eq!(parse_bytes("1024"), Some(1024));
        assert_eq!(parse_bytes("7GiB"), Some(7 << 30));
        assert_eq!(parse_bytes(" 6.5 GB "), Some(6_500_000_000));
        assert_eq!(parse_bytes("512mib"), Some(512 << 20));
        assert_eq!(parse_bytes("lots"), None);
        assert_eq!(parse_bytes("5 TB"), None);
    }
}
