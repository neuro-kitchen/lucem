//! Boundary padding modes for neighborhood operations.
//!
//! Kernels branch on [`PadMode::id`] as a `#[comptime] u32`, compiling a dedicated branch-free
//! kernel per boundary mode.

use serde::{Deserialize, Serialize};

use crate::error::{OpsError, OpsResult};

/// How a stencil or interpolation kernel reads coordinates outside `0..len` along an axis.
///
/// Shown for the left boundary of `a b c d`:
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PadMode {
    /// Half-sample symmetric reflection: `d c b a | a b c d | d c b a`.
    #[default]
    Reflect,
    /// Constant fill value (`cval`): `k k k k | a b c d | k k k k`.
    Constant,
    /// Replicate edge pixel: `a a a a | a b c d | d d d d`.
    Nearest,
    /// Whole-sample symmetric reflection: `d c b | a b c d | c b a`.
    Mirror,
    /// Periodic wrap-around: `a b c d | a b c d | a b c d`.
    Wrap,
}

pub const PAD_REFLECT: u32 = 0;
pub const PAD_CONSTANT: u32 = 1;
pub const PAD_NEAREST: u32 = 2;
pub const PAD_MIRROR: u32 = 3;
pub const PAD_WRAP: u32 = 4;

impl PadMode {
    /// Comptime integer identifier passed to `#[cube]` kernels.
    pub const fn id(self) -> u32 {
        match self {
            Self::Reflect => PAD_REFLECT,
            Self::Constant => PAD_CONSTANT,
            Self::Nearest => PAD_NEAREST,
            Self::Mirror => PAD_MIRROR,
            Self::Wrap => PAD_WRAP,
        }
    }

    /// Parses a boundary mode name (accepting common aliases).
    pub fn parse(name: &str) -> OpsResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "reflect" | "symmetric" | "grid-mirror" => Ok(Self::Reflect),
            "constant" | "grid-constant" | "zeros" => Ok(Self::Constant),
            "nearest" | "edge" | "replicate" => Ok(Self::Nearest),
            "mirror" => Ok(Self::Mirror),
            "wrap" | "grid-wrap" | "periodic" => Ok(Self::Wrap),
            other => Err(OpsError::UnknownPadMode(other.to_string())),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Reflect => "reflect",
            Self::Constant => "constant",
            Self::Nearest => "nearest",
            Self::Mirror => "mirror",
            Self::Wrap => "wrap",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modes_and_rejects_invalid() {
        assert_eq!(PadMode::parse("reflect").unwrap(), PadMode::Reflect);
        assert_eq!(PadMode::parse("grid-mirror").unwrap(), PadMode::Reflect);
        assert_eq!(PadMode::parse("constant").unwrap(), PadMode::Constant);
        assert_eq!(PadMode::parse("nearest").unwrap(), PadMode::Nearest);
        assert_eq!(PadMode::parse("mirror").unwrap(), PadMode::Mirror);
        assert_eq!(PadMode::parse("wrap").unwrap(), PadMode::Wrap);
        assert!(PadMode::parse("invalid").is_err());
    }
}
