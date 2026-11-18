//! The browser protocol's messages (protobuf, `proto/lucem_stream.proto`) and their conversions
//! to the crate's own types. No transport and no file in it: a page built for WebAssembly uses
//! it as the server does.

#![allow(clippy::all, missing_docs)]

include!(concat!(env!("OUT_DIR"), "/lucem_stream.v1.rs"));

use crate::protocol::{ChannelDisplay, ImageInfo as Described};

/// `info` as sent to a page.
pub fn info_to_wire(info: &Described) -> self::ImageInfo {
    let channels = info
        .channels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let d = info.display.get(i);
            self::Channel {
                label: label.clone(),
                color: d.map_or(0xff_ffff, |d| (u32::from(d.color[0]) << 16) | (u32::from(d.color[1]) << 8) | u32::from(d.color[2])),
                window_low: d.map_or(0.0, |d| d.window.0),
                window_high: d.map_or(65535.0, |d| d.window.1),
                range_low: d.map_or(0.0, |d| d.range.0),
                range_high: d.map_or(65535.0, |d| d.range.1),
                active: d.is_none_or(|d| d.active),
            }
        })
        .collect();
    let levels = info.levels.iter().map(|&[d, h, w, td, th, tw]| self::Level { depth: d as u32, height: h as u32, width: w as u32, tile_d: td as u32, tile_h: th as u32, tile_w: tw as u32 }).collect();
    self::ImageInfo { name: info.name.clone(), dtype: info.dtype.clone(), channels, pixel_size_um: info.pixel_size_um, z_spacing_um: info.z_spacing_um, levels }
}

/// A page's view of `wire`, as the crate's description.
pub fn info_from_wire(wire: &self::ImageInfo) -> Described {
    Described {
        name: wire.name.clone(),
        dtype: wire.dtype.clone(),
        channels: wire.channels.iter().map(|c| c.label.clone()).collect(),
        pixel_size_um: wire.pixel_size_um,
        z_spacing_um: wire.z_spacing_um,
        levels: wire.levels.iter().map(|l| [l.depth as usize, l.height as usize, l.width as usize, l.tile_d as usize, l.tile_h as usize, l.tile_w as usize]).collect(),
        display: wire
            .channels
            .iter()
            .map(|c| ChannelDisplay { color: [(c.color >> 16) as u8, (c.color >> 8) as u8, c.color as u8], window: (c.window_low, c.window_high), range: (c.range_low, c.range_high), active: c.active })
            .collect(),
    }
}

/// A tile's values packed for the wire: as the stored type (`dtype`) when every value is one
/// exactly (16-bit: half the bytes of f32), else f32.
pub fn encode_tile(width: usize, height: usize, values: &[f32], dtype: &str) -> self::Tile {
    let exact = |max: f32| values.iter().all(|v| *v >= 0.0 && *v <= max && v.fract() == 0.0);
    let (encoding, bytes) = match dtype {
        "uint8" | "u8" if exact(255.0) => (TileEncoding::U8, values.iter().map(|v| *v as u8).collect()),
        "uint8" | "u8" | "uint16" | "u16" if exact(65535.0) => (TileEncoding::U16, values.iter().flat_map(|v| (*v as u16).to_le_bytes()).collect()),
        _ => (TileEncoding::F32, values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    };
    self::Tile { width: width as u32, height: height as u32, encoding: encoding as i32, values: bytes }
}

/// A tile's values from the wire, checked against its size.
pub fn decode_tile(tile: &self::Tile) -> Result<(usize, usize, Vec<f32>), String> {
    let (w, h) = (tile.width as usize, tile.height as usize);
    let n = w.checked_mul(h).ok_or("a tile's size overflows")?;
    let values: Vec<f32> = match TileEncoding::try_from(tile.encoding).unwrap_or(TileEncoding::Unspecified) {
        TileEncoding::U8 => tile.values.iter().map(|&b| f32::from(b)).collect(),
        TileEncoding::U16 => tile.values.as_chunks::<2>().0.iter().map(|c| f32::from(u16::from_le_bytes(*c))).collect(),
        TileEncoding::F32 => tile.values.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        TileEncoding::Unspecified => return Err("a tile without its encoding".into()),
    };
    if values.len() != n {
        return Err(format!("a tile of {w} × {h} came with {} values", values.len()));
    }
    Ok((w, h, values))
}

/// A response carrying only `reason`.
pub fn error(reason: impl Into<String>) -> self::Response {
    self::Response { kind: Some(response::Kind::Error(reason.into())) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_pack_as_their_type_when_exact() {
        let values = [0.0, 1.0, 255.0, 4095.0];
        let t = encode_tile(2, 2, &values, "uint16");
        assert_eq!((t.encoding, t.values.len()), (TileEncoding::U16 as i32, 8));
        assert_eq!(decode_tile(&t).unwrap(), (2, 2, values.to_vec()));
        let t = encode_tile(2, 2, &[0.0, 1.0, 2.0, 3.0], "uint8");
        assert_eq!((t.encoding, t.values.len()), (TileEncoding::U8 as i32, 4));
        // Not exact (a processed tile): f32
        let t = encode_tile(2, 2, &[0.5, 1.0, 2.0, 3.0], "uint16");
        assert_eq!(t.encoding, TileEncoding::F32 as i32);
        assert_eq!(decode_tile(&t).unwrap().2, vec![0.5, 1.0, 2.0, 3.0]);
        assert!(decode_tile(&Tile { width: 3, height: 3, encoding: TileEncoding::U8 as i32, values: vec![0; 4] }).is_err());
    }

    #[test]
    fn info_goes_and_comes_back() {
        let info = Described {
            name: "a".into(),
            dtype: "uint16".into(),
            channels: vec!["dapi".into()],
            pixel_size_um: Some(0.5),
            z_spacing_um: None,
            levels: vec![[1, 20, 30, 1, 8, 8], [1, 10, 15, 1, 8, 8]],
            display: vec![ChannelDisplay { color: [0, 128, 255], window: (10.0, 900.0), range: (0.0, 65535.0), active: true }],
        };
        assert_eq!(info_from_wire(&info_to_wire(&info)), info);
    }
}
