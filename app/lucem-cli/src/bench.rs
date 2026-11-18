//! `lucem bench-codecs`: which codec suits this data. Tiles of an image (its stored values: 8- and
//! 16-bit as integers, else f32), each codec's ratio and speed, per tile as a viewer reads them.
//! Decoding is what a viewer pays at every tile: it is measured natively, and with ruzstd for
//! zstd (the pure-Rust decoder a browser page would run).

use std::path::Path;
use std::time::{Duration, Instant};

use lucem_io::Pyramid;

/// One tile's stored values.
enum Values {
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl Values {
    fn bytes(&self) -> Vec<u8> {
        match self {
            Values::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Values::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

/// What one codec did over all tiles.
#[derive(Default)]
struct Score {
    raw: usize,
    packed: usize,
    encode: Duration,
    decode: Duration,
}

impl Score {
    fn line(&self, name: &str) -> String {
        let mb = self.raw as f64 / 1e6;
        let rate = |d: Duration| if d.is_zero() { f64::INFINITY } else { mb / d.as_secs_f64() };
        format!("{name:<16} {:>7.2}×  {:>9.0} MB/s  {:>9.0} MB/s", self.raw as f64 / self.packed.max(1) as f64, rate(self.encode), rate(self.decode))
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let t0 = Instant::now();
    let out = f();
    (out, t0.elapsed())
}

/// `image`'s tiles at `level` (spread over the image, every channel), at most `count`.
fn sample(py: &Pyramid, level: usize, count: usize) -> Result<Vec<Values>, String> {
    let l = py.levels.get(level).ok_or_else(|| format!("no level {level}"))?;
    let all: Vec<(usize, usize)> = (0..l.tiles_y()).flat_map(|ty| (0..l.tiles_x()).map(move |tx| (tx, ty))).collect();
    let per_channel = count.div_ceil(py.channels.len()).max(1);
    let step = (all.len() / per_channel).max(1);
    let integer = matches!(py.dtype.as_str(), "uint8" | "uint16" | "u8" | "u16");
    let mut out = Vec::new();
    for c in 0..py.channels.len() {
        for &(tx, ty) in all.iter().step_by(step).take(per_channel) {
            let t = py.read_tile(level, c, tx, ty).map_err(|e| e.to_string())?;
            out.push(if integer && t.data.iter().all(|v| *v >= 0.0 && *v <= 65535.0 && v.fract() == 0.0) { Values::U16(t.data.iter().map(|v| *v as u16).collect()) } else { Values::F32(t.data) });
        }
    }
    Ok(out)
}

/// Measures every codec on `image`; the table as text.
pub fn bench_codecs(image: &Path, level: usize, count: usize) -> Result<String, String> {
    let py = Pyramid::open(image).map_err(|e| e.to_string())?;
    let tiles = sample(&py, level, count)?;
    if tiles.is_empty() {
        return Err("no tiles to measure".into());
    }
    let (mut zstd3, mut zstd_rs, mut lz4, mut pcodec) = (Score::default(), Score::default(), Score::default(), Score::default());
    for t in &tiles {
        let raw = t.bytes();
        // zstd, level 3 (zarr's default), C decoder and ruzstd
        let (z, e) = timed(|| zstd::bulk::compress(&raw, 3).map_err(|e| e.to_string()));
        let z = z?;
        let (back, d) = timed(|| zstd::bulk::decompress(&z, raw.len()).map_err(|e| e.to_string()));
        if back? != raw {
            return Err("zstd did not give the tile back".into());
        }
        zstd3 = Score { raw: zstd3.raw + raw.len(), packed: zstd3.packed + z.len(), encode: zstd3.encode + e, decode: zstd3.decode + d };
        let (back, d) = timed(|| {
            use std::io::Read as _;
            let mut out = Vec::with_capacity(raw.len());
            ruzstd::decoding::StreamingDecoder::new(&mut z.as_slice()).map_err(|e| e.to_string())?.read_to_end(&mut out).map_err(|e| e.to_string())?;
            Ok::<_, String>(out)
        });
        if back? != raw {
            return Err("ruzstd did not give the tile back".into());
        }
        zstd_rs = Score { raw: zstd_rs.raw + raw.len(), packed: zstd_rs.packed + z.len(), encode: zstd_rs.encode + e, decode: zstd_rs.decode + d };
        // LZ4
        let (z, e) = timed(|| lz4_flex::compress_prepend_size(&raw));
        let (back, d) = timed(|| lz4_flex::decompress_size_prepended(&z).map_err(|e| e.to_string()));
        if back? != raw {
            return Err("LZ4 did not give the tile back".into());
        }
        lz4 = Score { raw: lz4.raw + raw.len(), packed: lz4.packed + z.len(), encode: lz4.encode + e, decode: lz4.decode + d };
        // Pcodec, on the numbers themselves
        let config = pco::ChunkConfig::default();
        let (z, e, d, same) = match t {
            Values::U16(v) => {
                let (z, e) = timed(|| pco::standalone::simple_compress(v, &config).map_err(|e| e.to_string()));
                let z = z?;
                let (back, d) = timed(|| pco::standalone::simple_decompress::<u16>(&z).map_err(|e| e.to_string()));
                let same = &back? == v;
                (z, e, d, same)
            }
            Values::F32(v) => {
                let (z, e) = timed(|| pco::standalone::simple_compress(v, &config).map_err(|e| e.to_string()));
                let z = z?;
                let (back, d) = timed(|| pco::standalone::simple_decompress::<f32>(&z).map_err(|e| e.to_string()));
                let same = back?.iter().zip(v).all(|(a, b)| a.to_bits() == b.to_bits());
                (z, e, d, same)
            }
        };
        if !same {
            return Err("Pcodec did not give the tile back".into());
        }
        pcodec = Score { raw: pcodec.raw + raw.len(), packed: pcodec.packed + z.len(), encode: pcodec.encode + e, decode: pcodec.decode + d };
    }
    let kind = match &tiles[0] {
        Values::U16(_) => "integers (u16)",
        Values::F32(_) => "f32",
    };
    let l = &py.levels[level];
    let mut out = format!("{}: {} tile(s) of {} × {} at level {level}, {} channel(s), {kind}; {:.1} MB raw\n", py.name, tiles.len(), l.tile_w, l.tile_h, py.channels.len(), zstd3.raw as f64 / 1e6);
    out.push_str(&format!("{:<16} {:>8}  {:>14}  {:>14}\n", "codec", "ratio", "encode", "decode"));
    for (name, s) in [("zstd 3", &zstd3), ("zstd 3 (ruzstd)", &zstd_rs), ("LZ4", &lz4), ("Pcodec", &pcodec)] {
        out.push_str(&s.line(name));
        out.push('\n');
    }
    out.push_str("(ruzstd decodes what zstd encoded: a page would use it; every codec gave each tile back exactly)\n");
    Ok(out)
}
