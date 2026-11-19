// lucem: name = "Channel Ratio"
// lucem: family = "Intensity"
// lucem: description = "Computes A / (B + eps) pixel-by-pixel for ratiometric imaging"
// lucem: halo = 0
// lucem: input a: image
// lucem: input b: image
// lucem: param eps = 0.01
// lucem: output image

pub fn run(inputs: &Inputs, params: &Params) -> Result<Array, String> {
    let a = inputs.get("a").ok_or("missing input `a`")?;
    let b = inputs.get("b").ok_or("missing input `b`")?;
    if a.width != b.width || a.height != b.height {
        return Err("inputs `a` and `b` must have the same dimensions".into());
    }
    let eps = params.get("eps").copied().unwrap_or(0.01).max(1e-6) as f32;
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| (x / (y.max(0.0) + eps)).max(0.0))
        .collect();
    Ok(Array {
        width: a.width,
        height: a.height,
        data,
    })
}
