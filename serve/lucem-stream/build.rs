//! Generates the browser protocol's types from `proto/lucem_stream.proto` (protox: no protoc
//! needed).

const PROTO_DIR: &str = "proto";
const PROTO: &str = "proto/lucem_stream.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={PROTO}");
    let files = protox::compile([PROTO], [PROTO_DIR])?;
    prost_build::Config::new().compile_fds(files)?;
    Ok(())
}
