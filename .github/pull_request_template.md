## What and why

<!-- What this changes and why; link the issue it closes (Closes #…). -->

## Checks

- [ ] Built and tested with the one profile: `cargo test --workspace --exclude lumen-app --exclude lumen-py --lib --profile validate --no-run`, then the test binaries
- [ ] Runtimes the tests ran on: <!-- wgpu / CUDA / CPU; CI runs the CPU runtime only -->
- [ ] Desktop application started and the change tried in it, if the UI or what it calls changed: `cargo run -p lumen-app`
- [ ] Results checked against their reference (scipy.ndimage, scikit-image) where they can change
- [ ] Docs and the CHANGELOG updated if behaviour or API changed
- [ ] No research data, credentials or API keys are included

## Licences

- [ ] Code ported from elsewhere keeps its upstream notice (see CONTRIBUTING.md)
