# atlas-exfat

A read-only exFAT reader. It implements `atlas_fs_core::fs::FileSystem`, so a mounted volume lists folders, looks up names, reads
file content and reports its allocation through the same trait as the other `atlas-*` readers.

- Boot sector, allocation bitmap and FAT, contiguous and FAT-chained files.
- Directory entry sets (file, stream extension, name) with long UTF-16 names, timestamps with UTC offsets, and the volume label.
- Entry sets that straddle a cluster boundary are found again by `metadata`.
- No `unsafe`. Every length, offset and count read from the disk is checked; damage is an error, not a panic.

```rust,no_run
use std::sync::Arc;
use atlas_fs_core::fs::FileSystem;
use atlas_fs_core::io::block::BlockSource;
use atlas_fs_core::io::file::FileSource;

let source: Arc<dyn BlockSource> = Arc::new(FileSource::open("volume.img".as_ref())?);
let volume = atlas_exfat::mount(source)?;
let mut dir = volume.read_dir(volume.root())?;
# Ok::<(), atlas_fs_core::error::Error>(())
```

The `test-util` feature exposes the test image builder and fuzz seed inputs; it is for tests of crates that use
this one and is not part of the stable API.

## Licence

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)) or MIT license
([LICENSE-MIT](LICENSE-MIT)) at your option.
