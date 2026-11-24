// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! exFAT, read-only.
//!
//! [`mount`] parses the boot sector ([`boot`]), loads the first FAT
//! ([`table`]) and answers the [`FileSystem`] calls. Directories are walked
//! by [`dir::Walker`] over a cluster chain; a file is the chain of its
//! clusters (or one contiguous run when its stream says it has no FAT
//! chain), cut to its data length, with zeros after its valid data length.
//!
//! Node ids are the volume byte offset of the file entry set's primary. The
//! root is the first byte of the root cluster plus one (an entry offset is
//! a multiple of 32, so the root cannot collide with the first entry).
//!
//! A set whose primary is the last slot of a cluster continues in the
//! directory's next cluster, which a bare offset cannot find. The listing
//! and lookup remember where such a set's stream slot is, so `metadata`
//! can answer for it; the table holds at most [`MAX_REMEMBERED`] sets.
//!
//! ```no_run
//! use std::path::Path;
//! use std::sync::Arc;
//!
//! use atlas_fs_core::fs::{DirBatch, DirProgress, FileSystem};
//! use atlas_fs_core::io::file::FileSource;
//!
//! # fn main() -> Result<(), atlas_fs_core::error::Error> {
//! let source = Arc::new(FileSource::open(Path::new("volume.img"))?);
//! let volume = atlas_exfat::mount(source)?;
//! println!("{}", volume.info().name);
//!
//! let mut stream = volume.read_dir(volume.root())?;
//! let mut batch = DirBatch::new();
//! while stream.next_batch(&mut batch, 256)? == DirProgress::More {}
//! for entry in batch.iter() {
//!     println!("{}", entry.name);
//! }
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
// On-disk bytes are hostile. These lints make the usual ways of trusting
// them (unchecked indexing, arithmetic and casts, panicking shortcuts) fail
// the clippy gate. Unit tests and the test fixtures are exempt.
#![warn(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::unwrap_used
)]
#![cfg_attr(
    any(test, feature = "test-util"),
    allow(
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use atlas_fat::chain::Chain;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::fs::{
    AllocationMap, DirBatch, DirEntry, DirProgress, DirStream, EntryFlags, Extent, FileSystem,
    MAX_ALLOCATION_BUCKETS, Metadata, NodeId, NodeKind, Stream, VolumeInfo,
};
use atlas_fs_core::io::block::{BlockSource, check_range};

use boot::Geometry;
use dir::{Found, Meta, SLOT, Walker};
use table::ExTable;

pub mod boot;
pub mod dir;
// Seed inputs for the fuzz targets; `volumes()` is shared with Disk Atlas, which adds the probe heads.
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
pub mod fuzz_seeds;
pub mod table;
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod test_volume;

/// Bytes of a primary slot and the slot after it.
const PAIR: usize = 2 * SLOT;

/// Largest directory read, in bytes.
pub const MAX_DIR_BYTES: u64 = 256 * 1024 * 1024;
/// Most sets that straddle a cluster boundary kept for `metadata`.
pub const MAX_REMEMBERED: usize = 1 << 20;
/// Bytes of the root directory read at mount to find the label.
const LABEL_SCAN: u64 = 16 * 1024;
/// The free space is counted at mount only for a bitmap up to this size.
const FREE_COUNT_LIMIT: u64 = 1 << 20;
/// Bytes of the bitmap read at once.
const BITMAP_PIECE: usize = 1 << 16;

/// A mounted exFAT volume.
pub struct ExFat {
    source: Arc<dyn BlockSource>,
    geo: Geometry,
    table: Arc<ExTable>,
    root: u64,
    info: VolumeInfo,
    /// Stream slot offsets of sets that cross a cluster boundary, by node.
    straddling: Arc<Mutex<HashMap<u64, (Meta, u64)>>>,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ExFat>();
};

/// Mounts the exFAT volume in `source`.
///
/// Fails when the boot sector is not a valid exFAT boot sector or the
/// source ends inside the first FAT.
pub fn mount(source: Arc<dyn BlockSource>) -> Result<ExFat> {
    let mut sector = [0u8; boot::BOOT_LEN];
    let have = usize::try_from(source.len()).map_or(sector.len(), |l| l.min(sector.len()));
    if let Some(dst) = sector.get_mut(..have) {
        source.read_at(0, dst)?;
    }
    let geo = boot::parse(sector.get(..have).unwrap_or(&[]))?;
    let table = Arc::new(ExTable::load(Arc::clone(&source), &geo)?);
    let root = geo
        .cluster_offset(geo.root_cluster)
        .map(|at| at.saturating_add(1))
        .ok_or_else(|| {
            Error::from(Corruption::new(
                "exFAT boot sector",
                Location::Unspecified,
                CorruptKind::InvalidValue {
                    field: "root directory cluster",
                    value: u64::from(geo.root_cluster),
                },
            ))
        })?;
    let mut fs = ExFat {
        source,
        geo,
        table,
        root,
        info: VolumeInfo::new(""),
        straddling: Arc::new(Mutex::new(HashMap::new())),
    };
    fs.info = fs.volume_info();
    Ok(fs)
}

fn not_found(node: u64) -> Error {
    Error::NotFound { node }
}

/// A file's content with zeros after its valid data length.
struct ValidData {
    inner: Chain,
    valid: u64,
}

impl BlockSource for ValidData {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn sector_size(&self) -> u32 {
        512
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        check_range(offset, buf.len(), self.inner.len())?;
        let want = u64::try_from(buf.len()).unwrap_or(u64::MAX);
        let end = offset.saturating_add(want);
        if end <= self.valid {
            return self.inner.read_at(offset, buf);
        }
        // The part below the valid length comes from the chain, the rest is
        // zeros.
        let from_disk = usize::try_from(self.valid.saturating_sub(offset)).unwrap_or(0);
        let (head, tail) = buf.split_at_mut(from_disk.min(buf.len()));
        if !head.is_empty() {
            self.inner.read_at(offset, head)?;
        }
        tail.fill(0);
        Ok(())
    }
}

impl ExFat {
    /// The geometry the boot sector gave.
    pub fn geometry(&self) -> &Geometry {
        &self.geo
    }

    fn volume_info(&self) -> VolumeInfo {
        let mut label = None;
        let mut bitmap = None;
        if let Ok(root) = self.root_chain() {
            let len = root.len().min(LABEL_SCAN);
            if let Ok(len) = usize::try_from(len) {
                let mut data = vec![0u8; len];
                if root.read_at(0, &mut data).is_ok() {
                    label = dir::find_label(&data);
                    bitmap = dir::find_bitmap(&data);
                }
            }
        }
        let name = label.map(|l| l.trim_end().to_owned()).unwrap_or_default();
        let mut info = VolumeInfo::new(if name.is_empty() {
            "exFAT volume".to_owned()
        } else {
            name
        });
        info.format = Some("exFAT");
        info.total_bytes = Some(self.geo.heap_bytes());
        info.free_bytes = bitmap.and_then(|(first, len)| self.free_bytes(first, len));
        info.local_times = false;
        info
    }

    /// Free bytes from a small bitmap; `None` for a large or damaged one.
    fn free_bytes(&self, first: u32, len: u64) -> Option<u64> {
        if len > FREE_COUNT_LIMIT || len < self.bitmap_bytes() {
            return None;
        }
        let bitmap = self.bitmap_source(first, len).ok()?;
        let mut data = vec![0u8; usize::try_from(self.bitmap_bytes()).ok()?];
        bitmap.read_at(0, &mut data).ok()?;
        let total = u64::from(self.geo.cluster_count);
        let mut used = 0u64;
        for (i, byte) in data.iter().enumerate() {
            let base = u64::try_from(i).ok()?.saturating_mul(8);
            let live = total.saturating_sub(base).min(8);
            let mask = if live >= 8 {
                0xFF
            } else {
                (1u8 << live).wrapping_sub(1)
            };
            used = used.saturating_add(u64::from((byte & mask).count_ones()));
        }
        Some(
            total
                .saturating_sub(used)
                .saturating_mul(u64::from(self.geo.cluster_size)),
        )
    }

    /// Bytes of bitmap the cluster count needs.
    fn bitmap_bytes(&self) -> u64 {
        u64::from(self.geo.cluster_count).div_ceil(8)
    }

    fn bitmap_source(&self, first: u32, len: u64) -> Result<Chain> {
        let clusters = len.div_ceil(u64::from(self.geo.cluster_size));
        Ok(table::follow(
            &self.geo,
            &self.table,
            Arc::clone(&self.source),
            first,
            clusters,
        )?
        .with_size(len))
    }

    fn root_chain(&self) -> Result<Arc<Chain>> {
        let clusters = MAX_DIR_BYTES
            .checked_div(u64::from(self.geo.cluster_size))
            .unwrap_or(0);
        Ok(Arc::new(table::follow(
            &self.geo,
            &self.table,
            Arc::clone(&self.source),
            self.geo.root_cluster,
            clusters,
        )?))
    }

    /// The bytes of a folder or file as a chain of `meta.data_len` bytes
    /// (a folder's cut to [`MAX_DIR_BYTES`]).
    fn chain_of(&self, meta: &Meta) -> Result<Chain> {
        let mut len = meta.data_len;
        if meta.is_dir() {
            len = len.min(MAX_DIR_BYTES);
        }
        if len == 0 {
            return Ok(Chain::empty(Arc::clone(&self.source)));
        }
        let cs = u64::from(self.geo.cluster_size.max(1));
        let clusters = len.div_ceil(cs);
        if meta.contiguous() {
            let first = meta.first_cluster;
            let last = u64::from(first)
                .checked_add(clusters)
                .and_then(|n| n.checked_sub(1));
            let in_heap =
                first >= 2 && last.is_some_and(|l| l <= u64::from(self.geo.last_cluster()));
            let Some(disk) = self.geo.cluster_offset(first).filter(|_| in_heap) else {
                return Err(Corruption::new(
                    "exFAT file",
                    Location::Object(u64::from(first)),
                    CorruptKind::OutOfRange {
                        field: "contiguous clusters",
                        value: last.unwrap_or(u64::MAX),
                        limit_name: "last cluster",
                        limit: u64::from(self.geo.last_cluster()),
                    },
                )
                .into());
            };
            return Ok(Chain::region(Arc::clone(&self.source), disk, len));
        }
        Ok(table::follow(
            &self.geo,
            &self.table,
            Arc::clone(&self.source),
            meta.first_cluster,
            clusters,
        )?
        .with_size(len))
    }

    fn remember(&self, found: &Found) {
        if found.stream_at == found.node.saturating_add(SLOT as u64) {
            return;
        }
        let mut map = self
            .straddling
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if map.len() < MAX_REMEMBERED || map.contains_key(&found.node) {
            map.insert(found.node, (found.meta, found.stream_at));
        }
    }

    /// What the set at `node` says. The node must be a live file set.
    fn meta_at(&self, node: NodeId) -> Result<Meta> {
        let off = node.0;
        if !off.is_multiple_of(SLOT as u64) || off < self.geo.heap_offset {
            return Err(not_found(off));
        }
        let end = off.checked_add(PAIR as u64);
        if end.is_some_and(|e| e <= self.source.len()) {
            let mut buf = [0u8; PAIR];
            self.source.read_at(off, &mut buf)?;
            let (primary, stream) = buf.split_at(SLOT);
            // A primary that is the last slot of its cluster has its stream
            // slot elsewhere.
            let rel = off.saturating_sub(self.geo.heap_offset);
            let cs = u64::from(self.geo.cluster_size.max(1));
            let last_in_cluster = rel.checked_rem(cs) == Some(cs.saturating_sub(SLOT as u64));
            if !last_in_cluster {
                return Meta::parse(primary, stream).ok_or_else(|| not_found(off));
            }
        }
        let map = self
            .straddling
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        map.get(&off).map(|(m, _)| *m).ok_or_else(|| not_found(off))
    }

    /// The chain of the folder `dir`.
    fn dir_chain(&self, dir: NodeId) -> Result<Arc<Chain>> {
        if dir.0 == self.root {
            return self.root_chain();
        }
        let meta = self.meta_at(dir)?;
        if !meta.is_dir() {
            return Err(Error::NotADirectory { node: dir.0 });
        }
        Ok(Arc::new(self.chain_of(&meta)?))
    }
}

struct ExDir {
    walker: Walker,
    fs: Arc<Mutex<HashMap<u64, (Meta, u64)>>>,
    done: bool,
}

impl DirStream for ExDir {
    fn next_batch(&mut self, out: &mut DirBatch, max: usize) -> Result<DirProgress> {
        if self.done {
            return Ok(DirProgress::Done);
        }
        let mut added = 0usize;
        // Bounded: each turn consumes slots of a bounded chain.
        while added < max {
            match self.walker.next() {
                Ok(Some(found)) => {
                    if found.stream_at != found.node.saturating_add(SLOT as u64) {
                        let mut map = self.fs.lock().unwrap_or_else(PoisonError::into_inner);
                        if map.len() < MAX_REMEMBERED || map.contains_key(&found.node) {
                            map.insert(found.node, (found.meta, found.stream_at));
                        }
                    }
                    let m = found.meta;
                    let dir = m.is_dir();
                    out.push(DirEntry {
                        node: NodeId(found.node),
                        name: &self.walker.name,
                        kind: if dir {
                            NodeKind::Directory
                        } else {
                            NodeKind::File
                        },
                        size: if dir { None } else { Some(m.data_len) },
                        resource_size: None,
                        modified: m.modified_time(),
                        flags: m.entry_flags(),
                    });
                    added = added.saturating_add(1);
                }
                Ok(None) => {
                    self.done = true;
                    return Ok(DirProgress::Done);
                }
                Err(e) => {
                    self.done = true;
                    return Err(e);
                }
            }
        }
        Ok(DirProgress::More)
    }
}

impl FileSystem for ExFat {
    fn info(&self) -> VolumeInfo {
        self.info.clone()
    }

    fn root(&self) -> NodeId {
        NodeId(self.root)
    }

    fn metadata(&self, node: NodeId) -> Result<Metadata> {
        let mut meta = Metadata {
            node,
            kind: NodeKind::Directory,
            size: None,
            resource_size: None,
            created: None,
            modified: None,
            accessed: None,
            flags: EntryFlags::EMPTY,
            link_target: None,
            finder_codes: None,
            owner: None,
            attributes: None,
            finder_info: None,
        };
        if node.0 == self.root {
            return Ok(meta);
        }
        let m = self.meta_at(node)?;
        if !m.is_dir() {
            meta.kind = NodeKind::File;
            meta.size = Some(m.data_len);
        }
        meta.created = m.created_time();
        meta.modified = m.modified_time();
        meta.accessed = m.accessed_time();
        meta.flags = m.entry_flags();
        Ok(meta)
    }

    fn lookup(&self, dir: NodeId, name: &str) -> Result<Option<NodeId>> {
        let want: String = name.chars().flat_map(char::to_lowercase).collect();
        let mut walker = Walker::new(self.dir_chain(dir)?);
        let mut lowered = String::new();
        while let Some(found) = walker.next()? {
            lowered.clear();
            lowered.extend(walker.name.chars().flat_map(char::to_lowercase));
            if lowered == want {
                self.remember(&found);
                return Ok(Some(NodeId(found.node)));
            }
        }
        Ok(None)
    }

    fn read_dir(&self, dir: NodeId) -> Result<Box<dyn DirStream>> {
        let chain = self.dir_chain(dir)?;
        Ok(Box::new(ExDir {
            walker: Walker::new(chain),
            fs: Arc::clone(&self.straddling),
            done: false,
        }))
    }

    fn extents(&self, node: NodeId, stream: Stream) -> Result<Vec<Extent>> {
        if node.0 == self.root {
            return Err(Error::IsADirectory { node: node.0 });
        }
        let m = self.meta_at(node)?;
        if m.is_dir() {
            return Err(Error::IsADirectory { node: node.0 });
        }
        if stream == Stream::Resource {
            return Ok(Vec::new());
        }
        let chain = self.chain_of(&m)?;
        Ok(chain
            .runs()
            .iter()
            .map(|r| Extent {
                offset: r.disk,
                len: r.len,
            })
            .collect())
    }

    fn allocation_map(&self, buckets: usize) -> Result<AllocationMap> {
        let root = self.root_chain()?;
        let len = usize::try_from(root.len().min(LABEL_SCAN)).unwrap_or(0);
        let mut data = vec![0u8; len];
        root.read_at(0, &mut data)?;
        let Some((first, bitmap_len)) = dir::find_bitmap(&data) else {
            return Err(Error::Unsupported("allocation map without a bitmap".into()));
        };
        let need = self.bitmap_bytes();
        if bitmap_len < need {
            return Err(Corruption::new(
                "exFAT allocation bitmap",
                Location::Object(u64::from(first)),
                CorruptKind::Truncated {
                    needed: need,
                    available: bitmap_len,
                },
            )
            .into());
        }
        let bitmap = self.bitmap_source(first, need)?;
        let density = bitmap_density(&bitmap, self.geo.cluster_count, buckets)?;
        Ok(AllocationMap::new(
            self.geo.heap_offset,
            self.geo.heap_bytes(),
            density,
        ))
    }

    fn open(&self, node: NodeId, stream: Stream) -> Result<Arc<dyn BlockSource>> {
        if node.0 == self.root {
            return Err(Error::IsADirectory { node: node.0 });
        }
        let m = self.meta_at(node)?;
        if m.is_dir() {
            return Err(Error::IsADirectory { node: node.0 });
        }
        if stream == Stream::Resource {
            return Err(Error::NoSuchStream { node: node.0 });
        }
        let inner = self.chain_of(&m)?;
        if m.valid_len >= m.data_len {
            return Ok(Arc::new(inner));
        }
        Ok(Arc::new(ValidData {
            inner,
            valid: m.valid_len,
        }))
    }
}

/// How full each of `buckets` slices of `total` clusters is, from an exFAT
/// bitmap (bit 0 of byte 0 is cluster 2, least significant bit first).
///
/// Bucket `i` covers clusters `[i*T/n, (i+1)*T/n)`; its value is
/// `ceil(used * 255 / clusters)`. The count is clamped to
/// 1 ..= [`MAX_ALLOCATION_BUCKETS`] and to `total`. Bits past `total` are
/// ignored.
fn bitmap_density(bitmap: &dyn BlockSource, total: u32, buckets: usize) -> Result<Vec<u8>> {
    let total = u64::from(total);
    let n = buckets
        .clamp(1, MAX_ALLOCATION_BUCKETS)
        .min(usize::try_from(total).unwrap_or(usize::MAX))
        .max(1);
    let n64 = u64::try_from(n).unwrap_or(1);
    let edge = |i: u64| i.saturating_mul(total).checked_div(n64).unwrap_or(0);
    let mut used = vec![0u64; n];
    // The bucket of the clusters being counted, and the first cluster after it.
    let mut bucket = 0usize;
    let mut hi = edge(1);
    let mut piece = vec![0u8; BITMAP_PIECE];
    let bytes = total.div_ceil(8);
    let mut at = 0u64;
    while at < bytes {
        let take =
            usize::try_from(bytes.saturating_sub(at)).map_or(BITMAP_PIECE, |l| l.min(BITMAP_PIECE));
        let Some(dst) = piece.get_mut(..take) else {
            break;
        };
        bitmap.read_at(at, dst)?;
        for (i, byte) in dst.iter().enumerate() {
            let base = at
                .saturating_add(u64::try_from(i).unwrap_or(0))
                .saturating_mul(8);
            let live = total.saturating_sub(base).min(8);
            if live == 0 {
                break;
            }
            while base >= hi && bucket.saturating_add(1) < n {
                bucket = bucket.saturating_add(1);
                hi = edge(u64::try_from(bucket).unwrap_or(0).saturating_add(1));
            }
            let mask = if live >= 8 {
                0xFF
            } else {
                (1u8 << live).wrapping_sub(1)
            };
            if base.saturating_add(live) <= hi || bucket.saturating_add(1) >= n {
                if let Some(slot) = used.get_mut(bucket) {
                    *slot = slot.saturating_add(u64::from((byte & mask).count_ones()));
                }
                continue;
            }
            // A byte that crosses a bucket edge: count its bits one by one.
            for bit in 0..live {
                let cluster = base.saturating_add(bit);
                while cluster >= hi && bucket.saturating_add(1) < n {
                    bucket = bucket.saturating_add(1);
                    hi = edge(u64::try_from(bucket).unwrap_or(0).saturating_add(1));
                }
                if byte >> bit & 1 == 1
                    && let Some(slot) = used.get_mut(bucket)
                {
                    *slot = slot.saturating_add(1);
                }
            }
        }
        at = at.saturating_add(u64::try_from(take).unwrap_or(0));
    }
    let mut out = Vec::with_capacity(n);
    for (i, &u) in used.iter().enumerate() {
        let i = u64::try_from(i).unwrap_or(0);
        let size = edge(i.saturating_add(1)).saturating_sub(edge(i)).max(1);
        let d = u.saturating_mul(255).div_ceil(size);
        out.push(u8::try_from(d).unwrap_or(255));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::test_volume::{Built, Spec, VolumeBuilder, standard};
    use super::*;
    use atlas_fs_core::fs::DirBatch;
    use atlas_fs_core::io::block::MemSource;

    fn mounted(built: &Built) -> ExFat {
        mount(Arc::new(MemSource::new(built.trimmed()))).expect("mount")
    }

    fn list(fs: &ExFat, dir: NodeId) -> Vec<(String, NodeId, NodeKind, Option<u64>)> {
        let mut stream = fs.read_dir(dir).unwrap();
        let mut batch = DirBatch::default();
        let mut out = Vec::new();
        loop {
            batch.clear();
            let progress = stream.next_batch(&mut batch, 3).unwrap();
            out.extend(
                batch
                    .iter()
                    .map(|e| (e.name.to_owned(), e.node, e.kind, e.size)),
            );
            if progress == DirProgress::Done {
                return out;
            }
        }
    }

    fn read_all(source: &dyn BlockSource) -> Vec<u8> {
        let mut buf = vec![0u8; usize::try_from(source.len()).unwrap()];
        source.read_at(0, &mut buf).unwrap();
        buf
    }

    fn walk(fs: &ExFat, dir: NodeId, prefix: &str, out: &mut BTreeMap<String, (NodeId, NodeKind)>) {
        for (name, node, kind, _) in list(fs, dir) {
            let path = format!("{prefix}{name}");
            out.insert(path.clone(), (node, kind));
            if kind == NodeKind::Directory {
                walk(fs, node, &format!("{path}/"), out);
            }
        }
    }

    #[test]
    fn the_whole_tree_and_every_file_match_what_was_built() {
        let built = standard();
        let fs = mounted(&built);
        assert_eq!(fs.info().format, Some("exFAT"));
        assert_eq!(fs.info().name, "ATLAS");
        assert!(!fs.info().local_times);
        let mut seen = BTreeMap::new();
        walk(&fs, fs.root(), "/", &mut seen);
        let live: Vec<_> = built.items.iter().filter(|i| !i.skipped).collect();
        assert_eq!(seen.len(), live.len(), "{seen:?}");
        for item in live {
            let (node, kind) = seen
                .get(&item.path)
                .unwrap_or_else(|| panic!("missing {}", item.path));
            assert_eq!(node.0, item.record, "node of {}", item.path);
            if item.is_dir {
                assert_eq!(*kind, NodeKind::Directory);
                continue;
            }
            let meta = fs.metadata(*node).unwrap();
            assert_eq!(meta.size, Some(item.data.len() as u64));
            let source = fs.open(*node, Stream::Data).unwrap();
            assert_eq!(read_all(&*source), item.expected, "{}", item.path);
        }
    }

    #[test]
    fn valid_data_length_zero_fills_the_rest() {
        let built = standard();
        let fs = mounted(&built);
        let node = fs.lookup(fs.root(), "partial.bin").unwrap().unwrap();
        let source = fs.open(node, Stream::Data).unwrap();
        let all = read_all(&*source);
        let item = built.item("/partial.bin");
        assert_eq!(all.len(), item.data.len());
        let valid = item.valid.unwrap() as usize;
        assert_eq!(all[..valid], item.data[..valid]);
        assert!(all[valid..].iter().all(|&b| b == 0));
        // A read that straddles the valid length.
        let mut buf = vec![0xAAu8; 40];
        source.read_at(valid as u64 - 10, &mut buf).unwrap();
        assert_eq!(buf[..10], item.data[valid - 10..valid]);
        assert!(buf[10..].iter().all(|&b| b == 0));
    }

    #[test]
    fn times_are_reported_in_utc() {
        let built = standard();
        let fs = mounted(&built);
        let node = fs.lookup(fs.root(), "partial.bin").unwrap().unwrap();
        let meta = fs.metadata(node).unwrap();
        let utc = test_volume::UNIX_SECONDS * 1_000_000_000;
        // Created with a +1:00 offset, modified without one.
        assert_eq!(
            meta.created.unwrap().unix_nanos(),
            utc - 3600 * 1_000_000_000
        );
        assert_eq!(meta.modified.unwrap().unix_nanos(), utc);
    }

    #[test]
    fn hidden_names_and_malformed_sets() {
        let built = standard();
        let fs = mounted(&built);
        let root = list(&fs, fs.root());
        let secret = root.iter().find(|(n, ..)| n == "secret.sys").unwrap();
        assert!(
            fs.metadata(secret.1)
                .unwrap()
                .flags
                .contains(EntryFlags::HIDDEN)
        );
        // The malformed sets are skipped; every good one after them lists.
        assert!(root.iter().any(|(n, ..)| n == "after-damage.txt"));
        assert!(root.iter().all(|(n, ..)| !n.starts_with("damaged")));
    }

    #[test]
    fn lookup_ignores_case_and_finds_long_names() {
        let built = standard();
        let fs = mounted(&built);
        let node = fs.lookup(fs.root(), "FRAG.BIN").unwrap().unwrap();
        assert_eq!(node.0, built.item("/frag.bin").record);
        let docs = fs.lookup(fs.root(), "DOCS").unwrap().unwrap();
        let long = fs
            .lookup(docs, &test_volume::LONG_NAME.to_uppercase())
            .unwrap();
        assert!(long.is_some());
        assert_eq!(fs.lookup(fs.root(), "nothing").unwrap(), None);
        assert!(matches!(
            fs.lookup(node, "x"),
            Err(Error::NotADirectory { .. })
        ));
    }

    #[test]
    fn extents_show_contiguous_and_fragmented_files() {
        let built = standard();
        let fs = mounted(&built);
        let frag = fs.lookup(fs.root(), "frag.bin").unwrap().unwrap();
        let extents = fs.extents(frag, Stream::Data).unwrap();
        assert!(extents.len() > 1);
        let item = built.item("/frag.bin");
        for (ext, cluster) in extents.iter().zip(&item.clusters) {
            assert_eq!(
                ext.offset,
                built.heap_offset + u64::from(cluster - 2) * built.cluster_size
            );
        }
        let big = fs.lookup(fs.root(), "partial.bin").unwrap().unwrap();
        let one = fs.extents(big, Stream::Data).unwrap();
        assert_eq!(one.len(), 1, "a contiguous file is one run");
        assert!(fs.extents(big, Stream::Resource).unwrap().is_empty());
        let docs = fs.lookup(fs.root(), "docs").unwrap().unwrap();
        assert!(matches!(
            fs.extents(docs, Stream::Data),
            Err(Error::IsADirectory { .. })
        ));
    }

    #[test]
    fn sets_that_cross_clusters_still_have_metadata() {
        // 512-byte clusters hold 16 slots, so many long names make sets
        // cross from one directory cluster to the next, and the interleaved
        // allocation puts those clusters apart.
        let built = test_volume::crowded();
        let fs = mounted(&built);
        let mut seen = BTreeMap::new();
        walk(&fs, fs.root(), "/", &mut seen);
        let crossing = built.items.iter().filter(|i| i.straddles).count();
        assert!(crossing > 0, "the volume has crossing sets");
        for item in built.items.iter().filter(|i| !i.skipped) {
            let (node, _) = seen
                .get(&item.path)
                .unwrap_or_else(|| panic!("{}", item.path));
            let meta = fs
                .metadata(*node)
                .unwrap_or_else(|e| panic!("{}: {e}", item.path));
            if !item.is_dir {
                assert_eq!(meta.size, Some(item.data.len() as u64), "{}", item.path);
                let source = fs.open(*node, Stream::Data).unwrap();
                assert_eq!(read_all(&*source), item.expected, "{}", item.path);
            }
        }
    }

    #[test]
    fn a_crossing_set_is_unknown_until_it_is_listed() {
        let built = test_volume::crowded();
        let item = built.items.iter().find(|i| i.straddles).unwrap();
        let fs = mounted(&built);
        assert!(fs.metadata(NodeId(item.record)).is_err());
        let mut seen = BTreeMap::new();
        walk(&fs, fs.root(), "/", &mut seen);
        assert!(fs.metadata(NodeId(item.record)).is_ok());
    }

    #[test]
    fn allocation_map_counts_the_used_clusters() {
        let built = standard();
        let fs = mounted(&built);
        let total = usize::try_from(built.cluster_count).unwrap();
        let map = fs.allocation_map(total).unwrap();
        assert_eq!(map.start, built.heap_offset);
        assert_eq!(map.density.len(), total);
        assert!(map.density.iter().all(|&d| d == 0 || d == 255));
        let used = map.density.iter().filter(|&&d| d == 255).count();
        assert_eq!(used, built.used_clusters);
        let free = fs.info().free_bytes.unwrap() / built.cluster_size;
        assert_eq!(free as usize, total - used);
        let coarse = fs.allocation_map(4).unwrap();
        assert_eq!(coarse.density.len(), 4);
        assert!(coarse.density.iter().any(|&d| d > 0));
        assert_eq!(fs.allocation_map(0).unwrap().density.len(), 1);
        // The folded counts agree with the per-cluster ones.
        let six = fs.allocation_map(6).unwrap().density;
        let expect: Vec<u8> = (0..6usize)
            .map(|i| {
                let lo = i * total / 6;
                let hi = (i + 1) * total / 6;
                let u = map.density[lo..hi].iter().filter(|&&d| d == 255).count();
                ((u * 255).div_ceil(hi - lo)) as u8
            })
            .collect();
        assert_eq!(six, expect);
    }

    #[test]
    fn folders_and_the_root_are_not_files() {
        let built = standard();
        let fs = mounted(&built);
        let docs = fs.lookup(fs.root(), "docs").unwrap().unwrap();
        assert!(matches!(
            fs.open(docs, Stream::Data),
            Err(Error::IsADirectory { .. })
        ));
        assert!(matches!(
            fs.open(fs.root(), Stream::Data),
            Err(Error::IsADirectory { .. })
        ));
        let file = fs.lookup(docs, "notes.txt").unwrap().unwrap();
        assert!(matches!(
            fs.open(file, Stream::Resource),
            Err(Error::NoSuchStream { .. })
        ));
        assert_eq!(fs.metadata(fs.root()).unwrap().kind, NodeKind::Directory);
    }

    #[test]
    fn invented_node_ids_are_not_found() {
        let built = standard();
        let fs = mounted(&built);
        let first = list(&fs, fs.root())[0].1;
        let bogus = [
            1,
            3,
            31,
            first.0 + 1,
            first.0 + 5,
            built.fat_offset,
            built.heap_offset - 32,
            u64::MAX,
            u64::MAX - 31,
            built.image.len() as u64 + 64,
        ];
        for node in bogus {
            if node == fs.root().0 {
                continue;
            }
            assert!(fs.metadata(NodeId(node)).is_err(), "node {node}");
            assert!(fs.read_dir(NodeId(node)).is_err());
            assert!(fs.open(NodeId(node), Stream::Data).is_err());
        }
    }

    #[test]
    fn the_root_does_not_collide_with_its_first_entry() {
        let built = standard();
        let fs = mounted(&built);
        for (_, node, ..) in list(&fs, fs.root()) {
            assert_ne!(node, fs.root());
        }
    }

    #[test]
    fn mounting_rejects_garbage_and_truncation() {
        for bytes in [vec![], vec![0u8; 100], vec![0u8; 4096], vec![0xFFu8; 4096]] {
            assert!(mount(Arc::new(MemSource::new(bytes))).is_err());
        }
        let built = standard();
        let cut = usize::try_from(built.fat_offset).unwrap() + 40;
        assert!(mount(Arc::new(MemSource::new(built.image[..cut].to_vec()))).is_err());
    }

    #[test]
    fn a_damaged_chain_reads_up_to_the_damage() {
        let mut built = standard();
        let item = built.item("/frag.bin").clone();
        // Free the entry after the first cluster.
        let at = usize::try_from(built.fat_offset).unwrap() + item.clusters[0] as usize * 4;
        built.image[at..at + 4].fill(0);
        let fs = mounted(&built);
        let node = fs.lookup(fs.root(), "frag.bin").unwrap().unwrap();
        let source = fs.open(node, Stream::Data).unwrap();
        let cs = usize::try_from(built.cluster_size).unwrap();
        let mut first = vec![0u8; cs];
        source.read_at(0, &mut first).unwrap();
        assert_eq!(first, item.data[..cs]);
        let mut all = vec![0u8; item.data.len()];
        assert!(source.read_at(0, &mut all).is_err());
    }

    #[test]
    fn a_contiguous_file_outside_the_heap_does_not_open() {
        let mut b = VolumeBuilder::new();
        b.label("X");
        b.file_spec(Spec::file("far.bin", &[7u8; 2000]).first_cluster(Some(90)));
        let built = b.build();
        let fs = mounted(&built);
        let node = fs.lookup(fs.root(), "far.bin").unwrap().unwrap();
        assert!(matches!(
            fs.open(node, Stream::Data),
            Err(Error::Corrupt(_))
        ));
        assert!(fs.extents(node, Stream::Data).is_err());
    }

    #[test]
    fn a_directory_loop_ends_with_an_error() {
        let mut built = standard();
        // Make the root directory's chain point at itself.
        let at = usize::try_from(built.fat_offset).unwrap() + built.root_cluster as usize * 4;
        built.image[at..at + 4].copy_from_slice(&built.root_cluster.to_le_bytes());
        let fs = mount(Arc::new(MemSource::new(built.trimmed())));
        // The loop may be seen at mount (no label) or at the listing.
        if let Ok(fs) = fs
            && let Ok(mut s) = fs.read_dir(fs.root())
        {
            {
                let mut batch = DirBatch::default();
                let mut guard = 0;
                loop {
                    batch.clear();
                    guard += 1;
                    assert!(guard < 1000, "the walk must end");
                    match s.next_batch(&mut batch, 64) {
                        Ok(DirProgress::More) => {}
                        Ok(DirProgress::Done) | Err(_) => break,
                    }
                }
            }
        }
    }
}
