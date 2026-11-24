// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Builds exFAT volumes for tests and fuzz seeds.
//!
//! Clusters are 512 bytes. Cluster 2 is the allocation bitmap; the
//! directories' clusters come next, taken in turn from each directory so
//! that a directory's clusters are not adjacent; then the files. A file is
//! contiguous (no FAT chain) unless it is built fragmented, which puts a
//! free cluster between each pair of its clusters and chains it in the FAT.

use atlas_fat::test_volume::{DATE, TIME, pattern};

/// [`DATE`] and [`TIME`] as seconds since the Unix epoch, in UTC.
pub const UNIX_SECONDS: i64 = atlas_fat::test_volume::UNIX_SECONDS;

/// A name that takes three name entries and has non-ASCII letters.
pub const LONG_NAME: &str = "Ünïcode long name for the exFAT set.txt";

const CLUSTER: usize = 512;
const SLOT: usize = 32;
const SLOTS_PER_CLUSTER: usize = CLUSTER / SLOT;
const END: u32 = 0xFFFF_FFFF;

/// How a set is damaged on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Damage {
    /// The primary says one secondary entry.
    CountOne,
    /// A name entry has type `0xC2`.
    BadNameType,
    /// The stream extension says the name has no characters.
    NoName,
}

/// One entry to build.
#[derive(Debug, Clone)]
pub struct Spec {
    name: String,
    /// `None` for a folder.
    data: Option<Vec<u8>>,
    fragmented: bool,
    hidden: bool,
    valid: Option<u64>,
    created_offset: u8,
    first_override: Option<u32>,
    damage: Option<Damage>,
}

impl Spec {
    /// A file.
    pub fn file(name: &str, data: &[u8]) -> Self {
        Self {
            name: name.to_owned(),
            data: Some(data.to_vec()),
            fragmented: false,
            hidden: false,
            valid: None,
            created_offset: 0,
            first_override: None,
            damage: None,
        }
    }

    /// A folder.
    pub fn folder(name: &str) -> Self {
        Self {
            data: None,
            ..Self::file(name, &[])
        }
    }

    /// Chained in the FAT, with a gap between its clusters.
    #[must_use]
    pub fn fragmented(mut self) -> Self {
        self.fragmented = true;
        self
    }

    /// Hidden and system.
    #[must_use]
    pub fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Valid data length below the data length.
    #[must_use]
    pub fn valid(mut self, len: u64) -> Self {
        self.valid = Some(len);
        self
    }

    /// The creation time's UTC offset byte.
    #[must_use]
    pub fn created_offset(mut self, byte: u8) -> Self {
        self.created_offset = byte;
        self
    }

    /// Put the first cluster here instead of allocating.
    #[must_use]
    pub fn first_cluster(mut self, cluster: Option<u32>) -> Self {
        self.first_override = cluster;
        self
    }

    /// Damage the set.
    #[must_use]
    pub fn damaged(mut self, damage: Damage) -> Self {
        self.damage = Some(damage);
        self
    }
}

/// One entry of a built volume.
#[derive(Debug, Clone)]
pub struct Item {
    /// Path from the root, with a leading `/`.
    pub path: String,
    /// True for a folder.
    pub is_dir: bool,
    /// The content given.
    pub data: Vec<u8>,
    /// What reading it gives: the content, zeros after the valid length.
    pub expected: Vec<u8>,
    /// The valid data length, when below the data length.
    pub valid: Option<u64>,
    /// Byte offset of the set's primary (its node id).
    pub record: u64,
    /// Clusters of the content, in order.
    pub clusters: Vec<u32>,
    /// A damaged set the reader skips.
    pub skipped: bool,
    /// The primary is the last slot of its cluster.
    pub straddles: bool,
}

/// A built volume.
pub struct Built {
    /// The whole image.
    pub image: Vec<u8>,
    /// Bytes per cluster.
    pub cluster_size: u64,
    /// Byte offset of cluster 2.
    pub heap_offset: u64,
    /// Byte offset of the first FAT.
    pub fat_offset: u64,
    /// Clusters in the heap.
    pub cluster_count: u32,
    /// The root directory's first cluster.
    pub root_cluster: u32,
    /// Clusters marked used in the bitmap.
    pub used_clusters: usize,
    /// The entries, in the order they were added.
    pub items: Vec<Item>,
    used_end: u64,
}

impl Built {
    /// The image without its trailing unused clusters.
    pub fn trimmed(&self) -> Vec<u8> {
        let end = usize::try_from(self.used_end).unwrap();
        self.image[..end.min(self.image.len())].to_vec()
    }

    /// The item at `path`.
    pub fn item(&self, path: &str) -> &Item {
        self.items
            .iter()
            .find(|i| i.path == path)
            .unwrap_or_else(|| panic!("no item {path}"))
    }
}

struct DirModel {
    entries: Vec<Spec>,
    /// Index into `dirs` of the folder an entry is, by entry position.
    children: Vec<Option<usize>>,
    path: String,
}

/// Builds a volume. Directory 0 is the root.
pub struct VolumeBuilder {
    label: Option<String>,
    dirs: Vec<DirModel>,
}

impl Default for VolumeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn name_slots(name: &str) -> usize {
    name.encode_utf16().count().div_ceil(15)
}

fn set_checksum(set: &[[u8; SLOT]]) -> u16 {
    let mut sum = 0u16;
    for (i, slot) in set.iter().enumerate() {
        for (j, &b) in slot.iter().enumerate() {
            if i == 0 && (j == 2 || j == 3) {
                continue;
            }
            sum = sum.rotate_right(1).wrapping_add(u16::from(b));
        }
    }
    sum
}

impl VolumeBuilder {
    /// An empty volume.
    pub fn new() -> Self {
        Self {
            label: None,
            dirs: vec![DirModel {
                entries: Vec::new(),
                children: Vec::new(),
                path: String::new(),
            }],
        }
    }

    /// The volume label (at most 11 characters).
    pub fn label(&mut self, label: &str) -> &mut Self {
        self.label = Some(label.to_owned());
        self
    }

    /// Adds a folder to `parent`; returns its index.
    pub fn dir(&mut self, parent: usize, name: &str) -> usize {
        let path = format!("{}/{name}", self.dirs[parent].path);
        let index = self.dirs.len();
        self.dirs.push(DirModel {
            entries: Vec::new(),
            children: Vec::new(),
            path,
        });
        let p = &mut self.dirs[parent];
        p.entries.push(Spec::folder(name));
        p.children.push(Some(index));
        index
    }

    /// Adds a file to `parent`.
    pub fn file(&mut self, parent: usize, name: &str, data: &[u8]) -> &mut Self {
        self.file_in(parent, Spec::file(name, data))
    }

    /// Adds a file spec to the root.
    pub fn file_spec(&mut self, spec: Spec) -> &mut Self {
        self.file_in(0, spec)
    }

    /// Adds a file spec to `parent`.
    pub fn file_in(&mut self, parent: usize, spec: Spec) -> &mut Self {
        let p = &mut self.dirs[parent];
        p.entries.push(spec);
        p.children.push(None);
        self
    }

    /// Slots a directory takes.
    fn slot_count(&self, d: usize) -> usize {
        let dir = &self.dirs[d];
        let mut n = if d == 0 { 2 } else { 0 };
        for e in &dir.entries {
            n += 2 + name_slots(&e.name);
        }
        n.max(1)
    }

    /// Builds the image.
    pub fn build(&self) -> Built {
        // Clusters per directory, then the allocation order.
        let dir_clusters: Vec<usize> = (0..self.dirs.len())
            .map(|d| self.slot_count(d).div_ceil(SLOTS_PER_CLUSTER))
            .collect();
        let mut next = 2u32;
        let bitmap_cluster = next;
        next += 1;
        let mut dir_lists: Vec<Vec<u32>> = vec![Vec::new(); self.dirs.len()];
        let rounds = dir_clusters.iter().copied().max().unwrap_or(1);
        for round in 0..rounds {
            for (d, list) in dir_lists.iter_mut().enumerate() {
                if round < dir_clusters[d] {
                    list.push(next);
                    next += 1;
                }
            }
        }
        // File clusters, in directory then entry order.
        struct Alloc {
            clusters: Vec<u32>,
            first: u32,
        }
        let mut allocs: Vec<Vec<Alloc>> = Vec::new();
        let mut used: Vec<u32> = vec![bitmap_cluster];
        used.extend(dir_lists.iter().flatten());
        let mut chained: Vec<Vec<u32>> = vec![vec![bitmap_cluster]];
        chained.extend(dir_lists.iter().cloned());
        for dir in &self.dirs {
            let mut list = Vec::new();
            for (i, e) in dir.entries.iter().enumerate() {
                let Some(data) = &e.data else {
                    list.push(Alloc {
                        clusters: Vec::new(),
                        first: 0,
                    });
                    let _ = i;
                    continue;
                };
                if e.damage.is_some() {
                    list.push(Alloc {
                        clusters: Vec::new(),
                        first: 0,
                    });
                    continue;
                }
                let count = data.len().div_ceil(CLUSTER);
                let mut clusters = Vec::new();
                if e.first_override.is_none() {
                    for _ in 0..count {
                        clusters.push(next);
                        next += if e.fragmented { 2 } else { 1 };
                    }
                    used.extend(&clusters);
                    if e.fragmented && clusters.len() > 1 {
                        chained.push(clusters.clone());
                    }
                }
                let first = e
                    .first_override
                    .unwrap_or_else(|| clusters.first().copied().unwrap_or(0));
                list.push(Alloc { clusters, first });
            }
            allocs.push(list);
        }
        let cluster_count = next - 2 + 8;
        let fat_sector = 4u32;
        let fat_sectors = ((cluster_count + 2) * 4).div_ceil(512);
        let heap_sector = fat_sector + fat_sectors + 2;
        let heap_offset = u64::from(heap_sector) * 512;
        let fat_offset = u64::from(fat_sector) * 512;
        let total_sectors = u64::from(heap_sector) + u64::from(cluster_count);
        let mut image = vec![0u8; usize::try_from(total_sectors * 512).unwrap()];
        let at = |c: u32| heap_offset as usize + (c as usize - 2) * CLUSTER;

        // Boot sector.
        {
            let b = &mut image[..512];
            b[3..11].copy_from_slice(b"EXFAT   ");
            b[64..72].copy_from_slice(&0u64.to_le_bytes());
            b[72..80].copy_from_slice(&total_sectors.to_le_bytes());
            b[80..84].copy_from_slice(&fat_sector.to_le_bytes());
            b[84..88].copy_from_slice(&fat_sectors.to_le_bytes());
            b[88..92].copy_from_slice(&heap_sector.to_le_bytes());
            b[92..96].copy_from_slice(&cluster_count.to_le_bytes());
            b[96..100].copy_from_slice(&dir_lists[0][0].to_le_bytes());
            b[100..104].copy_from_slice(&0x1234_5678u32.to_le_bytes());
            b[104..106].copy_from_slice(&0x0100u16.to_le_bytes());
            b[108] = 9;
            b[109] = 0;
            b[110] = 1;
            b[510] = 0x55;
            b[511] = 0xAA;
        }
        // FAT.
        let fat = fat_offset as usize;
        image[fat..fat + 4].copy_from_slice(&0xFFFF_FFF8u32.to_le_bytes());
        image[fat + 4..fat + 8].copy_from_slice(&END.to_le_bytes());
        for chain in &chained {
            for (i, &c) in chain.iter().enumerate() {
                let value = chain.get(i + 1).copied().unwrap_or(END);
                let p = fat + c as usize * 4;
                image[p..p + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        // Bitmap.
        let bitmap_at = at(bitmap_cluster);
        for &c in &used {
            let bit = (c - 2) as usize;
            image[bitmap_at + bit / 8] |= 1 << (bit % 8);
        }

        // Directories and items.
        let mut items = Vec::new();
        let mut used_end = 0usize;
        for (d, dir) in self.dirs.iter().enumerate() {
            let mut slots: Vec<[u8; SLOT]> = Vec::new();
            if d == 0 {
                let mut bm = [0u8; SLOT];
                bm[0] = 0x81;
                bm[20..24].copy_from_slice(&bitmap_cluster.to_le_bytes());
                bm[24..32].copy_from_slice(&(u64::from(cluster_count).div_ceil(8)).to_le_bytes());
                slots.push(bm);
                let mut label = [0u8; SLOT];
                label[0] = 0x83;
                let l = self.label.clone().unwrap_or_default();
                let units: Vec<u16> = l.encode_utf16().take(11).collect();
                label[1] = units.len() as u8;
                for (i, u) in units.iter().enumerate() {
                    label[2 + 2 * i..4 + 2 * i].copy_from_slice(&u.to_le_bytes());
                }
                if units.is_empty() {
                    label[0] = 0x03;
                }
                slots.push(label);
            }
            for (i, e) in dir.entries.iter().enumerate() {
                let alloc = &allocs[d][i];
                let record_slot = slots.len();
                let units: Vec<u16> = e.name.encode_utf16().collect();
                let n_names = units.len().div_ceil(15);
                let len = e.data.as_ref().map_or(0, Vec::len) as u64;
                let child = dir.children[i];
                let (first, data_len, flags, valid) = if let Some(ci) = child {
                    (
                        dir_lists[ci][0],
                        (dir_clusters[ci] * CLUSTER) as u64,
                        0x01u8,
                        (dir_clusters[ci] * CLUSTER) as u64,
                    )
                } else {
                    let contiguous = !e.fragmented;
                    let flags = if contiguous { 0x03 } else { 0x01 };
                    (alloc.first, len, flags, e.valid.unwrap_or(len).min(len))
                };
                let mut primary = [0u8; SLOT];
                primary[0] = 0x85;
                primary[1] = (1 + n_names) as u8;
                let attr: u16 =
                    (if child.is_some() { 0x10 } else { 0x20 }) | if e.hidden { 0x06 } else { 0 };
                primary[4..6].copy_from_slice(&attr.to_le_bytes());
                let packed = (u32::from(DATE) << 16) | u32::from(TIME);
                primary[8..12].copy_from_slice(&packed.to_le_bytes());
                primary[12..16].copy_from_slice(&packed.to_le_bytes());
                primary[16..20].copy_from_slice(&packed.to_le_bytes());
                primary[22] = e.created_offset;
                let mut stream = [0u8; SLOT];
                stream[0] = 0xC0;
                stream[1] = flags;
                stream[3] = units.len() as u8;
                stream[8..16].copy_from_slice(&valid.to_le_bytes());
                stream[20..24].copy_from_slice(&first.to_le_bytes());
                stream[24..32].copy_from_slice(&data_len.to_le_bytes());
                let mut names: Vec<[u8; SLOT]> = Vec::new();
                for chunk in units.chunks(15) {
                    let mut n = [0u8; SLOT];
                    n[0] = 0xC1;
                    for (k, u) in chunk.iter().enumerate() {
                        n[2 + 2 * k..4 + 2 * k].copy_from_slice(&u.to_le_bytes());
                    }
                    names.push(n);
                }
                match e.damage {
                    Some(Damage::CountOne) => primary[1] = 1,
                    Some(Damage::BadNameType) => {
                        if let Some(n) = names.first_mut() {
                            n[0] = 0xC2;
                        }
                    }
                    Some(Damage::NoName) => stream[3] = 0,
                    None => {}
                }
                let mut set = vec![primary, stream];
                set.extend(names);
                let sum = set_checksum(&set);
                set[0][2..4].copy_from_slice(&sum.to_le_bytes());
                slots.extend(set);

                let cluster_of = |slot: usize| dir_lists[d][slot / SLOTS_PER_CLUSTER];
                let record =
                    (at(cluster_of(record_slot)) + (record_slot % SLOTS_PER_CLUSTER) * SLOT) as u64;
                let data = e.data.clone().unwrap_or_default();
                let mut expected = data.clone();
                if let Some(v) = e.valid {
                    for b in expected.iter_mut().skip(v as usize) {
                        *b = 0;
                    }
                }
                // File content.
                if child.is_none() && e.damage.is_none() {
                    for (k, &c) in alloc.clusters.iter().enumerate() {
                        let from = k * CLUSTER;
                        let to = (from + CLUSTER).min(data.len());
                        image[at(c)..at(c) + (to - from)].copy_from_slice(&data[from..to]);
                        used_end = used_end.max(at(c) + CLUSTER);
                    }
                }
                items.push(Item {
                    path: format!("{}/{}", dir.path, e.name),
                    is_dir: child.is_some(),
                    data,
                    expected,
                    valid: e.valid,
                    record,
                    clusters: alloc.clusters.clone(),
                    skipped: e.damage.is_some(),
                    straddles: record_slot % SLOTS_PER_CLUSTER == SLOTS_PER_CLUSTER - 1,
                });
            }
            for (k, slot) in slots.iter().enumerate() {
                let c = dir_lists[d][k / SLOTS_PER_CLUSTER];
                let p = at(c) + (k % SLOTS_PER_CLUSTER) * SLOT;
                image[p..p + SLOT].copy_from_slice(slot);
            }
            for &c in &dir_lists[d] {
                used_end = used_end.max(at(c) + CLUSTER);
            }
        }
        used_end = used_end.max(bitmap_at + CLUSTER);
        Built {
            image,
            cluster_size: CLUSTER as u64,
            heap_offset,
            fat_offset,
            cluster_count,
            root_cluster: dir_lists[0][0],
            used_clusters: used.len(),
            items,
            used_end: used_end as u64,
        }
    }
}

/// The standard volume:
///
/// ```text
/// /docs/<LONG_NAME>      700 bytes, contiguous
/// /docs/notes.txt         90 bytes
/// /docs/deep/a.bin        40 bytes
/// /frag.bin             2400 bytes, chained in the FAT with gaps
/// /partial.bin          1300 bytes, contiguous, 700 of them valid,
///                        created at +1:00
/// /secret.sys             20 bytes, hidden and system
/// /damaged1 .. /damaged3  sets the reader must skip
/// /after-damage.txt       10 bytes
/// ```
pub fn standard() -> Built {
    standard_builder().build()
}

/// The builder behind [`standard`], for variations.
pub fn standard_builder() -> VolumeBuilder {
    let mut b = VolumeBuilder::new();
    b.label("ATLAS");
    let docs = b.dir(0, "docs");
    b.file(docs, LONG_NAME, &pattern(700, 1));
    b.file(docs, "notes.txt", &pattern(90, 2));
    let deep = b.dir(docs, "deep");
    b.file(deep, "a.bin", &pattern(40, 3));
    b.file_spec(Spec::file("frag.bin", &pattern(2400, 4)).fragmented());
    b.file_spec(
        Spec::file("partial.bin", &pattern(1300, 5))
            .valid(700)
            .created_offset(0x80 | 4),
    );
    b.file_spec(Spec::file("secret.sys", &pattern(20, 6)).hidden());
    b.file_spec(Spec::file("damaged1", &[1]).damaged(Damage::CountOne));
    b.file_spec(Spec::file("damaged2", &[1]).damaged(Damage::BadNameType));
    b.file_spec(Spec::file("damaged3", &[1]).damaged(Damage::NoName));
    b.file(0, "after-damage.txt", &pattern(10, 7));
    b
}

/// A volume whose two big directories have clusters that are not adjacent
/// and sets that cross from one directory cluster to the next.
pub fn crowded() -> Built {
    let mut b = VolumeBuilder::new();
    b.label("CROWD");
    let docs = b.dir(0, "docs");
    for i in 0..14 {
        b.file(0, &format!("r{i:02}.bin"), &pattern(30 + i, i as u8));
        b.file(
            docs,
            &format!("d{i:02}.bin"),
            &pattern(600 + i, 40 + i as u8),
        );
    }
    b.build()
}
