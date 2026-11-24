// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The exFAT main boot sector and the geometry it gives.
//!
//! Little-endian. Every field is checked before any byte offset is made
//! from it: the shifts first, then the sector counts, with checked
//! arithmetic throughout.

use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};

/// Bytes of the boot sector this module reads.
pub const BOOT_LEN: usize = 512;

/// Most clusters a volume can have (the spec's `0xFFFFFFF5`).
pub const MAX_CLUSTERS: u32 = 0xFFFF_FFF5;

const STRUCTURE: &str = "exFAT boot sector";

/// What the boot sector says, in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Geometry {
    /// Bytes per sector (512 to 4096).
    pub bytes_per_sector: u32,
    /// Bytes per cluster.
    pub cluster_size: u32,
    /// Clusters in the heap.
    pub cluster_count: u32,
    /// Byte offset of the first FAT.
    pub fat_offset: u64,
    /// Bytes of the first FAT.
    pub fat_bytes: u64,
    /// Byte offset of cluster 2.
    pub heap_offset: u64,
    /// First cluster of the root directory.
    pub root_cluster: u32,
    /// Bytes of the volume.
    pub volume_bytes: u64,
    /// Volume serial number.
    pub serial: u32,
}

impl Geometry {
    /// The last valid cluster number.
    pub fn last_cluster(&self) -> u32 {
        self.cluster_count.saturating_add(1)
    }

    /// The byte offset of cluster `n`, or `None` outside 2 ..= last.
    pub fn cluster_offset(&self, n: u32) -> Option<u64> {
        if n < 2 || n > self.last_cluster() {
            return None;
        }
        u64::from(n.saturating_sub(2))
            .checked_mul(u64::from(self.cluster_size))?
            .checked_add(self.heap_offset)
    }

    /// Bytes of the heap.
    pub fn heap_bytes(&self) -> u64 {
        u64::from(self.cluster_count).saturating_mul(u64::from(self.cluster_size))
    }
}

fn err(kind: CorruptKind) -> Error {
    Corruption::new(STRUCTURE, Location::Unspecified, kind).into()
}

fn invalid(field: &'static str, value: u64) -> Error {
    err(CorruptKind::InvalidValue { field, value })
}

fn le16(b: &[u8], at: usize) -> u16 {
    b.get(at..at.saturating_add(2))
        .and_then(|s| <[u8; 2]>::try_from(s).ok())
        .map_or(0, u16::from_le_bytes)
}

fn le32(b: &[u8], at: usize) -> u32 {
    b.get(at..at.saturating_add(4))
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .map_or(0, u32::from_le_bytes)
}

fn le64(b: &[u8], at: usize) -> u64 {
    b.get(at..at.saturating_add(8))
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .map_or(0, u64::from_le_bytes)
}

/// True when `sector` looks like an exFAT boot sector (name and signature).
pub fn looks_like(sector: &[u8]) -> bool {
    sector.get(3..11) == Some(b"EXFAT   ") && le16(sector, 510) == 0xAA55
}

/// Parses the main boot sector.
///
/// Fails with `Corrupt` when the name or signature is wrong, a shift is out
/// of range, the FAT is too short for the cluster count, the heap starts
/// inside the FAT or runs past the volume, or the root cluster is outside
/// the heap.
pub fn parse(sector: &[u8]) -> Result<Geometry> {
    if sector.len() < BOOT_LEN {
        return Err(err(CorruptKind::Truncated {
            needed: u64::try_from(BOOT_LEN).unwrap_or(u64::MAX),
            available: u64::try_from(sector.len()).unwrap_or(u64::MAX),
        }));
    }
    let signature = le16(sector, 510);
    if signature != 0xAA55 {
        return Err(err(CorruptKind::BadSignature {
            expected: 0xAA55,
            found: u64::from(signature),
        }));
    }
    if sector.get(3..11) != Some(b"EXFAT   ") {
        return Err(invalid("file system name", le64(sector, 3)));
    }
    let bps_shift = u32::from(sector.get(108).copied().unwrap_or(0));
    let spc_shift = u32::from(sector.get(109).copied().unwrap_or(0));
    if !(9..=12).contains(&bps_shift) {
        return Err(invalid("bytes per sector shift", u64::from(bps_shift)));
    }
    if bps_shift.saturating_add(spc_shift) > 25 {
        return Err(invalid("sectors per cluster shift", u64::from(spc_shift)));
    }
    let bps = 1u64 << bps_shift;
    let sectors = |v: u64| v.checked_mul(bps);

    let volume_sectors = le64(sector, 72);
    let fat_sector = u64::from(le32(sector, 80));
    let fat_sectors = u64::from(le32(sector, 84));
    let heap_sector = u64::from(le32(sector, 88));
    let cluster_count = le32(sector, 92);
    let root_cluster = le32(sector, 96);
    let serial = le32(sector, 100);

    if cluster_count == 0 || cluster_count > MAX_CLUSTERS {
        return Err(invalid("cluster count", u64::from(cluster_count)));
    }
    if root_cluster < 2 || root_cluster > cluster_count.saturating_add(1) {
        return Err(invalid("root directory cluster", u64::from(root_cluster)));
    }
    let overflow = |field| err(CorruptKind::Overflow { field });
    let fat_needed = (u64::from(cluster_count).saturating_add(2))
        .checked_mul(4)
        .ok_or_else(|| overflow("FAT size"))?;
    let fat_bytes = sectors(fat_sectors).ok_or_else(|| overflow("FAT length"))?;
    if fat_sector == 0 || fat_bytes < fat_needed {
        return Err(err(CorruptKind::OutOfRange {
            field: "FAT length",
            value: fat_bytes,
            limit_name: "bytes the cluster count needs",
            limit: fat_needed,
        }));
    }
    let fat_end = fat_sector
        .checked_add(fat_sectors)
        .ok_or_else(|| overflow("FAT end"))?;
    if heap_sector < fat_end {
        return Err(invalid("cluster heap offset", heap_sector));
    }
    let cluster_sectors = 1u64 << spc_shift;
    let heap_end = u64::from(cluster_count)
        .checked_mul(cluster_sectors)
        .and_then(|n| n.checked_add(heap_sector))
        .ok_or_else(|| overflow("heap end"))?;
    if heap_end > volume_sectors {
        return Err(err(CorruptKind::OutOfRange {
            field: "cluster heap end",
            value: heap_end,
            limit_name: "volume length in sectors",
            limit: volume_sectors,
        }));
    }
    Ok(Geometry {
        bytes_per_sector: 1u32 << bps_shift,
        cluster_size: 1u32 << (bps_shift.saturating_add(spc_shift)),
        cluster_count,
        fat_offset: sectors(fat_sector).ok_or_else(|| overflow("FAT offset"))?,
        fat_bytes: sectors(fat_sectors).ok_or_else(|| overflow("FAT length"))?,
        heap_offset: sectors(heap_sector).ok_or_else(|| overflow("heap offset"))?,
        root_cluster,
        volume_bytes: sectors(volume_sectors).ok_or_else(|| overflow("volume length"))?,
        serial,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid sector: 512-byte sectors, 2 sectors a cluster, 100 clusters.
    fn valid() -> Vec<u8> {
        let mut b = vec![0u8; BOOT_LEN];
        b[3..11].copy_from_slice(b"EXFAT   ");
        b[72..80].copy_from_slice(&1000u64.to_le_bytes());
        b[80..84].copy_from_slice(&8u32.to_le_bytes());
        b[84..88].copy_from_slice(&1u32.to_le_bytes());
        b[88..92].copy_from_slice(&16u32.to_le_bytes());
        b[92..96].copy_from_slice(&100u32.to_le_bytes());
        b[96..100].copy_from_slice(&3u32.to_le_bytes());
        b[100..104].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        b[108] = 9;
        b[109] = 1;
        b[510] = 0x55;
        b[511] = 0xAA;
        b
    }

    #[test]
    fn a_valid_sector_parses() {
        let g = parse(&valid()).unwrap();
        assert_eq!(g.bytes_per_sector, 512);
        assert_eq!(g.cluster_size, 1024);
        assert_eq!(g.fat_offset, 8 * 512);
        assert_eq!(g.heap_offset, 16 * 512);
        assert_eq!(g.cluster_offset(2), Some(16 * 512));
        assert_eq!(g.cluster_offset(3), Some(16 * 512 + 1024));
        assert_eq!(g.cluster_offset(1), None);
        assert_eq!(g.cluster_offset(102), None);
        assert_eq!(g.serial, 0xDEAD_BEEF);
        assert!(looks_like(&valid()));
    }

    #[test]
    fn malformed_sectors_are_corrupt() {
        type Edit = fn(&mut Vec<u8>);
        let edits: [(&str, Edit); 14] = [
            ("signature", |b| b[510] = 0),
            ("name", |b| b[3] = b'X'),
            ("short", |b| b.truncate(100)),
            ("bps shift low", |b| b[108] = 8),
            ("bps shift high", |b| b[108] = 13),
            ("shift sum", |b| b[109] = 17),
            ("zero clusters", |b| b[92..96].fill(0)),
            ("too many clusters", |b| b[92..96].fill(0xFF)),
            ("root below 2", |b| {
                b[96..100].copy_from_slice(&1u32.to_le_bytes())
            }),
            ("root past heap", |b| {
                b[96..100].copy_from_slice(&200u32.to_le_bytes())
            }),
            ("fat too short", |b| b[84..88].fill(0)),
            ("fat at zero", |b| b[80..84].fill(0)),
            ("heap inside fat", |b| {
                b[88..92].copy_from_slice(&8u32.to_le_bytes())
            }),
            ("heap past volume", |b| {
                b[72..80].copy_from_slice(&100u64.to_le_bytes())
            }),
        ];
        for (name, edit) in edits {
            let mut b = valid();
            edit(&mut b);
            let r = parse(&b);
            assert!(
                matches!(r, Err(Error::Corrupt(_))),
                "{name}: expected Corrupt, got {r:?}"
            );
        }
        // Huge counts overflow checked arithmetic instead of wrapping.
        let mut b = valid();
        b[72..80].fill(0xFF);
        b[88..92].fill(0xFF);
        b[84..88].fill(0xFF);
        b[80..84].copy_from_slice(&1u32.to_le_bytes());
        b[92..96].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        b[96..100].copy_from_slice(&2u32.to_le_bytes());
        b[108] = 12;
        b[109] = 13;
        assert!(parse(&b).is_err());
    }
}
