// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! The exFAT file allocation table and cluster chains made from it.

use std::sync::Arc;

use super::boot::Geometry;
use atlas_fat::chain::{Chain, Run};
use atlas_fs_core::error::{CorruptKind, Corruption, Location, Result};
use atlas_fs_core::io::block::BlockSource;

/// The FAT is kept in memory when it is at most this many bytes.
pub const MEM_LIMIT: u64 = 64 * 1024 * 1024;

/// Most runs a chain may have. A hostile FAT that alternates clusters would
/// otherwise make a run per cluster.
pub const MAX_RUNS: usize = 1 << 22;

const STRUCTURE: &str = "exFAT cluster chain";

/// What a FAT entry says about the cluster after the one it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Next {
    /// The chain ends here.
    End,
    /// The cluster is marked bad.
    Bad,
    /// The chain continues at this cluster, a valid cluster number.
    Data(u32),
    /// Free (0), reserved, or out of range.
    Invalid(u32),
}

/// The first FAT of a volume.
pub struct ExTable {
    mem: Option<Vec<u8>>,
    source: Arc<dyn BlockSource>,
    offset: u64,
    last_cluster: u32,
}

impl ExTable {
    /// Opens the table `geo` describes. Fails with `Corrupt` when the source
    /// ends inside it.
    pub fn load(source: Arc<dyn BlockSource>, geo: &Geometry) -> Result<Self> {
        let needed = u64::from(geo.cluster_count)
            .saturating_add(2)
            .saturating_mul(4);
        let end = geo.fat_offset.checked_add(needed);
        if !end.is_some_and(|e| e <= source.len()) {
            return Err(Corruption::new(
                STRUCTURE,
                Location::Byte(geo.fat_offset),
                CorruptKind::Truncated {
                    needed,
                    available: source.len().saturating_sub(geo.fat_offset),
                },
            )
            .into());
        }
        let mem = if needed <= MEM_LIMIT {
            let len = usize::try_from(needed).unwrap_or(0);
            let mut bytes = vec![0u8; len];
            source.read_bulk(geo.fat_offset, &mut bytes)?;
            Some(bytes)
        } else {
            None
        };
        Ok(Self {
            mem,
            source,
            offset: geo.fat_offset,
            last_cluster: geo.last_cluster(),
        })
    }

    /// The raw entry for cluster `n`.
    pub fn entry(&self, n: u32) -> Result<u32> {
        if n > self.last_cluster {
            return Ok(0);
        }
        let at = u64::from(n).saturating_mul(4);
        let mut b = [0u8; 4];
        if let Some(mem) = &self.mem {
            let at = usize::try_from(at).unwrap_or(usize::MAX);
            if let Some(src) = mem.get(at..at.saturating_add(4)) {
                b.copy_from_slice(src);
            }
        } else {
            self.source
                .read_at(self.offset.saturating_add(at), &mut b)?;
        }
        Ok(u32::from_le_bytes(b))
    }

    /// What follows cluster `n`.
    pub fn next(&self, n: u32) -> Result<Next> {
        let v = self.entry(n)?;
        Ok(match v {
            0xFFFF_FFFF => Next::End,
            0xFFFF_FFF7 => Next::Bad,
            v if v < 2 || v > self.last_cluster => Next::Invalid(v),
            v => Next::Data(v),
        })
    }
}

fn corrupt(location: Location, kind: CorruptKind) -> Corruption {
    Corruption::new(STRUCTURE, location, kind)
}

/// Follows the FAT chain from `first` for at most `max` clusters, as runs.
///
/// Damage (a bad, free or out-of-range entry, a loop, too many runs) ends
/// the chain and is kept in it, reported when a read reaches it. Reaching
/// `max` clusters is not damage: the caller knows how many it needs. An
/// `Err` is only a failure to read the FAT.
pub fn follow(
    geo: &Geometry,
    table: &ExTable,
    source: Arc<dyn BlockSource>,
    first: u32,
    max: u64,
) -> Result<Chain> {
    let cs = u64::from(geo.cluster_size);
    let mut runs: Vec<Run> = Vec::new();
    let mut count = 0u64;
    let mut cur = first;
    let (mut tortoise, mut power, mut lam) = (first, 1u64, 0u64);
    let broken = loop {
        if count >= max {
            break None;
        }
        let Some(disk) = geo.cluster_offset(cur) else {
            break Some(corrupt(
                Location::Object(u64::from(cur)),
                CorruptKind::OutOfRange {
                    field: "cluster",
                    value: u64::from(cur),
                    limit_name: "last cluster",
                    limit: u64::from(geo.last_cluster()),
                },
            ));
        };
        let file_start = count.saturating_mul(cs);
        match runs.last_mut() {
            Some(last) if last.disk.checked_add(last.len) == Some(disk) => {
                last.len = last.len.saturating_add(cs);
            }
            _ => {
                if runs.len() >= MAX_RUNS {
                    break Some(corrupt(
                        Location::Object(u64::from(first)),
                        CorruptKind::LimitExceeded {
                            what: "runs in a cluster chain",
                            limit: u64::try_from(MAX_RUNS).unwrap_or(u64::MAX),
                        },
                    ));
                }
                runs.push(Run {
                    file_start,
                    disk,
                    len: cs,
                });
            }
        }
        count = count.saturating_add(1);
        if count >= max {
            break None;
        }
        match table.next(cur)? {
            Next::End => break None,
            Next::Data(n) => {
                if count > u64::from(geo.cluster_count) || n == tortoise {
                    break Some(corrupt(
                        Location::Object(u64::from(first)),
                        CorruptKind::Loop {
                            what: "cluster chain",
                        },
                    ));
                }
                lam = lam.saturating_add(1);
                if lam == power {
                    tortoise = n;
                    power = power.saturating_mul(2);
                    lam = 0;
                }
                cur = n;
            }
            Next::Bad | Next::Invalid(_) => {
                break Some(corrupt(
                    Location::Object(u64::from(cur)),
                    CorruptKind::Other("chain runs into a free, bad or invalid cluster"),
                ));
            }
        }
    };
    Ok(Chain::from_runs(source, runs, broken))
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_fs_core::io::block::MemSource;

    fn geo() -> Geometry {
        Geometry {
            bytes_per_sector: 512,
            cluster_size: 512,
            cluster_count: 20,
            fat_offset: 0,
            fat_bytes: 512,
            heap_offset: 512,
            root_cluster: 2,
            volume_bytes: 512 + 20 * 512,
            serial: 0,
        }
    }

    /// A FAT of 22 entries from `(cluster, value)` pairs, and the table.
    fn table(entries: &[(u32, u32)]) -> (ExTable, Arc<dyn BlockSource>) {
        let mut bytes = vec![0u8; 512 + 20 * 512];
        for &(n, v) in entries {
            let at = usize::try_from(n).unwrap() * 4;
            bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        let source: Arc<dyn BlockSource> = Arc::new(MemSource::new(bytes));
        (ExTable::load(Arc::clone(&source), &geo()).unwrap(), source)
    }

    const END: u32 = 0xFFFF_FFFF;

    #[test]
    fn a_chain_merges_adjacent_clusters_and_splits_others() {
        let (t, s) = table(&[(2, 3), (3, 7), (7, 8), (8, END)]);
        let c = follow(&geo(), &t, s, 2, 100).unwrap();
        assert!(c.broken().is_none());
        let runs = c.runs();
        assert_eq!(runs.len(), 2);
        assert_eq!((runs[0].disk, runs[0].len), (512, 1024));
        assert_eq!(
            (runs[1].file_start, runs[1].disk, runs[1].len),
            (1024, 512 + 5 * 512, 1024)
        );
    }

    #[test]
    fn the_limit_stops_a_chain_quietly() {
        let (t, s) = table(&[(2, 3), (3, 4), (4, END)]);
        let c = follow(&geo(), &t, s, 2, 2).unwrap();
        assert!(c.broken().is_none());
        assert_eq!(c.len(), 1024);
    }

    #[test]
    fn damage_is_kept_for_the_read_that_reaches_it() {
        // Free entry, bad cluster, out of range, reserved, loop.
        for entries in [
            vec![(2, 3), (3, 0)],
            vec![(2, 3), (3, 0xFFFF_FFF7)],
            vec![(2, 3), (3, 500)],
            vec![(2, 3), (3, 1)],
            vec![(2, 3), (3, 4), (4, 3)],
            vec![(2, 2)],
        ] {
            let (t, s) = table(&entries);
            let c = follow(&geo(), &t, s, 2, 1000).unwrap();
            assert!(c.broken().is_some(), "{entries:?}");
            assert!(c.len() >= 512 && c.len() <= 20 * 512, "{entries:?}");
        }
    }

    #[test]
    fn a_first_cluster_outside_the_heap_is_damage() {
        let (t, s) = table(&[]);
        for first in [0u32, 1, 22, u32::MAX] {
            let c = follow(&geo(), &t, Arc::clone(&s), first, 5).unwrap();
            assert!(c.broken().is_some(), "{first}");
            assert_eq!(c.len(), 0);
        }
    }

    #[test]
    fn loading_fails_when_the_source_ends_inside_the_fat() {
        let g = geo();
        let source: Arc<dyn BlockSource> = Arc::new(MemSource::new(vec![0u8; 40]));
        assert!(ExTable::load(source, &g).is_err());
    }
}
