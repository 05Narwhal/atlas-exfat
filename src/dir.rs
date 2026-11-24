// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! exFAT directory entries: file entry sets, timestamps, the label and the
//! allocation bitmap entry.
//!
//! A directory is a run of 32-byte slots. A file is a *set*: a `0x85`
//! primary, a `0xC0` stream extension and `0xC1` name entries. A set that
//! is malformed is skipped from its primary slot on and the walk goes on
//! with the next slot.

use std::sync::Arc;

use atlas_fat::chain::Chain;
use atlas_fs_core::error::Result;
use atlas_fs_core::fs::{EntryFlags, Timestamp};
use atlas_fs_core::io::block::BlockSource;

/// Bytes of a directory slot.
pub const SLOT: usize = 32;

/// Entry type: allocation bitmap.
pub const TYPE_BITMAP: u8 = 0x81;
/// Entry type: volume label.
pub const TYPE_LABEL: u8 = 0x83;
/// Entry type: file (the primary of a set).
pub const TYPE_FILE: u8 = 0x85;
/// Entry type: stream extension.
pub const TYPE_STREAM: u8 = 0xC0;
/// Entry type: file name.
pub const TYPE_NAME: u8 = 0xC1;

/// Attribute: hidden.
pub const ATTR_HIDDEN: u16 = 0x02;
/// Attribute: system.
pub const ATTR_SYSTEM: u16 = 0x04;
/// Attribute: directory.
pub const ATTR_DIRECTORY: u16 = 0x10;

/// Stream flag: the content is contiguous and has no FAT chain.
pub const FLAG_NO_FAT_CHAIN: u8 = 0x02;

/// Name units in one name entry.
const UNITS_PER_NAME_ENTRY: usize = 15;

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

/// What a file's set says, without the name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Meta {
    /// Attribute word of the primary.
    pub attr: u16,
    /// Creation: date in the high 16 bits, time in the low 16.
    pub created: u32,
    /// Modification, packed the same way.
    pub modified: u32,
    /// Last access, packed the same way.
    pub accessed: u32,
    /// 10 ms increments (0 to 199) of the creation and modification times.
    pub created_10ms: u8,
    /// See `created_10ms`.
    pub modified_10ms: u8,
    /// UTC offset bytes: bit 7 set = valid, low 7 bits signed quarter-hours.
    pub created_offset: u8,
    /// See `created_offset`.
    pub modified_offset: u8,
    /// See `created_offset`.
    pub accessed_offset: u8,
    /// Flags of the stream extension.
    pub flags: u8,
    /// Bytes that hold data; the rest of `data_len` reads as zeros.
    pub valid_len: u64,
    /// First cluster of the content; 0 when empty.
    pub first_cluster: u32,
    /// Bytes of the content.
    pub data_len: u64,
}

impl Meta {
    /// Reads a primary (`0x85`) and its stream extension (`0xC0`).
    pub fn parse(primary: &[u8], stream: &[u8]) -> Option<Self> {
        if primary.first() != Some(&TYPE_FILE) || stream.first() != Some(&TYPE_STREAM) {
            return None;
        }
        let count = primary.get(1).copied().unwrap_or(0);
        if !(2..=18).contains(&count) {
            return None;
        }
        let data_len = le64(stream, 24);
        Some(Self {
            attr: le16(primary, 4),
            created: le32(primary, 8),
            modified: le32(primary, 12),
            accessed: le32(primary, 16),
            created_10ms: primary.get(20).copied().unwrap_or(0),
            modified_10ms: primary.get(21).copied().unwrap_or(0),
            created_offset: primary.get(22).copied().unwrap_or(0),
            modified_offset: primary.get(23).copied().unwrap_or(0),
            accessed_offset: primary.get(24).copied().unwrap_or(0),
            flags: stream.get(1).copied().unwrap_or(0),
            valid_len: le64(stream, 8).min(data_len),
            first_cluster: le32(stream, 20),
            data_len,
        })
    }

    /// True for a folder.
    pub fn is_dir(&self) -> bool {
        self.attr & ATTR_DIRECTORY != 0
    }

    /// True when the content has no FAT chain.
    pub fn contiguous(&self) -> bool {
        self.flags & FLAG_NO_FAT_CHAIN != 0
    }

    /// Hidden or system entries are hidden.
    pub fn entry_flags(&self) -> EntryFlags {
        if self.attr & (ATTR_HIDDEN | ATTR_SYSTEM) != 0 {
            EntryFlags::HIDDEN
        } else {
            EntryFlags::EMPTY
        }
    }

    /// Creation time, in UTC when the entry gave an offset.
    pub fn created_time(&self) -> Option<Timestamp> {
        exfat_time(self.created, self.created_10ms, self.created_offset)
    }

    /// Modification time.
    pub fn modified_time(&self) -> Option<Timestamp> {
        exfat_time(self.modified, self.modified_10ms, self.modified_offset)
    }

    /// Last access time.
    pub fn accessed_time(&self) -> Option<Timestamp> {
        exfat_time(self.accessed, 0, self.accessed_offset)
    }
}

/// Converts a packed exFAT time. `None` when the date or time fields are
/// invalid. Without a valid offset the time is taken as UTC.
pub fn exfat_time(packed: u32, tens_of_ms: u8, offset: u8) -> Option<Timestamp> {
    let date = u16::try_from(packed >> 16).ok()?;
    let time = u16::try_from(packed & 0xFFFF).ok()?;
    let base = atlas_fat::dir::dos_time(date, time)?;
    let extra = i64::from(tens_of_ms.min(199)).saturating_mul(10_000_000);
    let shift = if offset & 0x80 != 0 {
        // Seven-bit two's complement, in quarter hours.
        let raw = i64::from(offset & 0x7F);
        let quarters = if raw >= 64 {
            raw.saturating_sub(128)
        } else {
            raw
        };
        quarters.saturating_mul(15 * 60 * 1_000_000_000)
    } else {
        0
    };
    let nanos = base.unix_nanos().checked_add(extra)?.checked_sub(shift)?;
    Some(Timestamp::from_unix_nanos(nanos))
}

/// One entry set found by a [`Walker`].
#[derive(Debug, Clone)]
pub struct Found {
    /// Volume byte offset of the primary: the node id.
    pub node: u64,
    /// Volume byte offset of the stream extension slot. It is `node + 32`
    /// unless the set crosses from one cluster of the directory to another.
    pub stream_at: u64,
    /// What the set says.
    pub meta: Meta,
}

/// Bytes of a directory read at once (a multiple of 32).
const BUF_LEN: usize = 4096;

/// Walks the entry sets of a directory, front to back.
///
/// Bounded by the directory's chain. The name of the set just returned is
/// in [`Self::name`].
pub struct Walker {
    chain: Arc<Chain>,
    pos: u64,
    buf: Vec<u8>,
    buf_start: u64,
    buf_len: usize,
    done: bool,
    /// The name of the set [`Self::next`] last returned.
    pub name: String,
}

impl Walker {
    /// A walker over the directory whose bytes are `chain`.
    pub fn new(chain: Arc<Chain>) -> Self {
        Self {
            chain,
            pos: 0,
            buf: vec![0; BUF_LEN],
            buf_start: 0,
            buf_len: 0,
            done: false,
            name: String::new(),
        }
    }

    /// The slot at `pos`; `Ok(None)` when the directory ends before it.
    fn slot(&mut self, pos: u64) -> Result<Option<[u8; SLOT]>> {
        let size = self.chain.len();
        let end = pos.saturating_add(SLOT as u64);
        if end > size {
            return Ok(None);
        }
        let buf_end = self
            .buf_start
            .saturating_add(u64::try_from(self.buf_len).unwrap_or(0));
        if self.buf_len == 0 || pos < self.buf_start || end > buf_end {
            let left = size.saturating_sub(pos);
            let want = usize::try_from(left).map_or(BUF_LEN, |l| l.min(BUF_LEN));
            let Some(dst) = self.buf.get_mut(..want) else {
                return Ok(None);
            };
            self.chain.read_at(pos, dst)?;
            self.buf_start = pos;
            self.buf_len = want;
        }
        let at = usize::try_from(pos.saturating_sub(self.buf_start)).unwrap_or(usize::MAX);
        let mut out = [0u8; SLOT];
        match self
            .buf
            .get(..self.buf_len)
            .and_then(|b| b.get(at..at.saturating_add(SLOT)))
        {
            Some(src) => out.copy_from_slice(src),
            None => return Ok(None),
        }
        Ok(Some(out))
    }

    /// The error that ended the chain early, when the walk got to its end.
    fn finish(&mut self) -> Result<Option<Found>> {
        self.done = true;
        match self.chain.broken() {
            Some(damage) => Err(damage.clone().into()),
            None => Ok(None),
        }
    }

    /// The next entry set, `Ok(None)` at the end. After an `Err` the walker
    /// is finished.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<Found>> {
        loop {
            if self.done {
                return Ok(None);
            }
            let at = self.pos;
            let slot = match self.slot(at) {
                Ok(Some(slot)) => slot,
                Ok(None) => return self.finish(),
                Err(e) => {
                    self.done = true;
                    return Err(e);
                }
            };
            self.pos = at.saturating_add(SLOT as u64);
            match slot.first().copied().unwrap_or(0) {
                0 => {
                    self.done = true;
                    return Ok(None);
                }
                TYPE_FILE => {}
                _ => continue,
            }
            let count = usize::from(slot.get(1).copied().unwrap_or(0));
            if !(2..=18).contains(&count) {
                continue;
            }
            // The secondary slots, read without moving `self.pos` until the
            // set is known to be good.
            let mut secondary = Vec::with_capacity(count);
            let mut cursor = self.pos;
            let mut ok = true;
            for _ in 0..count {
                match self.slot(cursor) {
                    Ok(Some(s)) => secondary.push((cursor, s)),
                    Ok(None) => {
                        ok = false;
                        break;
                    }
                    Err(e) => {
                        self.done = true;
                        return Err(e);
                    }
                }
                cursor = cursor.saturating_add(SLOT as u64);
            }
            if !ok {
                continue;
            }
            let Some((stream_pos, stream)) = secondary.first() else {
                continue;
            };
            let Some(meta) = Meta::parse(&slot, stream) else {
                continue;
            };
            let name_len = usize::from(stream.get(3).copied().unwrap_or(0));
            let name_entries = name_len.div_ceil(UNITS_PER_NAME_ENTRY);
            if name_len == 0 || name_entries.saturating_add(1) > count {
                continue;
            }
            let names = secondary.get(1..=name_entries).unwrap_or(&[]);
            if names.len() != name_entries
                || names.iter().any(|(_, s)| s.first() != Some(&TYPE_NAME))
            {
                continue;
            }
            let mut units: Vec<u16> = Vec::with_capacity(name_len);
            for (_, s) in names {
                for pair in s.get(2..).unwrap_or(&[]).as_chunks::<2>().0 {
                    if units.len() >= name_len {
                        break;
                    }
                    units.push(u16::from_le_bytes(*pair));
                }
            }
            self.name = String::from_utf16_lossy(&units);
            let Some((node, _)) = self.chain.locate(at) else {
                return self.finish();
            };
            let Some((stream_at, _)) = self.chain.locate(*stream_pos) else {
                return self.finish();
            };
            self.pos = cursor;
            return Ok(Some(Found {
                node,
                stream_at,
                meta,
            }));
        }
    }
}

/// The volume label in `data` (a directory's first bytes): the UTF-16 text
/// of the first `0x83` entry before the end marker.
pub fn find_label(data: &[u8]) -> Option<String> {
    for slot in data.as_chunks::<SLOT>().0 {
        match slot.first().copied().unwrap_or(0) {
            0 => return None,
            TYPE_LABEL => {
                let n = usize::from(slot.get(1).copied().unwrap_or(0)).min(11);
                let units: Vec<u16> = slot
                    .get(2..)
                    .unwrap_or(&[])
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .take(n)
                    .map(|p| u16::from_le_bytes(*p))
                    .collect();
                return Some(String::from_utf16_lossy(&units));
            }
            _ => {}
        }
    }
    None
}

/// The allocation bitmap entry in `data`: first cluster and length.
pub fn find_bitmap(data: &[u8]) -> Option<(u32, u64)> {
    for slot in data.as_chunks::<SLOT>().0 {
        match slot.first().copied().unwrap_or(0) {
            0 => return None,
            TYPE_BITMAP => return Some((le32(slot, 20), le64(slot, 24))),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(year: u32, month: u32, day: u32, h: u32, m: u32, s2: u32) -> u32 {
        (((year - 1980) << 9 | month << 5 | day) << 16) | (h << 11 | m << 5 | s2)
    }

    #[test]
    fn times_use_the_offset_and_the_increment() {
        let t = packed(2024, 3, 5, 13, 14, 8);
        let plain = exfat_time(t, 0, 0).unwrap().unix_nanos();
        assert_eq!(plain, 1_709_644_456 * 1_000_000_000);
        // +1:00 valid: the stored time is local, UTC is an hour earlier.
        let plus = exfat_time(t, 0, 0x80 | 4).unwrap().unix_nanos();
        assert_eq!(plain - plus, 3600 * 1_000_000_000);
        // -5:00 (seven-bit -20) is five hours later in UTC.
        let minus = exfat_time(t, 0, 0x80 | (128 - 20)).unwrap().unix_nanos();
        assert_eq!(minus - plain, 5 * 3600 * 1_000_000_000);
        // 10 ms increments add up to 1.99 s; larger values are clamped.
        let inc = exfat_time(t, 150, 0).unwrap().unix_nanos();
        assert_eq!(inc - plain, 1_500_000_000);
        assert_eq!(
            exfat_time(t, 255, 0).unwrap().unix_nanos() - plain,
            1_990_000_000
        );
    }

    #[test]
    fn invalid_dates_are_none() {
        assert!(exfat_time(0, 0, 0).is_none());
        assert!(exfat_time(packed(2024, 0, 5, 1, 1, 1), 0, 0).is_none());
        assert!(exfat_time(packed(2024, 3, 0, 1, 1, 1), 0, 0).is_none());
    }

    fn primary(count: u8, attr: u16) -> [u8; SLOT] {
        let mut b = [0u8; SLOT];
        b[0] = TYPE_FILE;
        b[1] = count;
        b[4..6].copy_from_slice(&attr.to_le_bytes());
        b
    }

    fn stream(name_len: u8, valid: u64, first: u32, len: u64, flags: u8) -> [u8; SLOT] {
        let mut b = [0u8; SLOT];
        b[0] = TYPE_STREAM;
        b[1] = flags;
        b[3] = name_len;
        b[8..16].copy_from_slice(&valid.to_le_bytes());
        b[20..24].copy_from_slice(&first.to_le_bytes());
        b[24..32].copy_from_slice(&len.to_le_bytes());
        b
    }

    #[test]
    fn a_set_parses_and_a_bad_one_does_not() {
        let m = Meta::parse(&primary(2, 0x12), &stream(1, 5, 7, 10, FLAG_NO_FAT_CHAIN)).unwrap();
        assert!(m.is_dir() && m.contiguous());
        assert_eq!((m.valid_len, m.first_cluster, m.data_len), (5, 7, 10));
        // A valid length past the data length is cut to it.
        let m = Meta::parse(&primary(2, 0), &stream(1, 99, 7, 10, 0)).unwrap();
        assert_eq!(m.valid_len, 10);
        for count in [0u8, 1, 19, 255] {
            assert!(Meta::parse(&primary(count, 0), &stream(1, 0, 0, 0, 0)).is_none());
        }
        let mut wrong = stream(1, 0, 0, 0, 0);
        wrong[0] = TYPE_NAME;
        assert!(Meta::parse(&primary(2, 0), &wrong).is_none());
        assert!(Meta::parse(&[], &[]).is_none());
    }

    #[test]
    fn label_and_bitmap_entries_are_found() {
        let mut data = vec![0u8; 5 * SLOT];
        data[SLOT] = TYPE_BITMAP;
        data[SLOT + 20..SLOT + 24].copy_from_slice(&5u32.to_le_bytes());
        data[SLOT + 24..SLOT + 32].copy_from_slice(&9u64.to_le_bytes());
        data[2 * SLOT] = TYPE_LABEL;
        data[2 * SLOT + 1] = 3;
        for (i, u) in "AbC".encode_utf16().enumerate() {
            data[2 * SLOT + 2 + 2 * i..2 * SLOT + 4 + 2 * i].copy_from_slice(&u.to_le_bytes());
        }
        // Nothing after an end marker counts.
        assert_eq!(find_label(&data), None);
        assert_eq!(find_bitmap(&data), None);
        data[0] = TYPE_FILE;
        assert_eq!(find_label(&data).as_deref(), Some("AbC"));
        assert_eq!(find_bitmap(&data), Some((5, 9)));
        // A label length past 11 is cut.
        data[2 * SLOT + 1] = 200;
        assert!(find_label(&data).unwrap().chars().count() <= 11);
        assert_eq!(find_label(&[]), None);
        assert_eq!(find_label(&data[..10]), None);
    }
}
