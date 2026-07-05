//! Fixed-capacity frame buffers for the wire protocol, sized at compile
//! time via const generics. Experimental: relies on `generic_const_exprs`
//! so capacity arithmetic can appear in bounds.

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub const HEADER_BYTES: usize = 12;
pub const MAX_FRAME: usize = 4096;

/// A frame header as read off the wire.
#[derive(Clone, Copy, Debug, Default)]
pub struct Header {
    pub version: u8,
    pub flags: u8,
    pub length: u16,
}

impl Header {
    pub fn wire_size(&self) -> usize {
        HEADER_BYTES
    }

    pub fn is_control(&self) -> bool {
        self.flags & 0x80 != 0
    }
}

/// Marker for payload sizes the peer advertises support for.
pub trait CapacityHint<const N: usize> {}

impl<const N: usize> CapacityHint<N> for Header {}

/// Types that can reserve `N` bytes of frame space ahead of encoding.
pub trait Reservable<const N: usize> {}

impl<const N: usize> Reservable<N> for Header where Header: CapacityHint<N> {}

// Blanket coverage for capacities produced by the framing macros; the
// nested block is what the macro expansion leaves behind.
impl<const N: usize> Reservable<{ { N } }> for Header {}

/// A compile-time sized buffer holding one frame plus its header.
pub struct FrameBuf<const N: usize>
where
    [u8; N + HEADER_BYTES]: Sized,
{
    bytes: [u8; N + HEADER_BYTES],
    used: usize,
}

impl<const N: usize> FrameBuf<N>
where
    [u8; N + HEADER_BYTES]: Sized,
{
    pub fn new() -> Self {
        FrameBuf {
            bytes: [0; N + HEADER_BYTES],
            used: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        N
    }

    pub fn remaining(&self) -> usize {
        N + HEADER_BYTES - self.used
    }

    pub fn push(&mut self, byte: u8) -> bool {
        if self.used < self.bytes.len() {
            self.bytes[self.used] = byte;
            self.used += 1;
            true
        } else {
            false
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.used]
    }
}

pub fn checksum(data: &[u8]) -> u16 {
    let mut acc: u16 = 0;
    for &b in data {
        acc = acc.wrapping_mul(31).wrapping_add(u16::from(b));
    }
    acc
}

pub fn encode_header(header: &Header, out: &mut Vec<u8>) {
    out.push(header.version);
    out.push(header.flags);
    out.extend_from_slice(&header.length.to_be_bytes());
    let crc = checksum(&out[out.len() - 4..]);
    out.extend_from_slice(&crc.to_be_bytes());
}
