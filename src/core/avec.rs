//! The buffers of the engine, each on a cache line.
//!
//! A `Vec<T>` is placed by the allocator: on glibc a block of 128 KiB or more
//! comes from `mmap` behind a 16-byte header, so the packed weights, the KV
//! cache and the prefill activations would all start at 16 mod 64, and every
//! 64-byte load of a kernel would straddle two cache lines. `AVec` is the
//! buffer of every one of them: a fixed length, zeroed at creation, its first
//! element on a 64-byte boundary whatever the allocator.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// The alignment of every buffer: a cache line.
pub const ALIGN: usize = 64;

/// A fixed-length buffer of `T` on a cache line.
pub struct AVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
}

// SAFETY: the block is owned by this value alone; `T` is `Copy` data.
unsafe impl<T: Copy + Send> Send for AVec<T> {}
unsafe impl<T: Copy + Sync> Sync for AVec<T> {}

impl<T: Copy> AVec<T> {
    /// The layout of `len` elements, `None` when that is no bytes at all.
    fn layout(len: usize) -> Option<Layout> {
        let bytes = len.checked_mul(std::mem::size_of::<T>()).expect("buffer size overflows");
        (bytes > 0).then(|| Layout::from_size_align(bytes, ALIGN.max(std::mem::align_of::<T>())).expect("buffer layout"))
    }

    /// `len` zeroed elements (every `T` used here is valid as zero bytes).
    pub fn zeroed(len: usize) -> Self {
        let Some(layout) = Self::layout(len) else {
            return AVec { ptr: NonNull::dangling(), len };
        };
        // SAFETY: a non-zero size; zero bytes are a valid u16, f32 or u8.
        let block = unsafe { alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(block).unwrap_or_else(|| handle_alloc_error(layout));
        AVec { ptr, len }
    }

    /// A copy of `v` on a cache line.
    pub fn from_vec(v: Vec<T>) -> Self {
        let mut a = Self::zeroed(v.len());
        a.copy_from_slice(&v);
        a
    }

    /// An empty buffer.
    pub fn new() -> Self {
        AVec { ptr: NonNull::dangling(), len: 0 }
    }
}

impl<T: Copy> Default for AVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> Deref for AVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `len` elements allocated and zeroed at `ptr` (or none), owned here.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for AVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as above, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Drop for AVec<T> {
    fn drop(&mut self) {
        if let Some(layout) = Self::layout(self.len) {
            // SAFETY: allocated by `alloc_zeroed` with this layout.
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
        }
    }
}

impl<T: Copy> Clone for AVec<T> {
    fn clone(&self) -> Self {
        let mut c = Self::zeroed(self.len);
        c.copy_from_slice(self);
        c
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for AVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Copy + PartialEq> PartialEq for AVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}
