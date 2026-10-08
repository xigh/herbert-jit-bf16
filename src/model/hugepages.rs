//! The buffer the packed weights live in: 2 MiB pages when the `hugepages`
//! feature is on and Linux has pages reserved (`MAP_HUGETLB`), an `AVec` on
//! a cache line otherwise (`core/avec.rs`). Fewer TLB misses on a 9.7 GiB
//! working set that is read from end to end at every token; nothing else
//! changes.

use crate::core::avec::AVec;
use std::ops::{Deref, DerefMut};

/// Below this size a huge-page mapping is not worth it.
#[cfg(all(target_os = "linux", feature = "hugepages"))]
const HUGEPAGE_MIN_BYTES: usize = 2 * 1024 * 1024;
/// The page size of `MAP_HUGETLB`.
#[cfg(all(target_os = "linux", feature = "hugepages"))]
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// A fixed-size buffer, huge-page backed when possible.
pub struct HugeVec<T: Copy> {
    inner: Inner<T>,
}

enum Inner<T: Copy> {
    Vec(AVec<T>),
    #[cfg(all(target_os = "linux", feature = "hugepages"))]
    Mmap { ptr: *mut T, len: usize, mmap_len: usize },
}

// SAFETY: the mapping is owned by this value alone; the elements are `Copy`.
unsafe impl<T: Copy + Send> Send for HugeVec<T> {}
unsafe impl<T: Copy + Sync> Sync for HugeVec<T> {}

impl<T: Copy> HugeVec<T> {
    /// Moves `v` into huge pages when the feature, the size and the machine
    /// allow it, onto a cache line otherwise.
    pub fn from_vec(v: Vec<T>) -> Self {
        #[cfg(all(target_os = "linux", feature = "hugepages"))]
        {
            if v.len() * std::mem::size_of::<T>() >= HUGEPAGE_MIN_BYTES {
                if let Some(inner) = Self::try_mmap_copy(&v) {
                    return HugeVec { inner };
                }
            }
        }
        HugeVec { inner: Inner::Vec(AVec::from_vec(v)) }
    }

    #[cfg(all(target_os = "linux", feature = "hugepages"))]
    fn try_mmap_copy(src: &[T]) -> Option<Inner<T>> {
        let byte_len = src.len() * std::mem::size_of::<T>();
        let mmap_len = (byte_len + HUGE_PAGE_SIZE - 1) & !(HUGE_PAGE_SIZE - 1);
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mmap_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return None;
        }
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, ptr as *mut u8, byte_len) };
        Some(Inner::Mmap { ptr: ptr as *mut T, len: src.len(), mmap_len })
    }

    pub fn len(&self) -> usize {
        self.deref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Is this buffer on 2 MiB pages?
    pub fn is_hugepage_backed(&self) -> bool {
        #[cfg(all(target_os = "linux", feature = "hugepages"))]
        if matches!(self.inner, Inner::Mmap { .. }) {
            return true;
        }
        false
    }
}

impl<T: Copy> Deref for HugeVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.inner {
            Inner::Vec(v) => v,
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            Inner::Mmap { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

impl<T: Copy> DerefMut for HugeVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.inner {
            Inner::Vec(v) => v,
            #[cfg(all(target_os = "linux", feature = "hugepages"))]
            Inner::Mmap { ptr, len, .. } => unsafe { std::slice::from_raw_parts_mut(*ptr, *len) },
        }
    }
}

impl<T: Copy> Clone for HugeVec<T> {
    fn clone(&self) -> Self {
        HugeVec::from_vec(self.deref().to_vec())
    }
}

impl<T: Copy> Drop for HugeVec<T> {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", feature = "hugepages"))]
        if let Inner::Mmap { ptr, mmap_len, .. } = &self.inner {
            unsafe { libc::munmap(*ptr as *mut libc::c_void, *mmap_len) };
        }
    }
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for HugeVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HugeVec(len={}, huge={})", self.len(), self.is_hugepage_backed())
    }
}
