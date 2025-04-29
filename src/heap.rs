use core::marker::PhantomData;

use alloc::collections::BinaryHeap;
use alloc::vec;
use alloc::vec::Vec;

use num_traits::Zero;
use ordered_float::NotNan;

use crate::internal_neighbour::InternalNeighbour;
use crate::{KDTree, Point, Scalar};

/// Global trait for tracking candidates during KDTree traversal
pub trait CandidateCollector<T: Scalar> {
    fn add(&mut self, dist2: NotNan<T>, index: u32);
    fn furthest_dist2(&self) -> NotNan<T>;
}

pub struct UnboundedCollector<T: Scalar, P: Point<T>>(Vec<u32>, NotNan<T>, PhantomData<P>);

impl<T: Scalar, P: Point<T>> CandidateCollector<T> for UnboundedCollector<T, P> {
    fn add(&mut self, dist2: NotNan<T>, index: u32) {
        self.0.push(index);
        self.1 = self.1.max(dist2);
    }

    fn furthest_dist2(&self) -> NotNan<T> {
        self.1
    }
}

impl<T: Scalar, P: Point<T>> UnboundedCollector<T, P> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity), NotNan::zero(), PhantomData)
    }

    pub fn clear(&mut self) {
        self.0.clear();
        self.1 = NotNan::zero();
    }
}

impl<T: Scalar, P: Point<T>> UnboundedCollector<T, P> {
    pub fn externalise<'a, 'b>(
        &'a self,
        tree: &'b KDTree<T, P>,
    ) -> impl ExactSizeIterator<Item = u32> + Clone + 'b
    where
        'a: 'b,
    {
        self.0.iter().map(|inner| tree.externalise_index(*inner))
    }
}

pub struct BoundedCollector<T: Scalar, P: Point<T>>(
    BinaryHeap<InternalNeighbour<T>>,
    PhantomData<P>,
);

impl<T: Scalar, P: Point<T>> CandidateCollector<T> for BoundedCollector<T, P> {
    fn add(&mut self, dist2: NotNan<T>, index: u32) {
        self.0.add(dist2, index);
    }

    fn furthest_dist2(&self) -> NotNan<T> {
        self.0.furthest_dist2()
    }
}

impl<T: Scalar, P: Point<T>> BoundedCollector<T, P> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(BinaryHeap::with_capacity(capacity as usize), PhantomData)
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T: Scalar, P: Point<T>> BoundedCollector<T, P> {
    pub fn externalise<'a, 'b>(
        &'a self,
        tree: &'b KDTree<T, P>,
    ) -> impl ExactSizeIterator<Item = u32> + Clone + 'b
    where
        'a: 'b,
    {
        self.0
            .iter()
            .map(|inner| tree.externalise_index(inner.index))
    }
}

/// Local trait for keeping candidates in a heap-like behaviour
pub(crate) trait CandidateHeap<T: Scalar> {
    fn new_with_k(k: u32) -> Self;
    fn into_vec(self) -> Vec<InternalNeighbour<T>>;
    fn into_sorted_vec(self) -> Vec<InternalNeighbour<T>>;
}

impl<T: Scalar> CandidateCollector<T> for BinaryHeap<InternalNeighbour<T>> {
    fn add(&mut self, dist2: NotNan<T>, index: u32) {
        let k = self.capacity();
        if self.len() < k {
            self.push(InternalNeighbour { index, dist2 });
        } else {
            let mut max_heap_value = self.peek_mut().unwrap();
            if dist2 < max_heap_value.dist2 {
                *max_heap_value = InternalNeighbour { index, dist2 };
            }
        }
    }
    fn furthest_dist2(&self) -> NotNan<T> {
        if self.len() < self.capacity() {
            NotNan::new(T::infinity()).unwrap()
        } else {
            self.peek()
                .map_or(NotNan::new(T::infinity()).unwrap(), |n| n.dist2)
        }
    }
}

impl<T: Scalar> CandidateHeap<T> for BinaryHeap<InternalNeighbour<T>> {
    fn new_with_k(k: u32) -> Self {
        BinaryHeap::with_capacity(k as usize)
    }
    fn into_vec(self) -> Vec<InternalNeighbour<T>> {
        BinaryHeap::into_vec(self)
    }
    fn into_sorted_vec(self) -> Vec<InternalNeighbour<T>> {
        BinaryHeap::into_sorted_vec(self)
    }
}

fn keep_finite_elements<T: Scalar>(v: Vec<InternalNeighbour<T>>) -> Vec<InternalNeighbour<T>> {
    let pos = v.iter().position(|n| n.dist2.into_inner() == T::infinity());
    match pos {
        None => v,
        Some(pos) => {
            let mut v = v;
            v.truncate(pos);
            v
        }
    }
}

impl<T: Scalar> CandidateCollector<T> for Vec<InternalNeighbour<T>> {
    fn add(&mut self, dist2: NotNan<T>, index: u32) {
        if dist2 > self.furthest_dist2() {
            return;
        }
        let mut i = self.len() - 1;
        while i > 0 {
            if self[i - 1].dist2 > dist2 {
                self[i] = self[i - 1];
            } else {
                break;
            }
            i -= 1;
        }
        self[i].dist2 = dist2;
        self[i].index = index;
    }
    fn furthest_dist2(&self) -> NotNan<T> {
        self[self.len() - 1].dist2
    }
}

impl<T: Scalar> CandidateHeap<T> for Vec<InternalNeighbour<T>> {
    fn new_with_k(k: u32) -> Self {
        vec![
            InternalNeighbour {
                index: 0,
                dist2: NotNan::new(T::infinity()).unwrap()
            };
            k as usize
        ]
    }
    fn into_vec(self) -> Vec<InternalNeighbour<T>> {
        keep_finite_elements(self)
    }
    fn into_sorted_vec(self) -> Vec<InternalNeighbour<T>> {
        keep_finite_elements(self)
    }
}

impl<T: Scalar> CandidateCollector<T> for InternalNeighbour<T> {
    fn add(&mut self, dist2: NotNan<T>, index: u32) {
        if dist2 < self.dist2 {
            self.dist2 = dist2;
            self.index = index;
        }
    }
    fn furthest_dist2(&self) -> NotNan<T> {
        self.dist2
    }
}
impl<T: Scalar> CandidateHeap<T> for InternalNeighbour<T> {
    fn new_with_k(k: u32) -> Self {
        debug_assert_eq!(k, 1);
        InternalNeighbour {
            index: 0,
            dist2: NotNan::new(T::infinity()).unwrap(),
        }
    }
    fn into_vec(self) -> Vec<InternalNeighbour<T>> {
        vec![self]
    }
    fn into_sorted_vec(self) -> Vec<InternalNeighbour<T>> {
        vec![self]
    }
}

#[cfg(test)]
mod tests {
    use crate::infinite::HasInfinite;
    use crate::*;
    #[test]
    fn keep_finite_elements() {
        let v = vec![
            InternalNeighbour {
                index: 0,
                dist2: NotNan::<f32>::zero(),
            },
            InternalNeighbour {
                index: 0,
                dist2: NotNan::<f32>::infinite(),
            },
            InternalNeighbour {
                index: 0,
                dist2: NotNan::<f32>::infinite(),
            },
        ];
        let finite_v = crate::heap::keep_finite_elements(v);
        assert_eq!(
            finite_v,
            vec![InternalNeighbour {
                index: 0,
                dist2: NotNan::<f32>::zero(),
            }]
        );
    }
}
