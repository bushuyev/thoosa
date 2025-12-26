use crate::copy_mode::{CpuOnly};
use crate::traits::{Data, Named, SimpleId, WithId};
use async_trait::async_trait;
use dashmap::{DashMap, DashSet, Entry};
use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use futures_util::future::join_all;
use log::debug;
use parking_lot::RwLock;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::marker::Tuple;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;
use std::time::Instant;

pub type CachableId = SimpleId;

pub trait Input: Data + Debug {

}

pub trait Output: Data + WithId + Debug {

}

impl <O> WithId for Arc<O>  where O:WithId + Debug {
    type ID = <O as WithId>::ID;

    fn id(&self) -> &Self::ID {
        self.as_ref().id()
    }
}
impl <O> Output for Arc<O> where O:Output + Debug {
}

impl Named for (){

}
impl WithId for () {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Output for () {}

impl WithId for CachableId {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl Data for CachableId {
    type Metrics = usize;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.0.len()
    }

    fn device(&self) -> usize {
        0
    }
}

impl Input for CachableId {}
