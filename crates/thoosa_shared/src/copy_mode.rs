use std::marker::PhantomData;
use crate::CopyToDevice;
use crate::traits::{Data, WithId};

pub trait CopyMode {}

#[derive(Debug)]
pub struct CpuOnly<O>
{
    pd:PhantomData<O>
}

impl<O> CopyMode for CpuOnly<O>
where
    O:Data<M=Self>
{}

impl<O> CpuOnly<O>
where
    O:Data
{
    pub fn new() -> Self {
        Self { pd: PhantomData }
    }
}

#[derive(Debug)]
pub struct CpuOrGpu<O>
{
    pd:PhantomData<O>
}

impl<O> CopyMode for CpuOrGpu<O>
where
    O:Data<M=Self> + CopyToDevice
{}
