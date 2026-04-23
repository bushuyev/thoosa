use std::marker::PhantomData;
use crate::CopyToDevice;
use crate::traits::Data;

pub trait CopyMode {}

#[derive(Debug)]
pub struct CpuOnly<O>//TODO rename to MonoDevice - it's used also for structs having refs to  Tensors like CrossInput
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
pub struct CpuOrGpu<O>//TODO rename to PolyDevice - can be copied between devices
{
    pd:PhantomData<O>
}

impl<O> CopyMode for CpuOrGpu<O>
where
    O:Data<M=Self> + CopyToDevice
{}
