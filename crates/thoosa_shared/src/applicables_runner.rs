use crate::traits::{Data, Named, SimpleId, WithId};
use std::fmt::Debug;
use std::sync::Arc;


pub trait Input: Debug {

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

impl WithId for SimpleId {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}


