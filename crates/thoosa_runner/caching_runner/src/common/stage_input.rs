use std::any::Any;
use std::sync::Arc;
use thoosa_shared::traits::WithId;

pub struct StageInput<T>
where
    T: WithId + 'static,
{
    id: T::ID,
    kind: StageInputKind<T>,
}

enum StageInputKind<T>
where
    T: WithId + 'static,
{
    Plain(Arc<T>),
    Cached(Arc<dyn Any + Send + Sync>),
}

impl<T> Clone for StageInput<T>
where
    T: WithId + 'static,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            kind: self.kind.clone(),
        }
    }
}

impl<T> Clone for StageInputKind<T>
where
    T: WithId + 'static,
{
    fn clone(&self) -> Self {
        match self {
            StageInputKind::Plain(raw) => StageInputKind::Plain(Arc::clone(raw)),
            StageInputKind::Cached(erased) => StageInputKind::Cached(Arc::clone(erased)),
        }
    }
}

impl<T> StageInput<T>
where
    T: WithId + 'static,
{
    pub fn from_plain(raw: Arc<T>) -> Self {
        let id = raw.id().clone();
        Self {
            id,
            kind: StageInputKind::Plain(raw),
        }
    }

    pub fn from_cached_arc<C>(raw: Arc<C>) -> Self
    where
        C: Any + Send + Sync + WithId<ID = T::ID> + 'static,
    {
        let id = raw.id().clone();
        let erased: Arc<dyn Any + Send + Sync> = raw;
        Self {
            id,
            kind: StageInputKind::Cached(erased),
        }
    }

    pub fn id(&self) -> &T::ID {
        &self.id
    }

    pub fn clone_plain(&self) -> Option<Arc<T>> {
        match &self.kind {
            StageInputKind::Plain(raw) => Some(Arc::clone(raw)),
            StageInputKind::Cached(_) => None,
        }
    }

    pub fn clone_cached<C>(&self) -> Option<Arc<C>>
    where
        C: Any + Send + Sync + 'static,
    {
        match &self.kind {
            StageInputKind::Cached(erased) => Arc::clone(erased).downcast::<C>().ok(),
            StageInputKind::Plain(_) => None,
        }
    }
}