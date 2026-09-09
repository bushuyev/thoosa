use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThoDeviceStats {
    pub allocated: i64,
    pub active: i64,
    pub reserved: i64,
    pub requested: i64,
    pub non_releasable: i64,
}

impl ThoDeviceStats {
    pub fn reusable_cached(self) -> i64 {
        self.reserved.saturating_sub(self.active)
    }

    pub fn pending_free(self) -> i64 {
        self.active.saturating_sub(self.allocated)
    }

    pub fn rounding_overhead(self) -> i64 {
        self.allocated.saturating_sub(self.requested)
    }

    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            allocated: self.allocated.saturating_add(other.allocated),
            active: self.active.saturating_add(other.active),
            reserved: self.reserved.saturating_add(other.reserved),
            requested: self.requested.saturating_add(other.requested),
            non_releasable: self.non_releasable.saturating_add(other.non_releasable),
        }
    }
}
