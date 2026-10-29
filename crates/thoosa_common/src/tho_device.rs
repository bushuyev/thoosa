use std::fmt::Debug;
use std::ops::Deref;
use tch::Device;
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ThoDevice(Device);

impl Deref for ThoDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Only CPU/CUDA devices with a representable application index enter this wrapper.
impl TryFrom<Device> for ThoDevice {
    type Error = thoosa_shared::ThoError;

    fn try_from(value: Device) -> Result<Self, Self::Error> {
        match value {
            Device::Cpu => Ok(Self(value)),
            Device::Cuda(ordinal) if ordinal < usize::MAX => Ok(Self(value)),
            _ => Err(thoosa_shared::ThoError::new(format!("unsupported device: {value:?}"))),
        }
    }
}

impl From<ThoDevice> for Device {
    fn from(value: ThoDevice) -> Self {
        value.0
    }
}

impl From<usize> for ThoDevice {
    fn from(value: usize) -> Self {
        match value {
            0 => ThoDevice(Device::Cpu),
            d => ThoDevice(Device::Cuda(d - 1)),
        }
    }
}


impl ThoDevice {
    pub fn tho_device_idx(&self) -> usize {
        match &self.0 {
            Device::Cpu => 0,
            Device::Cuda(v) => v + 1,
            _ => {
                unreachable!("device validated at construction")
            }
        }
    }

    pub fn gpu_ordinal(&self) -> Result<usize, thoosa_shared::ThoError> {
        match &self.0 {
            Device::Cpu => {
                Err(thoosa_shared::ThoError::new("CPU has no GPU ordinal"))
            },
            Device::Cuda(v) => Ok(*v),
            _ => {
                unreachable!("device validated at construction")
            }
        }
    }

}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_devices_and_index_boundaries() {
        assert_eq!(ThoDevice::try_from(Device::Cpu).unwrap().tho_device_idx(), 0);
        assert_eq!(ThoDevice::try_from(Device::Cuda(2)).unwrap().tho_device_idx(), 3);
        assert_eq!(Device::from(ThoDevice::from(usize::MAX)), Device::Cuda(usize::MAX - 1));
        assert!(ThoDevice::try_from(Device::Cuda(usize::MAX)).is_err());
        assert!(ThoDevice::try_from(Device::Mps).is_err());
        assert!(ThoDevice::try_from(Device::Vulkan).is_err());
        assert!(ThoDevice::from(0usize).gpu_ordinal().is_err());
    }
}
