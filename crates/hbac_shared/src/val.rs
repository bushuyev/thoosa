use crate::HbacError;

//TODO move to shared
#[derive(Debug, PartialEq, Clone)]
pub enum Val {
    Real(f64),
    Int(i64),
    Str(String)
}

impl Val {

    pub fn as_usize(&self) -> Result<usize, HbacError> {
        match self {
            Self::Int(v)=> Ok(*v as usize),
            _=> Err(HbacError::new("can't be usize"))
        }
    }

    pub fn as_i64(&self) -> Result<i64, HbacError> {
        match self {
            Self::Int(v)=> Ok(*v as i64),
            _=> Err(HbacError::new("can't be i64"))
        }
    }

    pub fn as_str(&self) -> Result<String, HbacError> {
        match self {
            Self::Str(v)=> Ok(v.clone()),
            Self::Int(v)=> Ok(format!("{}", v)),
            Self::Real(v)=> Ok(format!("{}", v))
        }
    }
}

impl From<f32> for Val {
    fn from(v: f32) -> Self {
        Val::Real(v as f64)
    }
}

impl From<usize> for Val {
    fn from(v: usize) -> Self {
        Val::Int(v as i64)
    }
}

impl From<i64> for Val {
    fn from(v: i64) -> Self {
        Val::Int(v)
    }
}

impl From<i32> for Val {
    fn from(v: i32) -> Self {
        Val::Int(v as i64)
    }
}

impl From<f64> for Val {
    fn from(v: f64) -> Self {
        Val::Real(v)
    }
}


impl From<&str> for Val {
    fn from(v: &str) -> Self {
        Val::Str(v.to_string())
    }
}

impl From<String> for Val {
    fn from(v: String) -> Self {
        Val::Str(v)
    }
}

impl From<&f32> for Val {
    fn from(v: &f32) -> Self {
        Val::Real(*v as f64)
    }
}

impl From<&usize> for Val {
    fn from(v: &usize) -> Self {
        Val::Int(*v as i64)
    }
}

impl From<&i64> for Val {
    fn from(v: &i64) -> Self {
        Val::Int(*v)
    }
}

impl From<&i32> for Val {
    fn from(v: &i32) -> Self {
        Val::Int(*v as i64)
    }
}
