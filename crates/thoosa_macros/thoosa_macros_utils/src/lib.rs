use syn::{Type, Path};

pub fn is_primitive(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(ident) = type_path.path.get_ident() {
            match ident.to_string().as_str() {
                "i8" | "i16" | "i32" | "i64" | "i128" |
                "u8" | "u16" | "u32" | "u64" | "u128" |
                "f32" | "f64" | "bool" | "char" => {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}