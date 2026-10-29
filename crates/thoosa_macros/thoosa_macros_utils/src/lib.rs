use syn::{Field, GenericArgument, Lit, PathArguments, Type};

/// Recognize documented Rust scalar syntax. Aliases/newtypes use #[config_scalar]
/// on their field; the generated Into<Val>/TryFrom<&Val> calls check the contract.
pub fn is_primitive(ty: &Type) -> bool {
    let Type::Path(ty) = ty else { return false; };
    if ty.qself.is_some() { return false; }
    let parts: Vec<_> = ty.path.segments.iter().map(|segment| segment.ident.to_string()).collect();
    let Some(name) = parts.last() else { return false; };
    let prefix: Vec<_> = parts[..parts.len()-1].iter().map(String::as_str).collect();
    let scalar = matches!(name.as_str(), "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "isize" | "f32" | "f64" | "bool" | "char" | "str");
    (scalar && (prefix.is_empty() || prefix == ["std", "primitive"] || prefix == ["core", "primitive"]))
        || (name == "String" && (prefix.is_empty() || prefix == ["std", "string"] || prefix == ["alloc", "string"]))
}

pub fn is_option_primitive(ty: &Type) -> bool {
    get_option_inner_ty(ty).is_some_and(|inner| is_primitive(&inner))
}

fn container_inner(ty: &Type, name: &str, module: &str, alternate_root: &str) -> Option<Type> {
    let Type::Path(ty) = ty else { return None; };
    if ty.qself.is_some() { return None; }
    let parts: Vec<_> = ty.path.segments.iter().map(|segment| segment.ident.to_string()).collect();
    let prefix: Vec<_> = parts[..parts.len().checked_sub(1)?].iter().map(String::as_str).collect();
    let segment = ty.path.segments.last()?;
    if segment.ident != name || !(prefix.is_empty() || prefix == ["std", module] || prefix == [alternate_root, module]) { return None; }
    let PathArguments::AngleBracketed(args) = &segment.arguments else { return None; };
    if args.args.len() != 1 { return None; }
    match args.args.first()? { GenericArgument::Type(inner) => Some(inner.clone()), _ => None }
}

pub fn get_option_inner_ty(ty: &Type) -> Option<Type> {
    container_inner(ty, "Option", "option", "core")
}

pub fn get_vec_inner_ty(ty: &Type) -> Option<Type> {
    container_inner(ty, "Vec", "vec", "alloc")
}

pub fn comment(field: &&Field, field_name: &String) -> String {
    let mut descr = field_name.clone();
    for attr in &field.attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(meta) = &attr.meta {
                if let syn::Expr::Lit(lit_str) = &meta.value {
                    // let doc_comment = lit_str.lit.to_string();
                    if let Lit::Str(token) = &lit_str.lit {
                        descr = token.value();
                    }
                }
            }
        }
    }
    descr
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use std::fmt::{Debug, Display, Formatter};
    use std::hash::Hash;
    use thoosa_shared::RangeOrValue;
    use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
    use thoosa_shared::val::{NamedRange, NamedVal, Val};

    #[test]
    fn test_qualified_containers_and_scalar_syntax() {
        use super::*;
        for name in ["Option<u32>", "std::option::Option<u32>", "::core::option::Option<u32>"] {
            assert!(get_option_inner_ty(&syn::parse_str(name).unwrap()).is_some());
        }
        for name in ["Vec<u32>", "std::vec::Vec<u32>", "::alloc::vec::Vec<u32>"] {
            assert!(get_vec_inner_ty(&syn::parse_str(name).unwrap()).is_some());
        }
        assert!(is_primitive(&syn::parse_quote!(::core::primitive::u32)));
        assert!(is_primitive(&syn::parse_quote!(std::string::String)));
        assert!(!is_primitive(&syn::parse_quote!(RunId)));
        assert!(get_option_inner_ty(&syn::parse_quote!(other::Option<u32>)).is_none());
        assert!(get_vec_inner_ty(&syn::parse_quote!(Vec<u32, Allocator>)).is_none());
    }

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }
}

pub fn has_attr(field: &Field, name: &str) -> bool {
    field.attrs.iter().any(|attr| attr.path().is_ident(name))
}
