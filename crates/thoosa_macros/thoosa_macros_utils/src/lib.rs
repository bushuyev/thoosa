use syn::__private::ToTokens;
use syn::{Field, GenericArgument, Ident, Lit, PathArguments, Type, TypePath};

pub fn is_primitive(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(ident) = type_path.path.get_ident() {
            return  _is_primitive(ident);
        }
    }
    false
}

fn _is_primitive(ident: &Ident) -> bool {
    match ident.to_string().as_str() {
        "i8" | "i16" | "i32" | "i64" | "i128" |
        "u8" | "u16" | "u32" | "u64" | "u128" |
        "f32" | "f64" | "bool" | "char" => {
             true
        }
        _ => {
            false
        }
    }
}

pub fn is_option_primitive(ty: &Type) -> bool {
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(first_segment) = path.segments.first() {
            if first_segment.ident == "Option" {
                if let PathArguments::AngleBracketed(ref args) = first_segment.arguments {
                    if let Some(GenericArgument::Type(Type::Path(inner_type_path))) = args.args.first() {
                        if let Some(inner_ident) = inner_type_path.path.get_ident() {
                            return _is_primitive(inner_ident);
                        }
                    }
                }
            }
        }
    }
    false
}

pub fn get_option_inner_ty(ty: &Type) -> Option<Type> {
    if let Type::Path(TypePath { path, .. }) = ty {
        if let Some(first_segment) = path.segments.first() {
            if first_segment.ident == "Option" {
                if let PathArguments::AngleBracketed(ref args) = first_segment.arguments {
                    if let Some(GenericArgument::Type(inner_type)) = args.args.first() {
                       return Some(inner_type.clone())
                    }
                }
            }
        }
    }
    None
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
    use std::fmt::{Debug, Display, Formatter};
    use std::hash::Hash;
    use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_shared::RangeOrValue;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

}