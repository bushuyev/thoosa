use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Expr, parse_macro_input};

/// CPU-only Data implementation, independent of Clone. Size accounting must be
/// explicit: #[cpu_data(shallow)] counts inline bytes; #[cpu_data(mem_size = expr)]
/// supplies an owned-size estimate in bytes (including inline storage).
#[proc_macro_derive(CpuData, attributes(cpu_data))]
pub fn derive_cpu_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut size = None;
    for attr in &input.attrs {
        if attr.path().is_ident("cpu_data") {
            attr.parse_nested_meta(|meta| {
                if size.is_some() {
                    return Err(meta.error("specify one memory-size policy"));
                }
                if meta.path.is_ident("shallow") {
                    size = Some(quote!(::core::mem::size_of_val(self) as u64));
                } else if meta.path.is_ident("mem_size") {
                    let expression: Expr = meta.value()?.parse()?;
                    size = Some(quote!(#expression));
                } else {
                    return Err(meta.error("expected shallow or mem_size = expression"));
                }
                Ok(())
            })?;
        }
    }
    let size = size.ok_or_else(|| syn::Error::new_spanned(&input.ident, "CpuData requires #[cpu_data(shallow)] or #[cpu_data(mem_size = expression)]"))?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::thoosa_shared::traits::Data for #ident #ty_generics #where_clause {
            type Metrics = ();
            fn size(&self) -> Self::Metrics { () }
            fn tho_device_idx(&self) -> usize { 0 }
            fn mem_size(&self) -> u64 { #size }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_size_policy_required_and_clone_not_required() {
        assert!(
            expand(&syn::parse_quote!(
                struct Data(Vec<u8>);
            ))
            .is_err()
        );
        let tokens = expand(&syn::parse_quote!(
            #[cpu_data(shallow)]
            struct Data<T>(T);
        ))
        .unwrap()
        .to_string();
        assert!(!tokens.contains("Clone"));
        assert!(
            expand(&syn::parse_quote!(
                #[cpu_data(mem_size = 8 + self.0.capacity() as u64)]
                struct Data(Vec<u8>);
            ))
            .is_ok()
        );
        assert!(
            expand(&syn::parse_quote!(
                #[cpu_data(shallow, shallow)]
                struct Data;
            ))
            .is_err()
        );
    }
}
