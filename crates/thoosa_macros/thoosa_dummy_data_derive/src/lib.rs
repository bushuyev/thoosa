use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;
use syn::spanned::Spanned;
use syn::{parse_quote, Attribute, DeriveInput, Meta, Path, Token};
use syn::punctuated::Punctuated;

#[proc_macro_derive(DummyData, attributes(dummy_data))]
pub fn derive_dummy_data(input: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(input as DeriveInput);

    let mode = match extract_mode(&mut input) {
        Ok(mode) => mode,
        Err(err) => return TokenStream::from(err.to_compile_error()),
    };

    let ident = &input.ident;

    let mut generics = input.generics.clone();
    if matches!(mode, DummyDataMode::Clone) {
        generics
            .make_where_clause()
            .predicates
            .push(parse_quote!(Self: ::core::clone::Clone));
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let _copy_body = match mode {
        DummyDataMode::Clone => quote! { Ok(self.clone()) },
        DummyDataMode::NoClone => quote! { unimplemented!() },
    };

    let expanded = quote! {

        #[automatically_derived]
        impl #impl_generics ::thoosa_shared::traits::Data for #ident #ty_generics #where_clause {
            type Metrics = ();
            type M = thoosa_shared::copy_mode::CpuOnly<Self>;

            fn size(&self) -> Self::Metrics {
                ()
            }

            fn device(&self) -> usize {
                0
            }

            fn mem_size(&self) -> u64 {
                size_of::<Self>() as u64
            }
        }
    };

    TokenStream::from(expanded)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DummyDataMode {
    Clone,
    NoClone,
}

fn extract_mode(input: &mut DeriveInput) -> syn::Result<DummyDataMode> {
    let mut mode = DummyDataMode::Clone;
    let mut other_attrs = Vec::with_capacity(input.attrs.len());

    for attr in input.attrs.drain(..) {
        if attr.path().is_ident("dummy_data") {
            mode = parse_dummy_data_attr(attr)?;
        } else {
            other_attrs.push(attr);
        }
    }

    input.attrs = other_attrs;
    Ok(mode)
}

fn parse_dummy_data_attr(attr: Attribute) -> syn::Result<DummyDataMode> {
    match attr.meta {
        Meta::Path(_) => Ok(DummyDataMode::Clone),
        Meta::List(list) => {
            let mut mode = DummyDataMode::Clone;
            let paths = list.parse_args_with(Punctuated::<Path, Token![,]>::parse_terminated)?;
            for path in paths {
                if path.is_ident("clone") {
                    mode = DummyDataMode::Clone;
                } else if path.is_ident("no_clone") {
                    mode = DummyDataMode::NoClone;
                } else {
                    return Err(syn::Error::new(path.span(), "expected `clone` or `no_clone`"));
                }
            }
            Ok(mode)
        }
        Meta::NameValue(meta) => Err(syn::Error::new(meta.path.span(), "unexpected `dummy_data` attribute")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn dummy_data_attr_defaults_to_clone() {
        let mut input: DeriveInput = parse_quote! {
            #[dummy_data]
            struct Foo;
        };

        let mode = extract_mode(&mut input).unwrap();
        assert_eq!(mode, DummyDataMode::Clone);
    }

    #[test]
    fn dummy_data_attr_explicit_clone() {
        let mut input: DeriveInput = parse_quote! {
            #[dummy_data(clone)]
            struct Foo;
        };

        let mode = extract_mode(&mut input).unwrap();
        assert_eq!(mode, DummyDataMode::Clone);
    }

    #[test]
    fn dummy_data_attr_no_clone() {
        let mut input: DeriveInput = parse_quote! {
            #[dummy_data(no_clone)]
            struct Foo;
        };

        let mode = extract_mode(&mut input).unwrap();
        assert_eq!(mode, DummyDataMode::NoClone);
    }
}
