use proc_macro::TokenStream;
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Expr, Ident, Lit, Meta, parse_macro_input};

#[proc_macro_derive(Named, attributes(name))]
pub fn derive_named(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = input.ident.clone();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let mut name_literal: Option<syn::LitStr> = None;
    let mut error: Option<syn::Error> = None;

    for attr in &input.attrs {
        if attr.path().is_ident("name") {
            match parse_name_literal(attr, "name") {
                Ok(lit) => {
                    if name_literal.replace(lit).is_some() {
                        push_error(&mut error, syn::Error::new_spanned(attr, "duplicate `name` attribute"));
                    }
                }
                Err(err) => push_error(&mut error, err),
            }
        }

    }

    if let Some(err) = error {
        return TokenStream::from(err.to_compile_error());
    }

    let name_expr = if let Some(lit) = name_literal {
        quote! { #lit }
    } else {
        quote! { stringify!(#ident) }
    };

    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics ::thoosa_shared::traits::Named for #ident #ty_generics #where_clause {
            fn name() -> &'static str {
                #name_expr
            }
        }
    };

    TokenStream::from(expanded)
}

fn parse_name_literal(attr: &Attribute, label: &str) -> syn::Result<syn::LitStr> {
    match attr.meta.clone() {
        Meta::NameValue(nv) => match nv.value {
            Expr::Lit(expr_lit) => match expr_lit.lit {
                Lit::Str(lit) => Ok(lit),
                other => Err(syn::Error::new_spanned(other, format!("`{label}` must be a string literal"))),
            },
            other => Err(syn::Error::new_spanned(other, format!("`{label}` must be a string literal"))),
        },
        meta => Err(syn::Error::new_spanned(meta, format!("expected `{label} = \"...\"`"))),
    }
}

fn push_error(target: &mut Option<syn::Error>, err: syn::Error) {
    if let Some(existing) = target {
        existing.combine(err);
    } else {
        *target = Some(err);
    }
}
