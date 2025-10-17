use proc_macro::TokenStream;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{parse_macro_input, DeriveInput, Path, Token};

#[proc_macro_derive(DummyData)]
pub fn derive_dummy_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let is_clone = has_clone_derive(&input);

    let ident = &input.ident;
    let generics = input.generics.clone();
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let copy_body = if is_clone {
        quote! { self.clone() }
    } else {
        quote! { unimplemented!() }
    };

    let expanded = quote! {
        #[automatically_derived]
        #[async_trait::async_trait]
        impl #impl_generics ::thoosa_shared::CopyToDevice for #ident #ty_generics #where_clause {
            async fn copy_to_device(&self, _dst_device: usize) -> Self {
                #copy_body
            }
        }

        #[automatically_derived]
        impl #impl_generics ::thoosa_shared::traits::Data for #ident #ty_generics #where_clause {
            type Metrics = ();

            fn size(&self) -> Self::Metrics {
                ()
            }

            fn device(&self) -> usize {
                0
            }
        }
    };

    TokenStream::from(expanded)
}

fn has_clone_derive(input: &DeriveInput) -> bool {
    input.attrs.iter().filter(|attr| attr.path().is_ident("derive")).any(|attr| {
        attr.parse_args_with(Punctuated::<Path, Token![,]>::parse_terminated)
            .map(|paths| {
                paths.iter().any(|path| {
                    path.is_ident("Clone")
                        || path.segments
                            .last()
                            .map(|segment| segment.ident == "Clone")
                            .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    })
}
