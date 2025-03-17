use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(ThoosaPropertiesMap, attributes(in_map))]
pub fn thoosa_args_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let target_struct_ident = &ast.ident;

    eprintln!("thoosa_args_derive: target_struct_ident={target_struct_ident:?}");

    let gen = quote! {

    };

    gen.into()
}