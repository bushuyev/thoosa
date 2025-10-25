use proc_macro::{TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{Data, DeriveInput, parse_macro_input};

#[proc_macro]
pub fn with_data(input: TokenStream) -> TokenStream {


    TokenStream::from(quote! {
        //lallaa

    })
}