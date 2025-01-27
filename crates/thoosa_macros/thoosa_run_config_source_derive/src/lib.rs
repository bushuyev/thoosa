use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_derive(RunConfigSource)]
pub fn run_config_source_derive(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).unwrap();

    let target_struct = &ast.ident;

    let gen = quote! {
        use log::debug;

        impl #target_struct {
            //
            pub fn to_run_configs(&self) {
                debug!("RunConfigSource::test_macro_fn");
            }
        }
    };
    gen.into()
}


#[cfg(test)]
mod tests {
    use super::*;

}
