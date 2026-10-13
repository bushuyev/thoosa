use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};
// Legacy macro kept for compatibility in this workspace (unused here)
#[proc_macro_derive(ACachable)]
pub fn derive_acachable(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}

// Derive macro for thoosa_common::runner::common::cachable::Cached
#[proc_macro_derive(Cached, attributes(cache_name))]
pub fn derive_cached(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let DeriveInput { attrs: _attrs, ident: name, generics, .. } = input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // The runtime owns each cache and registers its purger atomically. A concrete
    // stage type, including every generic argument, determines cache identity.
    let expanded = quote! {
        #[async_trait::async_trait]
        impl #impl_generics thoosa_common::runner::common::cachable::Cached for #name #ty_generics #where_clause {
            async fn cache() -> Result<std::sync::Arc<thoosa_common::runner::common::cachable::StageCache<Self>>, thoosa_shared::ThoError>
                where
                    Self: Sized,
                    Self::Out: Sync + Send
            {
                use mockall_double::double;
                #[double]
                use thoosa_common::runner::global_ctx::GlobalCtx;

                GlobalCtx::global().cache_registry().cache_for::<Self>()
            }
        }
    };

    TokenStream::from(expanded)
}
