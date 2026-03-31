use proc_macro::TokenStream;
use quote::{format_ident, quote};
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

    let map_type = quote! {
        dashmap::DashMap<
            <<Self as thoosa_common::runner::common::runnable::Stage>::Out as thoosa_shared::traits::WithId>::ID,
            futures_util::future::Shared<
                futures_util::future::BoxFuture<'static, Result<
                    std::sync::Arc<thoosa_common::runner::common::cachable::DevicesWrap<
                        <Self as thoosa_common::runner::common::runnable::Stage>::Out
                    >>,
                    thoosa_shared::ThoosaError
                >>
            >,
        >
    };

    let cache_static_ident = format_ident!("__cached_map_ptr_for_{}", name);
    let register_static_ident = format_ident!("__cached_register_for_{}", name);

    // Generate impl that matches the signature used across the workspace.
    // It creates a OnceLock-held DashMap and registers it with GlobalCtx once.
    let expanded = quote! {
        #[allow(non_upper_case_globals)]
        static #cache_static_ident: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        #[allow(non_upper_case_globals)]
        static #register_static_ident: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();

        #[async_trait::async_trait]
        impl #impl_generics thoosa_common::runner::common::cachable::Cached for #name #ty_generics #where_clause {
            async fn cache() -> &'static #map_type
            where
                Self: Sized,
            {
                use mockall_double::double;
                #[double]
                use thoosa_common::runner::global_ctx::GlobalCtx;

                let ptr_ref = #cache_static_ident.get_or_init(|| {
                    let map: &'static #map_type = Box::leak(Box::new(Default::default()));
                    map as *const #map_type as usize
                });

                let map: &'static #map_type = unsafe {
                    &*((*ptr_ref) as *const #map_type)
                };


                #register_static_ident
                    .get_or_init(|| async move {
                        use thoosa_shared::traits::Named;

                        #[cfg(any(test, feature = "mock_global_ctx"))]
                        let _mock_ctx = thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder::install_global_ctx_mock();

                        GlobalCtx::global()
                            .register_cache((
                                Self::name(),
                                map as &'static dyn thoosa_common::runner::purge::FuturesRunnerMapPurger,
                            ))
                            .await;
                    })
                    .await;

                map
            }
        }
    };

    TokenStream::from(expanded)
}
