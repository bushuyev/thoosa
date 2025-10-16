use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{DeriveInput, Expr, Lit, Meta, parse_macro_input};
// Legacy macro kept for compatibility in this workspace (unused here)
#[proc_macro_derive(ACachable)]
pub fn derive_acachable(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}

// Derive macro for caching_runner::common::cachable::Cached
#[proc_macro_derive(Cached, attributes(cache_name))]
pub fn derive_cached(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let DeriveInput { attrs, ident: name, generics, .. } = input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Optional #[cache_name = "..."] attribute
    let mut reg_name = None::<syn::LitStr>;
    for attr in attrs.iter() {
        if attr.path().is_ident("cache_name") {
            if let Meta::NameValue(nv) = attr.meta.clone() {
                if let Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(lit) = &expr_lit.lit {
                        reg_name = Some(lit.clone());
                    }
                }
            }
        }
    }

    let reg_name_tokens = if let Some(lit) = reg_name {
        quote! { #lit }
    } else {
        quote! {
            stringify!(<<#name #ty_generics as caching_runner::common::runnable::Stage>::Out as thoosa_shared::traits::WithId>::ID)
        }
    };

    let map_type = quote! {
        dashmap::DashMap<
            <<Self as caching_runner::common::runnable::Stage>::Out as thoosa_shared::traits::WithId>::ID,
            futures_util::future::Shared<
                futures_util::future::BoxFuture<'static, Result<
                    std::sync::Arc<caching_runner::common::cachable::DataCacheWrapper<
                        <Self as caching_runner::common::runnable::Stage>::Out
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
        impl #impl_generics caching_runner::common::cachable::Cached for #name #ty_generics #where_clause {
            async fn cache() -> &'static #map_type
            where
                Self: Sized,
            {
                let ptr_ref = #cache_static_ident.get_or_init(|| {
                    let map: &'static #map_type = Box::leak(Box::new(Default::default()));
                    map as *const #map_type as usize
                });

                let map: &'static #map_type = unsafe {
                    &*((*ptr_ref) as *const #map_type)
                };

                let cache_name: &'static str = #reg_name_tokens;

                #register_static_ident
                    .get_or_init(|| async move {
                        caching_runner::global_ctx::GlobalCtx::global()
                            .register_cache((
                                cache_name,
                                map as &'static dyn caching_runner::purge::FuturesRunnerMapPurger,
                            ))
                            .await;
                    })
                    .await;

                map
            }

            fn get_name()->&'static str {
                #reg_name_tokens
            }

        }
    };

    TokenStream::from(expanded)
}
