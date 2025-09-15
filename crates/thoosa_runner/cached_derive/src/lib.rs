use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Meta, Lit, Expr};
// Legacy macro kept for compatibility in this workspace (unused here)
#[proc_macro_derive(ACachable)]
pub fn derive_acachable(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}

// Derive macro for caching_runner::common::cachable::Cached
#[proc_macro_derive(Cached, attributes(cache_name))]
pub fn derive_cached(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // Optional #[cache_name = "..."] attribute
    let mut reg_name = None::<syn::LitStr>;
    for attr in input.attrs.iter() {
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
        quote! { stringify!(<<#name as caching_runner::common::runnable::Stage>::Out as thoosa_shared::traits::WithId>::ID) }
    };

    // Generate impl that matches the signature used across the workspace.
    // It creates a OnceLock-held DashMap and registers it with GlobalCtx once.
    let expanded = quote! {
        #[async_trait::async_trait]
        impl caching_runner::common::cachable::Cached for #name {
            async fn cache() -> &'static dashmap::DashMap<
                <<Self as caching_runner::common::runnable::Stage>::Out as thoosa_shared::traits::WithId>::ID,
                futures_util::future::Shared<futures_util::future::BoxFuture<'static, Result<std::sync::Arc<caching_runner::common::cachable::DataCacheWrapper<<Self as caching_runner::common::runnable::Stage>::Out>>, thoosa_shared::ThoosaError>>>,
            >
            where
                Self: Sized,
            {
                // Static cache map for this Runnable's output type
                static CACHE: std::sync::OnceLock<dashmap::DashMap<
                    <<#name as caching_runner::common::runnable::Stage>::Out as thoosa_shared::traits::WithId>::ID,
                    futures_util::future::Shared<futures_util::future::BoxFuture<'static, Result<std::sync::Arc<caching_runner::common::cachable::DataCacheWrapper<<#name as caching_runner::common::runnable::Stage>::Out>>, thoosa_shared::ThoosaError>>>,
                >> = std::sync::OnceLock::new();

                let map = CACHE.get_or_init(|| dashmap::DashMap::new());

                // Register once with GlobalCtx for purge/dump operations.
                // Use a stable, type-based name to tag this cache.
                static REGISTERED: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
                REGISTERED
                    .get_or_init(|| async {
                        caching_runner::global_ctx::GlobalCtx::global()
                            .register_cache((
                                // Prefer a descriptive, static tag; allow override via #[cache_name = "..."]
                                #reg_name_tokens,
                                CACHE.get().expect("maps"),
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
