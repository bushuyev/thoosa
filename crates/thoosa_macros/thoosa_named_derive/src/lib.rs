use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Attribute, DeriveInput, Expr, FnArg, Ident, Lit, Meta, Path, Result as SynResult, Token, Type, parse_macro_input};

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

#[proc_macro_derive(Proxy, attributes(proxy))]
pub fn derive_proxy(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_proxy(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_proxy(input: DeriveInput) -> SynResult<proc_macro2::TokenStream> {
    let base_ident = input.ident;
    let attrs = ProxyAttrs::from_attrs(&input.attrs)?;
    let proxy_ident = attrs.proxy_name.ok_or_else(|| syn::Error::new_spanned(&base_ident, "missing `name = \"ProxyType\"` in `proxy` attribute"))?;
    let endpoint = attrs.endpoint.ok_or_else(|| syn::Error::new_spanned(&base_ident, "missing `endpoint = \"...\"` in `proxy` attribute"))?;
    let codec = attrs.codec.unwrap_or_else(|| syn::parse_quote!(crate::ipc::codec::JsonIpcCodec));
    let new_args = attrs.new_args.unwrap_or_default();
    let device = attrs.device.ok_or_else(|| syn::Error::new_spanned(&base_ident, "missing `device = \"...\"` in `proxy` attribute"))?;
    let fields = attrs.fields.unwrap_or_default();
    let out_id = attrs.out_id.ok_or_else(|| syn::Error::new_spanned(&base_ident, "missing `out_id = \"...\"` in `proxy` attribute"))?;
    let prepare = attrs.prepare.unwrap_or_default();
    let pre_run = attrs.pre_run.unwrap_or_default();
    let run_input = attrs.run_input;
    let endpoint_init = attrs.endpoint_init;
    let init_type = attrs.init_type;
    let in_type = attrs.in_type;
    let init = attrs.init;
    let init_stage = attrs.init_stage;
    let worker_body = attrs.worker_body;
    let wrapper_name = attrs.wrapper_name.unwrap_or_else(|| Ident::new(&format!("{base_ident}IpcWrapper"), base_ident.span()));
    let counter_method = attrs.counter_method;

    let field_defs = fields.iter().map(|field| {
        let name = &field.name;
        let ty = &field.ty;
        quote! { #name: #ty }
    });
    let field_inits = fields.iter().map(|field| {
        let name = &field.name;
        let init = &field.init;
        quote! { #name: #init }
    });

    let install = match init {
        Some(init) => quote! {
            crate::ipc::codec::install_ipc_endpoint::<#codec, _>(
                crate::runner::global_ctx::GlobalCtx::global().cuda_actor(#device).as_ref(),
                #endpoint,
                &#init,
            ).await?;
        },
        None => quote! {
            crate::ipc::codec::install_empty_ipc_endpoint(
                crate::runner::global_ctx::GlobalCtx::global().cuda_actor(#device).as_ref(),
                #endpoint,
            ).await?;
        },
    };

    let run_call = match run_input {
        Some(run_input) => quote! {
            crate::ipc::codec::invoke_ipc_endpoint::<#codec, _, <Self as crate::runner::common::runnable::Stage>::Out>(
                crate::runner::global_ctx::GlobalCtx::global().cuda_actor(crate::runner::common::RunCtx::tho_device_idx(ctx)).as_ref(),
                #endpoint,
                &#run_input,
            ).await
        },
        None => quote! {
            crate::ipc::codec::invoke_ipc_sync_stage::<#codec, _, <Self as crate::runner::common::runnable::Stage>::Out>(
                crate::runner::global_ctx::GlobalCtx::global().cuda_actor(crate::runner::common::RunCtx::tho_device_idx(ctx)).as_ref(),
                #endpoint,
                input,
                ctx,
            ).await
        },
    };

    let endpoint_init_tokens = match (endpoint_init, worker_body) {
        (Some(endpoint_init), Some(worker_body)) => quote! {
            #[derive(Debug)]
            struct #wrapper_name {
                stage: #base_ident,
            }

            impl crate::ipc::json_ser::SyncStageIpcWrapper for #wrapper_name {
                fn do_run(
                    &self,
                    input: &[u8],
                    all_cuda_tensors: &std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<u64, tch::Tensor>>>,
                ) -> Result<Vec<u8>, thoosa_shared::ThoError> {
                    type __ProxyIpcCodec = #codec;
                    #worker_body
                }
            }

            pub(crate) fn #endpoint_init(_init_msg: Vec<u8>) -> Result<Box<dyn crate::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError> {
                Ok(Box::new(#wrapper_name {
                    stage: #base_ident::new(),
                }))
            }
        },
        (Some(endpoint_init), None) => {
            let init_type = init_type.ok_or_else(|| syn::Error::new_spanned(&base_ident, "`endpoint_init` without `worker_body` requires `init_type = \"...\"`"))?;
            let init_stage = init_stage.ok_or_else(|| syn::Error::new_spanned(&base_ident, "`endpoint_init` without `worker_body` requires `init_stage = \"...\"`"))?;
            let in_type = in_type.unwrap_or_else(|| syn::parse_quote!(<#base_ident as crate::runner::common::runnable::Stage>::In<'static>));
            quote! {
                pub(crate) fn #endpoint_init(init_msg: Vec<u8>) -> Result<Box<dyn crate::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError> {
                    let init = <#codec as crate::ipc::codec::IpcCodec>::decode::<#init_type>(&init_msg)?;
                    Ok(crate::ipc::json_ser::make_ipc_sync_stage_wrapper::<
                        #codec,
                        #base_ident,
                        #in_type,
                        <#base_ident as crate::runner::common::runnable::Stage>::Out,
                    >(
                        #endpoint,
                        #init_stage,
                    ))
                }
            }
        },
        (None, Some(_)) => return Err(syn::Error::new_spanned(&base_ident, "`worker_body` requires `endpoint_init = \"...\"`")),
        (None, None) => quote! {},
    };

    let counter_method = match counter_method {
        Some(method) => quote! {
            pub(crate) fn #method(&self) -> usize {
                self.#method.load(std::sync::atomic::Ordering::Acquire)
            }
        },
        None => quote! {},
    };

    Ok(quote! {
        #[derive(Debug)]
        pub(crate) struct #proxy_ident {
            #(#field_defs,)*
        }

        impl #proxy_ident {
            pub async fn new(#new_args) -> Result<Self, thoosa_shared::ThoError> {
                #prepare
                #install

                Ok(Self {
                    #(#field_inits,)*
                })
            }

            #counter_method
        }

        impl thoosa_shared::traits::Adjustable for #proxy_ident {}

        impl crate::runner::common::OutId for #proxy_ident {
            fn out_id<'a>(
                &self,
                input_id: &<<Self as crate::runner::common::runnable::Stage>::In<'a> as thoosa_shared::traits::WithId>::ID,
            ) -> <<Self as crate::runner::common::runnable::Stage>::Out as thoosa_shared::traits::WithId>::ID {
                #out_id
            }
        }

        impl crate::runner::common::runnable::Stage for #proxy_ident {
            type Ctx = <#base_ident as crate::runner::common::runnable::Stage>::Ctx;
            type In<'a> = <#base_ident as crate::runner::common::runnable::Stage>::In<'a>;
            type Out = <#base_ident as crate::runner::common::runnable::Stage>::Out;
        }

        impl crate::runner::common::runnable::SyncStage for #proxy_ident {
            fn do_run<'a>(
                self: std::sync::Arc<Self>,
                input: &Self::In<'a>,
                ctx: &Self::Ctx,
            ) -> Result<Self::Out, thoosa_shared::ThoError>
            where
                Self: Sized,
            {
                #pre_run
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        #run_call
                    })
                })
            }
        }

        #endpoint_init_tokens
    })
}

#[derive(Default)]
struct ProxyAttrs {
    proxy_name: Option<Ident>,
    endpoint: Option<Expr>,
    codec: Option<Path>,
    new_args: Option<Punctuated<FnArg, Token![,]>>,
    prepare: Option<proc_macro2::TokenStream>,
    init: Option<Expr>,
    device: Option<Expr>,
    fields: Option<Punctuated<ProxyField, Token![,]>>,
    out_id: Option<Expr>,
    pre_run: Option<proc_macro2::TokenStream>,
    run_input: Option<Expr>,
    endpoint_init: Option<Ident>,
    init_type: Option<Type>,
    in_type: Option<Type>,
    init_stage: Option<Expr>,
    worker_body: Option<proc_macro2::TokenStream>,
    wrapper_name: Option<Ident>,
    counter_method: Option<Ident>,
}

impl ProxyAttrs {
    fn from_attrs(attrs: &[Attribute]) -> SynResult<Self> {
        let mut out = ProxyAttrs::default();
        for attr in attrs {
            if !attr.path().is_ident("proxy") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                let key = meta.path.require_ident()?.to_string();
                let value = meta.value()?;
                let lit = value.parse::<syn::LitStr>()?;
                match key.as_str() {
                    "name" => out.proxy_name = Some(lit.parse()?),
                    "endpoint" => out.endpoint = Some(lit.parse()?),
                    "codec" => out.codec = Some(lit.parse()?),
                    "new_args" => out.new_args = Some(parse_fn_args(&lit)?),
                    "prepare" => out.prepare = Some(parse_tokens(&lit)?),
                    "init" => out.init = Some(lit.parse()?),
                    "device" => out.device = Some(lit.parse()?),
                    "fields" => out.fields = Some(lit.parse_with(Punctuated::<ProxyField, Token![,]>::parse_terminated)?),
                    "out_id" => out.out_id = Some(lit.parse()?),
                    "pre_run" => out.pre_run = Some(parse_tokens(&lit)?),
                    "run_input" => out.run_input = Some(lit.parse()?),
                    "endpoint_init" => out.endpoint_init = Some(lit.parse()?),
                    "init_type" => out.init_type = Some(lit.parse()?),
                    "in_type" => out.in_type = Some(lit.parse()?),
                    "init_stage" => out.init_stage = Some(lit.parse()?),
                    "worker_body" => out.worker_body = Some(parse_tokens(&lit)?),
                    "wrapper" => out.wrapper_name = Some(lit.parse()?),
                    "counter" => out.counter_method = Some(lit.parse()?),
                    _ => return Err(meta.error(format!("unsupported proxy attribute `{key}`"))),
                }
                Ok(())
            })?;
        }
        Ok(out)
    }
}

struct ProxyField {
    name: Ident,
    ty: Type,
    init: Expr,
}

impl Parse for ProxyField {
    fn parse(input: ParseStream<'_>) -> SynResult<Self> {
        let name = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty = input.parse()?;
        input.parse::<Token![=]>()?;
        let init = input.parse()?;
        Ok(Self { name, ty, init })
    }
}

fn parse_fn_args(lit: &syn::LitStr) -> SynResult<Punctuated<FnArg, Token![,]>> {
    lit.parse_with(Punctuated::<FnArg, Token![,]>::parse_terminated)
}

fn parse_tokens(lit: &syn::LitStr) -> SynResult<proc_macro2::TokenStream> {
    lit.value().parse::<proc_macro2::TokenStream>().map_err(|e| syn::Error::new(lit.span(), e))
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
