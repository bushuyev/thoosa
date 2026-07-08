use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Attribute, DeriveInput, Expr, FnArg, Ident, Pat, Path, Result as SynResult, Token, Type, parse_macro_input};

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
    let common = attrs.common.unwrap_or_else(|| syn::parse_quote!(crate));
    let base_ty = attrs.base_type.unwrap_or_else(|| syn::parse_quote!(#base_ident));
    let endpoint = attrs.endpoint.ok_or_else(|| syn::Error::new_spanned(&base_ident, "missing `endpoint = \"...\"` in `proxy` attribute"))?;
    let codec = attrs.codec.map(|codec| quote!(#codec)).unwrap_or_else(|| quote!(#common::ipc::codec::JsonIpcCodec));
    let new_args = attrs.new_args.unwrap_or_default();
    let device_param = proxy_device_param(&new_args).ok_or_else(|| syn::Error::new_spanned(&base_ident, "proxy constructor arguments must include one of `tho_device`, `tho_device_idx`, `device`, or `device_idx` for endpoint installation"))?;
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
            #common::ipc::codec::install_ipc_endpoint::<#codec, _>(
                #common::runner::global_ctx::GlobalCtx::global().cuda_actor(#device_param).as_ref(),
                #endpoint,
                &#init,
            ).await?;
        },
        None => quote! {
            #common::ipc::codec::install_empty_ipc_endpoint(
                #common::runner::global_ctx::GlobalCtx::global().cuda_actor(#device_param).as_ref(),
                #endpoint,
            ).await?;
        },
    };

    let run_call = match run_input {
        Some(run_input) => quote! {
            #common::ipc::codec::invoke_ipc_endpoint::<#codec, _, <Self as #common::runner::common::runnable::Stage>::Out>(
                #common::runner::global_ctx::GlobalCtx::global().cuda_actor(#common::runner::common::RunCtx::tho_device_idx(ctx)).as_ref(),
                #endpoint,
                &#run_input,
            ).await
        },
        None => quote! {
            #common::ipc::codec::invoke_ipc_sync_stage::<#codec, _, <Self as #common::runner::common::runnable::Stage>::Out>(
                #common::runner::global_ctx::GlobalCtx::global().cuda_actor(#common::runner::common::RunCtx::tho_device_idx(ctx)).as_ref(),
                #endpoint,
                input,
                ctx,
            ).await
        },
    };

    let endpoint_init_tokens = match (endpoint_init, worker_body) {
        (Some(endpoint_init), Some(worker_body)) => {
            let stage_init = match (init_type.as_ref(), init_stage.as_ref()) {
                (Some(init_type), Some(init_stage)) => quote! {
                    let init = <#codec as #common::ipc::codec::IpcCodec>::decode::<#init_type>(&init_msg)?;
                    let stage = #init_stage;
                },
                (None, None) => quote! {
                    let stage = #base_ident::new();
                },
                _ => return Err(syn::Error::new_spanned(&base_ident, "`worker_body` endpoint init must provide both `init_type` and `init_stage`, or neither")),
            };
            quote! {
                #[derive(Debug)]
                struct #wrapper_name {
                    stage: std::sync::Arc<#base_ty>,
                }

                impl #common::ipc::json_ser::SyncStageIpcWrapper for #wrapper_name {
                    fn do_run(
                        &self,
                        input: &[u8],
                        all_cuda_tensors: &std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<u64, tch::Tensor>>>,
                    ) -> Result<Vec<u8>, thoosa_shared::ThoError> {
                        type __ProxyIpcCodec = #codec;
                        #worker_body
                    }
                }

                pub fn #endpoint_init(init_msg: Vec<u8>) -> Result<Box<dyn #common::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError> {
                    #stage_init
                    Ok(Box::new(#wrapper_name {
                        stage: std::sync::Arc::new(stage),
                    }))
                }
            }
        },
        (Some(endpoint_init), None) => {
            let init_type = init_type.ok_or_else(|| syn::Error::new_spanned(&base_ident, "`endpoint_init` without `worker_body` requires `init_type = \"...\"`"))?;
            let init_stage = init_stage.ok_or_else(|| syn::Error::new_spanned(&base_ident, "`endpoint_init` without `worker_body` requires `init_stage = \"...\"`"))?;
            let in_type = in_type.unwrap_or_else(|| syn::parse_quote!(<#base_ty as #common::runner::common::runnable::Stage>::In<'static>));
            quote! {
                pub fn #endpoint_init(init_msg: Vec<u8>) -> Result<Box<dyn #common::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError> {
                    let init = <#codec as #common::ipc::codec::IpcCodec>::decode::<#init_type>(&init_msg)?;
                    Ok(#common::ipc::json_ser::make_ipc_sync_stage_wrapper::<
                        #codec,
                        #base_ty,
                        #in_type,
                        <#base_ty as #common::runner::common::runnable::Stage>::Out,
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
            pub fn #method(&self) -> usize {
                self.#method.load(std::sync::atomic::Ordering::Acquire)
            }
        },
        None => quote! {},
    };

    Ok(quote! {
        #[derive(Debug)]
        pub struct #proxy_ident {
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

        impl #common::runner::common::OutId for #proxy_ident {
            fn out_id<'a>(
                &self,
                input_id: &<<Self as #common::runner::common::runnable::Stage>::In<'a> as thoosa_shared::traits::WithId>::ID,
            ) -> <<Self as #common::runner::common::runnable::Stage>::Out as thoosa_shared::traits::WithId>::ID {
                #out_id
            }
        }

        impl #common::runner::common::runnable::Stage for #proxy_ident {
            type Ctx = <#base_ty as #common::runner::common::runnable::Stage>::Ctx;
            type In<'a> = <#base_ty as #common::runner::common::runnable::Stage>::In<'a>;
            type Out = <#base_ty as #common::runner::common::runnable::Stage>::Out;
        }

        impl #common::runner::common::runnable::SyncStage for #proxy_ident {
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
    common: Option<Path>,
    base_type: Option<Type>,
    endpoint: Option<Expr>,
    codec: Option<Path>,
    new_args: Option<Punctuated<FnArg, Token![,]>>,
    prepare: Option<proc_macro2::TokenStream>,
    init: Option<Expr>,
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
                    "common" => out.common = Some(lit.parse()?),
                    "base" => out.base_type = Some(lit.parse()?),
                    "endpoint" => out.endpoint = Some(lit.parse()?),
                    "codec" => out.codec = Some(lit.parse()?),
                    "new_args" => out.new_args = Some(parse_fn_args(&lit)?),
                    "prepare" => out.prepare = Some(parse_tokens(&lit)?),
                    "init" => out.init = Some(lit.parse()?),
                    "device" => return Err(meta.error("`device` is no longer supported; add a constructor argument named `tho_device`, `tho_device_idx`, `device`, or `device_idx`")),
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

fn proxy_device_param(new_args: &Punctuated<FnArg, Token![,]>) -> Option<Ident> {
    const DEVICE_PARAM_NAMES: [&str; 4] = ["tho_device", "tho_device_idx", "device", "device_idx"];

    DEVICE_PARAM_NAMES.iter().find_map(|target| {
        new_args.iter().find_map(|arg| match arg {
            FnArg::Typed(arg) => match arg.pat.as_ref() {
                Pat::Ident(pat_ident) if pat_ident.ident == *target => Some(pat_ident.ident.clone()),
                _ => None,
            },
            FnArg::Receiver(_) => None,
        })
    })
}
