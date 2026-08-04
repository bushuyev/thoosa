use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Attribute, Data, DeriveInput, Expr, Fields, FnArg, Ident, Pat, Path, Result as SynResult, Token, Type, parse_macro_input};

#[proc_macro_derive(Proxy, attributes(proxy))]
pub fn derive_proxy(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_proxy(input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_proxy(input: DeriveInput) -> SynResult<proc_macro2::TokenStream> {
    let base_ty = proxy_base_type(&input)?;
    let base_ident = input.ident;
    let attrs = ProxyAttrs::from_attrs(&input.attrs)?;
    let proxy_ident = attrs.proxy_name.ok_or_else(|| syn::Error::new_spanned(&base_ident, "missing `name = \"ProxyType\"` in `proxy` attribute"))?;
    let common = attrs.common.unwrap_or_else(|| syn::parse_quote!(crate));
    let endpoint = attrs.endpoint.ok_or_else(|| syn::Error::new_spanned(&base_ident, "missing `endpoint = \"...\"` in `proxy` attribute"))?;
    let codec = attrs.codec.map(|codec| quote!(#codec)).unwrap_or_else(|| quote!(#common::ipc::codec::JsonIpcCodec));
    let new_args = attrs.new_args.unwrap_or_default();
    let device_param = proxy_device_param(&new_args).ok_or_else(|| syn::Error::new_spanned(&base_ident, "proxy constructor arguments must include one of `tho_device`, `tho_device_idx`, `device`, or `device_idx` for endpoint installation"))?;
    let fields = attrs.fields.unwrap_or_default();
    let out_id = attrs.out_id; 
    let prepare = attrs.prepare.unwrap_or_default();
    let pre_run = attrs.pre_run.unwrap_or_default();
    let run_input = attrs.run_input;
    let endpoint_init = attrs.endpoint_init;
    let init_type = attrs.init_type;
    let in_type = attrs.in_type;
    let init = attrs.init;
    let init_stage = attrs.init_stage;
    let wrapper_name = attrs.wrapper_name;
    let counter_method = attrs.counter_method;
    let consume = attrs.consume;

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

    let run_call = match run_input.as_ref() {
        Some(run_input) => quote! {
            #common::ipc::codec::invoke_ipc_endpoint::<#codec, _, <Self as thoosa_shared::traits::Stage>::Out>(

                #common::runner::global_ctx::GlobalCtx::global().cuda_actor(thoosa_shared::traits::RunCtx::tho_device_idx(ctx)).as_ref(),

                #endpoint,

                &#run_input,

            ).await
        },
        None => quote! {
            #common::ipc::codec::invoke_ipc_sync_stage::<#codec, _, <Self as thoosa_shared::traits::Stage>::Out>(

                #common::runner::global_ctx::GlobalCtx::global().cuda_actor(thoosa_shared::traits::RunCtx::tho_device_idx(ctx)).as_ref(),

                #endpoint,

                input,

            ).await
        },
    };

    let consume_call = match run_input.as_ref() {
        Some(run_input) => quote! {
            #common::ipc::codec::consume_ipc_endpoint::<#codec, _, <Self as thoosa_shared::traits::Stage>::Out>(
                #common::runner::global_ctx::GlobalCtx::global().cuda_actor(thoosa_shared::traits::RunCtx::tho_device_idx(&ctx)).as_ref(),

                #endpoint,

                &#run_input,
            ).await
        },
        None => quote! {
            #common::ipc::codec::consume_ipc_sync_stage::<#codec, _, <Self as thoosa_shared::traits::Stage>::Out>(
                #common::runner::global_ctx::GlobalCtx::global().cuda_actor(thoosa_shared::traits::RunCtx::tho_device_idx(&ctx)).as_ref(),

                #endpoint,

                &input,

                &ctx,
            ).await
        },
    };

    let endpoint_init_tokens = match endpoint_init {
        Some(endpoint_init) => {
            let stage_init = match (init_type.as_ref(), init_stage.as_ref()) {
                (Some(init_type), Some(init_stage)) => quote! {
                    let init = <#codec as #common::ipc::codec::IpcCodec>::decode::<#init_type>(&init_msg)?;
                    let stage = #init_stage;
                },
                (None, None) => quote! {
                    let stage = #base_ident::new();
                },
                _ => return Err(syn::Error::new_spanned(&base_ident, "`endpoint_init` must provide both `init_type` and `init_stage`, or neither")),
            };

            match wrapper_name {
                Some(wrapper_name) => quote! {
                    pub fn #endpoint_init(init_msg: Vec<u8>,ws:Arc<parking_lot::RwLock<Box<dyn thoosa_common::ws::Workspace>>>) -> Result<Box<dyn #common::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError> {
                        #stage_init
                        Ok(Box::new(#wrapper_name {
                            stage: std::sync::Arc::new(stage),
                        }))
                    }
                },
                None => {
                    let init_type = init_type.ok_or_else(|| syn::Error::new_spanned(&base_ident, "`endpoint_init` without `wrapper` requires `init_type = \"...\"`"))?;
                    let init_stage = init_stage.ok_or_else(|| syn::Error::new_spanned(&base_ident, "`endpoint_init` without `wrapper` requires `init_stage = \"...\"`"))?;
                    let in_type = in_type.unwrap_or_else(|| syn::parse_quote!(<#base_ty as thoosa_shared::traits::Stage>::In<'static>));
                    quote! {
                        pub fn #endpoint_init(init_msg: Vec<u8>,ws:Arc<parking_lot::RwLock<Box<dyn thoosa_common::ws::Workspace>>>) -> Result<Box<dyn #common::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError> {
                            let init = <#codec as #common::ipc::codec::IpcCodec>::decode::<#init_type>(&init_msg)?;
                            Ok(#common::ipc::json_ser::make_ipc_sync_stage_wrapper::<
                                #codec,
                                #base_ty,
                                #in_type,
                                <#base_ty as thoosa_shared::traits::Stage>::Out,
                            >(
                                #endpoint,
                                #init_stage,
                            ))
                        }
                    }
                }
            }
        },
        None => quote! {},
    };

    let counter_method = match counter_method {
        Some(method) => quote! {
            pub fn #method(&self) -> usize {
                self.#method.load(std::sync::atomic::Ordering::Acquire)
            }
        },
        None => quote! {},
    };

    let sync_stage_impl = if consume {
        quote! {
            impl thoosa_shared::traits::SyncStage for #proxy_ident {
                fn consume<'a>(
                    self: std::sync::Arc<Self>,
                    input: Self::In<'a>,
                    ctx: Self::Ctx,
                ) -> thoosa_shared::traits::ConsumeResult<Self::In<'a>, Self::Out>
                where
                    Self: Sized,
                {
                    let result = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            #consume_call
                        })
                    });
                    match result {
                        Ok(output) => thoosa_shared::traits::ConsumeResult::Ok(output),
                        Err(error) => thoosa_shared::traits::ConsumeResult::Failed { input, error },
                    }
                }
            }
        }
    } else {
        quote! {
            impl thoosa_shared::traits::SyncStage for #proxy_ident {
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
        }
    };

    let out_id_impl = match out_id {

        Some(out_id) => {
            quote!{
                impl thoosa_shared::traits::OutId for #proxy_ident {
                    fn out_id<'a>(
                        &self,
                        input_id: &<<Self as thoosa_shared::traits::Stage>::In<'a> as thoosa_shared::traits::WithId>::ID,
                    ) -> <<Self as thoosa_shared::traits::Stage>::Out as thoosa_shared::traits::WithId>::ID {
                        #out_id
                    }
                }
            }
        }
        None => {
            quote!{}
        }
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

        #out_id_impl

        impl thoosa_shared::traits::Stage for #proxy_ident {
            type Ctx = <#base_ty as thoosa_shared::traits::Stage>::Ctx;
            type In<'a> = <#base_ty as thoosa_shared::traits::Stage>::In<'a>;
            type Out = <#base_ty as thoosa_shared::traits::Stage>::Out;
        }

        #sync_stage_impl

        #endpoint_init_tokens
    })
}

#[derive(Default)]
struct ProxyAttrs {
    proxy_name: Option<Ident>,
    common: Option<Path>,
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
    wrapper_name: Option<Ident>,
    counter_method: Option<Ident>,
    consume: bool,
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
                    "wrapper" => out.wrapper_name = Some(lit.parse()?),
                    "counter" => out.counter_method = Some(lit.parse()?),
                    "consume" => out.consume = lit.value().parse().map_err(|_| meta.error("`consume` must be `\"true\"` or `\"false\"`"))?,
                    _ => return Err(meta.error(format!("unsupported proxy attribute `{key}`"))),
                }
                Ok(())
            })?;
        }
        Ok(out)
    }
}

fn proxy_base_type(input: &DeriveInput) -> SynResult<Type> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&input.generics, "Proxy can only be derived for an unparameterized struct"));
    }

    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(input, "Proxy can only be derived for an unparameterized struct"));
    };

    if let Fields::Named(fields) = &data.fields
        && let Some(field) = fields.named.iter().find(|field| field.ident.as_ref().is_some_and(|ident| ident == "base"))
    {
        return Ok(field.ty.clone());
    }

    let ident = &input.ident;
    Ok(syn::parse_quote!(#ident))
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
    const DEVICE_PARAM_NAMES: [&str; 4] = ["tho_device", "tho_device_idx", "device", "device_idx"];//TODO remove "device", "device_idx"

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_base_type_uses_target_struct_without_base_field() {
        let input: DeriveInput = syn::parse_quote!(struct ConcreteStage;);

        let base_ty = proxy_base_type(&input).unwrap();

        assert_eq!(quote!(#base_ty).to_string(), "ConcreteStage");
    }

    #[test]
    fn test_proxy_base_type_uses_base_field_type() {
        let input: DeriveInput = syn::parse_quote! {
            struct ProxyTarget {
                base: GenericStage<ConcreteCtx>,
            }
        };

        let base_ty = proxy_base_type(&input).unwrap();

        assert_eq!(quote!(#base_ty).to_string(), "GenericStage < ConcreteCtx >");
    }

    #[test]
    fn test_proxy_base_type_rejects_parameterized_struct() {
        let input: DeriveInput = syn::parse_quote!(struct GenericStage<C>(C););

        let error = match proxy_base_type(&input) {
            Ok(_) => panic!("parameterized proxy target should be rejected"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("unparameterized struct"));
    }

}
