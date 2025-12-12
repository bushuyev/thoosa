use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{bracketed, parse_macro_input, Expr, ExprClosure, Ident, Result, Token};

struct WithDataInput {
    permits_ctx: Expr,
    device: Expr,
    args: Vec<Arg>,
    closure: ExprClosure,
}

enum Arg {
    Single(Expr),
    Vector(Ident),
}

impl Parse for WithDataInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let permits_ctx = input.parse::<Expr>()?;
        input.parse::<Token![,]>()?;
        let device = input.parse::<Expr>()?;
        input.parse::<Token![,]>()?;

        let mut args = Vec::new();
        while !input.peek(Token![;]) {
            if input.peek(syn::token::Bracket) {
                let content;
                bracketed!(content in input);
                let ident = content.parse::<Ident>()?;
                args.push(Arg::Vector(ident));
            } else {
                args.push(Arg::Single(input.parse::<Expr>()?));
            }

            if input.peek(Token![,]) {
                let _ = input.parse::<Token![,]>()?;
            } else {
                break;
            }
        }

        input.parse::<Token![;]>()?;
        let closure = input.parse::<ExprClosure>()?;

        Ok(Self { permits_ctx, device, args, closure })
    }
}

#[proc_macro]
pub fn with_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as WithDataInput);
    expand_with_data(input).into()
}

fn expand_with_data(input: WithDataInput) -> proc_macro2::TokenStream {
    let permits_ctx_expr = input.permits_ctx;
    let device_expr = input.device;
    let closure = input.closure;

    let mut prelude = Vec::new();
    let mut load_handles = Vec::new();
    let mut load_handle_names = Vec::new();
    let mut load_handle_results = Vec::new();
    let mut post_load = Vec::new();
    let mut arg_refs = Vec::new();

    for (idx, arg) in input.args.into_iter().enumerate() {
        match arg {
            Arg::Single(expr) => {
                let var_ident = format_ident!("__with_data_var_{idx}");
                let load_handle_ident = format_ident!("__with_data_handle_{idx}");
                let load_result_ident = format_ident!("__with_data_handle_res_{idx}");
                let guard_ident = format_ident!("__with_data_guard_{idx}");
                let ref_ident = format_ident!("__with_data_ref_{idx}");

                prelude.push(quote! {
                    let #var_ident = ::std::sync::Arc::clone(&(#expr));
                });

                load_handles.push(quote! {
                    let #load_handle_ident = ::tokio::spawn({
                        let __wrap = ::std::sync::Arc::clone(&#var_ident);
                        let __device_idx = __device_idx;
                        async move {
                            __wrap.load_from_any_device(__device_idx).await?;
                            Ok::<(), ::thoosa_shared::ThoosaError>(())
                        }
                    });
                });

                load_handle_names.push(load_handle_ident);
                load_handle_results.push(load_result_ident.clone());

                post_load.push(quote! {
                    let #guard_ident = #var_ident.get_copied(__device_idx).await?;
                    let #ref_ident = &*#guard_ident;
                });

                arg_refs.push(quote! { #ref_ident });
            }
            Arg::Vector(ident) => {
                let vec_ident = format_ident!("__with_data_vec_{idx}");
                let vec_for_load_ident = format_ident!("__with_data_vec_load_{idx}");
                let load_handle_ident = format_ident!("__with_data_handle_{idx}");
                let load_result_ident = format_ident!("__with_data_handle_res_{idx}");
                let guards_ident = format_ident!("__with_data_vec_guards_{idx}");
                let refs_ident = format_ident!("__with_data_vec_refs_{idx}");

                prelude.push(quote! {
                    let #vec_ident = (#ident).clone();
                    let #vec_for_load_ident: ::std::vec::Vec<_> = #vec_ident
                        .iter()
                        .map(|wrap| ::std::sync::Arc::clone(wrap))
                        .collect();
                });

                load_handles.push(quote! {
                    let #load_handle_ident = ::tokio::spawn({
                        let __device_idx = __device_idx;
                        let __items = #vec_for_load_ident;
                        async move {
                            ::futures_util::future::try_join_all(
                                __items.into_iter().map(|wrap| async move {
                                    wrap.load_from_any_device(__device_idx).await
                                })
                            ).await?;
                            Ok::<(), ::thoosa_shared::ThoosaError>(())
                        }
                    });
                });

                load_handle_names.push(load_handle_ident);
                load_handle_results.push(load_result_ident.clone());

                post_load.push(quote! {
                    let #guards_ident = ::futures_util::future::try_join_all(
                        #vec_ident.iter().map(|wrap| wrap.get_copied(__device_idx))
                    ).await?;
                    let #refs_ident: ::std::vec::Vec<_> = #guards_ident.iter().map(|item| &**item).collect();
                });

                arg_refs.push(quote! { #refs_ident.as_slice() });
            }
        }
    }

    let await_blocks: Vec<_> = load_handle_names
        .iter()
        .zip(load_handle_results.iter())
        .map(|(handle, result)| {
            let handle_ident = handle.clone();
            let result_ident = result.clone();
            quote! {
                let #result_ident = #handle_ident.await?;
                #result_ident?;
            }
        })
        .collect();

    quote! {
        {
            let __with_data_permits_ctx = ::std::sync::Arc::clone(&(#permits_ctx_expr));
            let __with_data_device = #device_expr;
            async {
                let to_device_idx = __with_data_device;
                let __device_idx = to_device_idx;
                if __with_data_permits_ctx.read().acquired_permits() == 0 {
                    let name = __with_data_permits_ctx.read().name();
                    let result = caching_runner::global_ctx::GlobalCtx::global()
                        .acquire_many(to_device_idx, 1, ::std::sync::Arc::new(::thoosa_shared::traits::SimpleId(name)))
                        .await?;
                    __with_data_permits_ctx.write().add_permit(result);//permits can be dropped only from OOM handler or when context is dropped
                }
                #(#prelude)*
                #(#load_handles)*
                #(#await_blocks)*
                #(#post_load)*
                (#closure)(#(#arg_refs),*)
            }.await
        }
    }
}
