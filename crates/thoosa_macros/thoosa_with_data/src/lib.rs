use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{bracketed, parse_macro_input, Expr, ExprClosure, Ident, Result, Token};

struct WithDataInput {
    ctx: Expr,
    args: Vec<Arg>,
    closure: ExprClosure,
}

enum Arg {
    Single(Expr),
    Vector(Ident),
}

impl Parse for WithDataInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let ctx = input.parse::<Expr>()?;
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

        Ok(Self { ctx, args, closure })
    }
}

#[proc_macro]
pub fn with_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as WithDataInput);
    expand_with_data(input, false).into()
}

#[proc_macro]
pub fn async_with_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as WithDataInput);
    expand_with_data(input, true).into()
}

fn expand_with_data(input: WithDataInput, await_closure: bool) -> proc_macro2::TokenStream {
    let ctx_expr = input.ctx;
    let closure = input.closure;

    let mut prelude = Vec::new();
    let mut load_steps = Vec::new();
    let mut arg_refs = Vec::new();

    for (idx, arg) in input.args.into_iter().enumerate() {
        match arg {
            Arg::Single(expr) => {
                let var_ident = format_ident!("__with_data_var_{idx}");
                let guard_ident = format_ident!("__with_data_guard_{idx}");
                let ref_ident = format_ident!("__with_data_ref_{idx}");

                prelude.push(quote! {
                    let #var_ident = ::std::sync::Arc::clone(&(#expr));
                });

                load_steps.push(quote! {
                    let #guard_ident = #var_ident.load_from_any_device_(__device_idx).await?;
                    let #ref_ident = &*#guard_ident;
                });

                arg_refs.push(quote! { #ref_ident });
            }
            Arg::Vector(ident) => {
                let vec_ident = format_ident!("__with_data_vec_{idx}");
                let guards_ident = format_ident!("__with_data_vec_guards_{idx}");
                let refs_ident = format_ident!("__with_data_vec_refs_{idx}");

                prelude.push(quote! {
                    let #vec_ident = (#ident).clone();
                });

                load_steps.push(quote! {
                    let #guards_ident = ::futures_util::future::try_join_all(
                        #vec_ident.iter().map(|wrap| wrap.load_from_any_device_(__device_idx))
                    ).await?;
                    let #refs_ident: ::std::vec::Vec<_> = #guards_ident.iter().map(|item| &**item).collect();
                });

                arg_refs.push(quote! { #refs_ident.as_slice() });
            }
        }
    }

    let invoke_closure = if await_closure {
        quote! { (#closure)(__with_data_ctx #(, #arg_refs)*).await }
    } else {
        quote! { (#closure)(__with_data_ctx #(, #arg_refs)*) }
    };

    let permits = if await_closure {
        quote! {
            let permit = if __with_data_ctx.acquired_permits() == 0 {
                Some(thoosa_shared::runner::global_ctx::GlobalCtx::global().acquire_one(__device_idx, ::std::sync::Arc::new(__with_data_ctx.id().clone())).await?)
    //             __with_data_ctx.add_permit(permit);
            } else {
                None
            };
        }
    } else {
        quote! {
            let permit = if __with_data_ctx.acquired_permits() == 0 {
                Some(thoosa_shared::runner::global_ctx::GlobalCtx::global().acquire_one(__device_idx, ::std::sync::Arc::new(__with_data_ctx.id().clone())).await?)
    //             __with_data_ctx.add_permit(permit);
            } else {
                None
            };
        }
    };
    quote! {
        {
            use thoosa_shared::traits::WithId;
            use thoosa_shared::runner::common::PermitsCtx;

            let mut __with_data_ctx = #ctx_expr;
            let __device_idx = __with_data_ctx.device_idx();
            let r = async {
                #(#prelude)*
                #permits
                #(#load_steps)*

                #invoke_closure
            }.await;

            r
        }
    }
}
