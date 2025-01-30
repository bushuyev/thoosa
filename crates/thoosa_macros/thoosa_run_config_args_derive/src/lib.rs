use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput, Field, Lit};
use thoosa_macros_utils::is_primitive;


#[proc_macro_derive(ThoosaArgs, attributes(run_config))]
pub fn thoosa_args_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let target_struct_ident = &ast.ident;

    let fields_code = match &ast.data {
        syn::Data::Struct(data_struct) => {
            match &data_struct.fields {

                syn::Fields::Named(named_fields) => {

                    let args_for_fields = named_fields.named.iter().map(|field| {

                        let field_ident = field.ident.as_ref().unwrap();
                        let field_name = field_ident.to_string().replace("_", "-");
                        let field_type = field.ty.clone();
                        eprintln!("field_name={field_name:?} field_type={field_type:?}");

                        let dynamic_long = quote! {
                             let dynamic_long = match prefix {
                                   Some(p) => format!("{}-{}", p, #field_name),
                                   None =>#field_name.to_string(),
                                };
                        };
                        if is_primitive(&field_type) {
                            let descr = comment(&field, &field_name);

                            quote! {
                                #dynamic_long

                                let long_static: &'static str = Box::leak(dynamic_long.into_boxed_str());
                                args.push(
                                    Arg::new(long_static)
                                    .long(long_static)
                                    .help(#descr)
                                    .value_parser(thoosa_shared::parse_range_or_value::<#field_type>)
                                    .action(clap::ArgAction::Set)
                                );

                            }
                        } else if field.attrs.iter().any(|attr|attr.path().is_ident("run_config")) {
                            quote! {
                                #dynamic_long
                                let (sub_args_op, commands_op) = #field_type::args(Some(dynamic_long.as_str()));
                                if let Some(sub_args) = sub_args_op {
                                    args.extend(sub_args.into_iter().map(|v|v.into()));
                                }
                                if let Some(sub_commands) = commands_op {
                                    commands.extend(sub_commands);
                                }

                            }
                        } else {
                            quote! {}
                        }
                    });

                    quote! { #(#args_for_fields)* }
                },
                _ => {
                    println!("unnamed fields");
                    quote! {

                    }
                }
            }
        },
        syn::Data::Enum(data_enum) => {
            eprintln!("data_struct={data_enum:?}");

            data_enum.variants.iter().map(|v|{
                eprintln!("variant={v:?}");

                quote! {

                    let command = clap::Command::new("test");

                    commands.push(command);
                }
            }).collect()

        }
        _ => {
            // If not a struct, do nothing
            quote! {}
        }
    };


    let gen = quote! {
        impl ThoosaArgs for #target_struct_ident {

            fn args(prefix: Option<&str>) -> (Option<Vec<impl Into<Arg>>>, Option<Vec<Command>>){
                let mut args:Vec<Arg> = vec![];
                let mut commands:Vec<Command> = vec![];

                #fields_code

                (
                    if args.is_empty() {
                        None
                    } else {
                        Some(args)
                    },

                    if commands.is_empty() {
                        None
                    } else {
                        Some(commands)
                    }
                )
            }
        }
    };

    gen.into()
}

#[proc_macro_derive(ThoosaArgsCommand, attributes(run_config))]
pub fn thoosa_args_command_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let target_struct_ident = &ast.ident;
    let target_struct_name = target_struct_ident.to_string();

    let fields_code = match &ast.data {

        syn::Data::Struct(data_struct) => {
            match &data_struct.fields {

                syn::Fields::Named(named_fields) => {

                    let args_for_fields = named_fields.named.iter().map(|field| {

                        let field_ident = field.ident.as_ref().unwrap();
                        let field_name = field_ident.to_string().replace("_", "-");
                        let field_type = field.ty.clone();
                        eprintln!("field_name={field_name:?} field_type={field_type:?}");

                        if is_primitive(&field_type) {
                            let descr = comment(&field, &field_name);

                            quote! {
                                command = command.arg(
                                    arg!(
                                        --#field_name
                                        #descr
                                    )
                                    .value_parser(thoosa_shared::parse_range_or_value::<#field_type>)
                                    .action(clap::ArgAction::Set)
                                );
                            }
                        } else if field.attrs.iter().any(|attr|attr.path().is_ident("run_config")) {
                            let field_type = field.ty.clone();
                            eprintln!("run_config field={field:?} field_type={field_type:?}");
                            quote! {

                                // command = command.args(#field_type::args(None));

                                let (sub_args_op, commands_op) = #field_type::args(None);
                                if let Some(sub_args) = sub_args_op {
                                    command = command.args(sub_args);
                                }
                                 if let Some(sub_commands) = commands_op {
                                    command = command.subcommands(sub_commands);
                                }
                            }
                        } else {
                            quote! {}
                        }
                    });

                    quote! { #(#args_for_fields)* }
                },
                _ => {
                    println!("unnamed fields");
                    // If struct has unnamed fields or other forms, do nothing
                    quote! {}
                }
            }
        },
        _ => {
            // If not a struct, do nothing
            quote! {}
        }
    };

    let gen = quote! {
        impl ThoosaArgsCommand for #target_struct_ident {
            fn command() -> clap::Command {
                let mut command = clap::Command::new(#target_struct_name);
                #fields_code

                command
            }
        }
    };

    gen.into()
}

fn comment(field: &&Field, field_name: &String) -> String {
    let mut descr = field_name.clone();
    for attr in &field.attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(meta) = &attr.meta {
                if let syn::Expr::Lit(lit_str) = &meta.value {
                    // let doc_comment = lit_str.lit.to_string();
                    if let Lit::Str(token) = &lit_str.lit {
                        descr = token.value();
                    }
                }
            }
        }
    }
    descr
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_it(){

    }
}
