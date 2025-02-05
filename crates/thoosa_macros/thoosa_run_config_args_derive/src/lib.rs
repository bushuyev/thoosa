use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Attribute, DeriveInput, Field, Lit, Path, Variant};
use thoosa_macros_utils::*;

fn parse_arg_type(attr: &Attribute) -> Option<proc_macro2::Ident> {
    if !attr.path().is_ident("arg_type") {
        return None;
    }
    // Parse the tokens inside parentheses as a `Path` (e.g. `TrainLabelsRunConfig`)
    if let Ok(path) = attr.parse_args::<Path>() {
        eprintln!("parse_arg_type: path={path:?}");

        // If you know it's always a single ident (e.g. no module path),
        // you can either use `path.get_ident()` or the last segment in `path.segments`.
        return path.get_ident().cloned().or_else(|| {
            // If user wrote something like `some::module::MyRunConfig`,
            // we can return the last segment as an Ident
            path.segments.last().map(|seg| seg.ident.clone())
        });
    }

    None
}

fn parse_enum_arg(variant: &Variant) -> Option<proc_macro2::Ident> {
    if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &variant.fields {
        if let Some(syn::Field { ty: syn::Type::Path(type_path), .. }) = unnamed.first() {
            return type_path.path.segments.last().map(|segment| segment.ident.clone());
        }
    }

    None
}



#[proc_macro_derive(ThoosaArgs, attributes(run_config, arg_type))]
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
                        // eprintln!("field_name={field_name:?} field_type={field_type:?}");

                        let dynamic_long = quote! {
                            let dynamic_long = match prefix {
                               Some(p) => format!("{}.{}", p, #field_name),
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
            // eprintln!("data_struct={data_enum:?}");

            data_enum.variants.iter().map(|variant|{
                // eprintln!("variant={variant:?} ");

                let variant_name = &variant.ident.to_string();
                eprintln!("variant_name={variant_name} variant.fields={:?}", variant);
                // for attr in &variant.attrs {
                //     eprintln!("variant.fields.attr={:?}", attr);
                //     if attr.path().is_ident("arg_type") {
                //
                //         eprintln!("arg_type={:?}", parse_arg_type(attr));
                //     }
                //
                // }

                let (struct_type, is_top) = parse_enum_arg(variant).map(|v|(v, false)).unwrap_or_else(||variant.attrs
                    .iter()
                    .find(|attr|attr.path().is_ident("arg_type"))
                    .and_then(|attr|{
                        parse_arg_type(attr)
                    }).map(|v|(v, true)).expect("could not find arg_type"));

                let struct_name = struct_type.to_string().replace("_", "-");

                eprintln!("struct_name={struct_type:?} is_top={is_top}");

                //TODO use is_top in gen.
                quote! {


                    let new_prefix = match prefix.clone() {
                       Some(p) => format!("{}.{}", p, thoosa_shared::camel_to_snake(#struct_name)),
                       None => thoosa_shared::camel_to_snake(#struct_name),
                    };
                    let command_name_static: &'static str = if #is_top {
                        Box::leak(thoosa_shared::camel_to_snake(#variant_name).replace("_", "-").into_boxed_str())
                    } else {
                        Box::leak(format!("{}.{}", prefix.expect("no prefix"), thoosa_shared::camel_to_snake(#variant_name)).replace("_", "-").into_boxed_str())
                    };
                    debug!("command_name_static={command_name_static} variant_name={} is_top={}", #variant_name, #is_top);
                    let mut command = clap::Command::new(command_name_static);

                    let (sub_args_op, commands_op) = #struct_type::args(None);//if #is_top { None} else {Some(new_prefix.as_str())}
                    if let Some(sub_args) = sub_args_op {
                        command = command.args(sub_args.into_iter().map(|v|v.into()));
                    }
                    if let Some(sub_commands) = commands_op {
                        command = command.subcommands(sub_commands);
                    }

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
                debug!("ThoosaArgs for #target_struct_ident: prefix={prefix:?}");

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
                        // eprintln!("field_name={field_name:?} field_type={field_type:?}");

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
                            // eprintln!("run_config field={field:?} field_type={field_type:?}");
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
                    eprintln!("unnamed fields");
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