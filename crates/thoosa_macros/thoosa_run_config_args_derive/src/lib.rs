use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Attribute, DeriveInput, Field, Lit, Path, Variant};
use thoosa_macros_utils::*;
// fn parse_arg_type(attr: &Attribute) -> Option<proc_macro2::Ident> {
//     if !attr.path().is_ident("arg_type") {
//         return None;
//     }
//     // Parse the tokens inside parentheses as a `Path` (e.g. `TrainLabelsRunConfig`)
//     if let Ok(path) = attr.parse_args::<Path>() {
//         eprintln!("parse_arg_type: path={path:?}");
//
//         // If you know it's always a single ident (e.g. no module path),
//         // you can either use `path.get_ident()` or the last segment in `path.segments`.
//         return path.get_ident().cloned().or_else(|| {
//             // If user wrote something like `some::module::MyRunConfig`,
//             // we can return the last segment as an Ident
//             path.segments.last().map(|seg| seg.ident.clone())
//         });
//     }
//
//     None
// }
//
// fn parse_enum_arg(variant: &Variant) -> Option<proc_macro2::Ident> {
//     if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &variant.fields {
//         if let Some(syn::Field { ty: syn::Type::Path(type_path), .. }) = unnamed.first() {
//             return type_path.path.segments.last().map(|segment| segment.ident.clone());
//         }
//     }
//
//     None
// }



#[proc_macro_derive(ThoosaArgs, attributes(run_config))]
pub fn thoosa_args_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let target_struct_ident = &ast.ident;

    eprintln!("thoosa_args_derive: target_struct_ident={target_struct_ident:?}");

    let range_or_value_struct_ident = Ident::new(
        &format!("{}RangeOrValue", target_struct_ident),
        target_struct_ident.span()
    );
    let fields_enum_ident = Ident::new(
        &format!("{}Fields", target_struct_ident),
        target_struct_ident.span()
    );

    let (range_or_value_fields, enum_fields, add_range_or_value_calls, range_or_value_type) = match &ast.data {
        syn::Data::Struct(data_struct) => {
            match &data_struct.fields {

                syn::Fields::Named(named_fields) => {

                    let (range_or_value_fields, enum_fields, add_range_or_value_calls) = named_fields.named.iter().fold((vec![], vec![], vec![]), |mut a, field| {

                        let field_ident = field.ident.as_ref().unwrap();
                        let camel_field_ident = Ident::new(&thoosa_shared::snake_to_camel(&field_ident.to_string()), fields_enum_ident.span());
                        let field_type = field.ty.clone();


                        if is_primitive(&field_type) {
                            a.0.push(quote! {
                                #field_ident: RangeOrValue<#field_type>,
                            });
                            a.1.push(quote! {
                                #camel_field_ident,
                            });
                            a.2.push(quote! {
                                thoosa_shared::add_range_or_value::<#field_type, T>(append(#fields_enum_ident::#camel_field_ident), &self.#field_ident, const_params, grid_params, opt_params);
                            });

                        } else if field.attrs.iter().any(|attr|attr.path().is_ident("run_config")) {
                                let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", field_type.to_token_stream().to_string()), camel_field_ident.span());

                                a.0.push(quote! {
                                    #field_ident: #field_type_range_or_value,
                                });

                                let field_type_fields = Ident::new(&format!("{}Fields", field_type.to_token_stream().to_string()), camel_field_ident.span());
                                a.1.push(quote! {
                                    #camel_field_ident(#field_type_fields),
                                });
                                a.2.push(quote! {
                                    self.#field_ident.fill_run_configs(|v| append(#fields_enum_ident::#camel_field_ident(v)), const_params, grid_params, opt_params);
                                });

                        } else {

                            a.0.push(quote! {});
                            a.1.push(quote! {});
                            a.2.push(quote! {});

                        }

                        a
                    });

                    (
                    quote! { #(# range_or_value_fields)* },
                    quote! { #(# enum_fields)* },
                    quote! { #(# add_range_or_value_calls)* },
                    quote! {struct},
                    )
                },
                _ => {
                    println!("unnamed fields");
                    (
                    quote! {},
                    quote! {},
                    quote! {},
                    quote! {},
                    )
                }
            }
        },
        syn::Data::Enum(data_enum) => {
            let (range_or_value_fields, enum_fields, add_range_or_value_calls) = data_enum.variants.iter().fold((vec![], vec![], vec![]), |mut a, variant| {

                let variant_name = &variant.ident;





                if let syn::Fields::Unnamed(fields) = &variant.fields {
                    if fields.unnamed.len() == 1 {
                        let ty = &fields.unnamed.first().expect("no config type").ty;

                        let field_type_range_or_value = Ident::new(&format!("{}RangeOrValue", ty.to_token_stream().to_string()), variant.ident.span());

                        a.0.push(quote! {
                            #variant_name(#field_type_range_or_value),

                        });

                        let field_type_fields = Ident::new(&format!("{}Fields", ty.to_token_stream().to_string()), variant.ident.span());

                        a.1.push(quote! {
                            #variant_name(#field_type_fields),
                        });

                        a.2.push(quote! {
                            #range_or_value_struct_ident::#variant_name(v) => {
                                v.fill_run_configs(|v|append(#fields_enum_ident::#variant_name(v)), const_params, grid_params, opt_params);
                            }


                        });
                    }
                }


                // a.2.push(quote! {});

                a
            });

            (
                quote! { #(# range_or_value_fields)* },
                quote! { #(# enum_fields)* },
                quote! {  match self {#(# add_range_or_value_calls)* }},
                quote! {enum},

            )

        }
        _ => {
            // If not a struct, do nothing
            (
            quote! {},
            quote! {},
            quote! {},
            quote! {}
            )
        }
    };


    let gen = quote! {

        #[derive(Debug, serde::Deserialize, serde::Serialize)]
        #range_or_value_type #range_or_value_struct_ident {
            #range_or_value_fields
        }

        #[derive(Debug, Clone)]
        enum #fields_enum_ident {
            #enum_fields
        }

        impl #range_or_value_struct_ident {
            pub(crate) fn fill_run_configs<T:thoosa_shared::hyperparameter::FieldName, F>(&self, append: F, const_params: &mut Vec<thoosa_shared::val::NamedVal<T>>, grid_params: &mut Vec<thoosa_shared::val::NamedRange<T>>, opt_params: &mut Vec<thoosa_shared::hyperparameter::Hyperparameter<T>>)
                where F:Fn(#fields_enum_ident)->T{

                #add_range_or_value_calls
            }
        }
    };

    gen.into()
}
